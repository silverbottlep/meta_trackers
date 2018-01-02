import numpy as np
import os
import sys
import time
import argparse
import json
from PIL import Image
from matplotlib import pyplot as plt
from scipy import io

import torch
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable

sys.path.insert(0,'/home/eunbyung/Works2/src/tracking/meta_trackers/meta_sdnet/modules')
from sample_generator import *
from data_prov import *
from meta_model import *
from bbreg import *
from options import *

import vot

np.random.seed(int(time.time()))
torch.manual_seed(int(time.time()))
torch.cuda.manual_seed(int(time.time()))

def forward_samples(net, image, samples, out_layer=None):
    net.eval()
    extractor = RegionExtractor(image, samples, opts['img_size'], opts['padding'], opts['batch_test'])
    for i, regions in enumerate(extractor):
        regions = Variable(regions)
        if opts['use_gpu']:
            regions = regions.cuda()
        feat = net(regions, out_layer=out_layer)
        if i==0:
            feats = feat.data.clone()
        else:
            feats = torch.cat((feats,feat.data.clone()),0)
    return feats

def extract_regions(image, samples, crop_size, padding):
    regions = np.zeros((len(samples),crop_size,crop_size,3),dtype='uint8')
    for i, sample in enumerate(samples):
        regions[i] = crop_image(image, sample, crop_size, padding, True)
    
    regions = regions.transpose(0,3,1,2)
    regions = regions.astype('float32') - 128.
    return regions

def set_optimizer(model, lr_base, lr_mult, momentum=opts['momentum'], w_decay=opts['w_decay']):
    params = model.get_learnable_params()
    param_list = []
    for k, p in params.items():
        lr = lr_base
        for l, m in lr_mult.items():
            if k.startswith(l):
                lr = lr_base * m
        param_list.append({'params': [p], 'lr':lr})
    optimizer = optim.SGD(param_list, lr = lr, momentum=momentum, weight_decay=w_decay)
    #optimizer = optim.Adam(param_list)
    return optimizer

def train_init(init_net, meta_alpha, loss_fn, image, target_bbox, evaluator):
    init_net.train()
    
    # Draw pos/neg samples
    pos_examples = gen_samples(SampleGenerator('gaussian', image.size, 0.1, 1.2),
                               target_bbox, opts['n_pos_init'], opts['overlap_pos_init'])
    neg_examples = np.concatenate([
                    gen_samples(SampleGenerator('uniform', image.size, 1, 2, 1.1), 
                                target_bbox, opts['n_neg_init']//2, opts['overlap_neg_init']),
                    gen_samples(SampleGenerator('whole', image.size, 0, 1.2, 1.1),
                                target_bbox, opts['n_neg_init']//2, opts['overlap_neg_init'])])
    # Crop images
    crop_size = opts['img_size']
    padding = opts['padding']
    image = np.asarray(image)
    pos_regions = extract_regions(image, pos_examples, crop_size, padding)
    neg_regions = extract_regions(image, neg_examples, crop_size, padding)
    pos_regions_var = Variable(torch.from_numpy(pos_regions[:opts['batch_pos']]))
    neg_regions_var = Variable(torch.from_numpy(neg_regions[:opts['batch_neg']]))
    if opts['use_gpu']:
        pos_regions_var = pos_regions_var.cuda()
        neg_regions_var = neg_regions_var.cuda()
    
    # training
    tracker_init_weights = OrderedDict((name, param) for (name, param) in init_net.named_parameters())
    tracker_keys = [name for (name, _) in init_net.named_parameters()]
    # the first iteration
    pos_score = init_net.forward(pos_regions_var)
    neg_score = init_net.forward(neg_regions_var)
    init_loss = loss_fn(pos_score,neg_score)
    init_acc,init_acc_pos,init_acc_neg = evaluator(pos_score, neg_score)
    grads = torch.autograd.grad(init_loss, tracker_init_weights.values(), create_graph=True)
    tracker_weights = OrderedDict((name, param - torch.mul(alpha,grad)) for
                                  ((name, param),(_,alpha),grad) in
                                  zip(tracker_init_weights.items(),
                                      meta_alpha.items(), grads))
    # rest of iterations
    for i in range(opts['n_init_updates']-1):
        pos_score = init_net.forward(pos_regions_var, tracker_weights)
        neg_score = init_net.forward(neg_regions_var, tracker_weights)
        loss = loss_fn(pos_score,neg_score)
        grads = torch.autograd.grad(loss, tracker_weights.values(), create_graph=True)
        tracker_weights = OrderedDict((name, param - torch.mul(alpha,grad))
                                      for ((name, param),(_,alpha),grad) in
                                      zip(tracker_weights.items(),meta_alpha.items(), grads))
    # update tracker
    init_net.copy_meta_weights(tracker_weights)
    init_net.eval()
    pos_score = init_net.forward(pos_regions_var)
    neg_score = init_net.forward(neg_regions_var)
    acc,acc_pos,acc_neg = evaluator(pos_score, neg_score)

    pos_regions_var = Variable(torch.from_numpy(pos_regions))
    neg_regions_var = Variable(torch.from_numpy(neg_regions))
    if opts['use_gpu']:
        pos_regions_var = pos_regions_var.cuda()
        neg_regions_var = neg_regions_var.cuda()
    pos_feats = init_net(pos_regions_var, out_layer='features')
    neg_feats = init_net(neg_regions_var, out_layer='features')
    return pos_feats.data.clone(), neg_feats.data.clone(), init_acc, acc

def train_online(online_net, online_optimizer, loss_fn, pos_feats, neg_feats, maxiter, in_layer='fc'):
    online_net.train()
    
    batch_pos = opts['batch_pos']
    batch_neg = opts['batch_neg']
    batch_test = opts['batch_test']
    batch_neg_cand = max(opts['batch_neg_cand'], batch_neg)

    pos_idx = np.random.permutation(pos_feats.size(0))
    neg_idx = np.random.permutation(neg_feats.size(0))
    while(len(pos_idx) < batch_pos*maxiter):
        pos_idx = np.concatenate([pos_idx, np.random.permutation(pos_feats.size(0))])
    while(len(neg_idx) < batch_neg_cand*maxiter):
        neg_idx = np.concatenate([neg_idx, np.random.permutation(neg_feats.size(0))])
    pos_pointer = 0
    neg_pointer = 0

    for iter in range(maxiter):
        # select pos idx
        pos_next = pos_pointer+batch_pos
        pos_cur_idx = pos_idx[pos_pointer:pos_next]
        pos_cur_idx = pos_feats.new(pos_cur_idx.astype(float)).long()
        pos_pointer = pos_next
        # select neg idx
        neg_next = neg_pointer+batch_neg_cand
        neg_cur_idx = neg_idx[neg_pointer:neg_next]
        neg_cur_idx = neg_feats.new(neg_cur_idx.astype(float)).long()
        neg_pointer = neg_next
        # create batch
        batch_pos_feats = Variable(pos_feats.index_select(0, pos_cur_idx))
        batch_neg_feats = Variable(neg_feats.index_select(0, neg_cur_idx))
        # hard negative mining
        if batch_neg_cand > batch_neg:
            online_net.eval()
            for start in range(0,batch_neg_cand,batch_test):
                end = min(start+batch_test,batch_neg_cand)
                score = online_net.forward(batch_neg_feats[start:end], in_layer=in_layer)
                if start==0:
                    neg_cand_score = score.data[:,1].clone()
                else:
                    neg_cand_score = torch.cat((neg_cand_score, score.data[:,1].clone()),0)

            _, top_idx = neg_cand_score.topk(batch_neg)
            batch_neg_feats = batch_neg_feats.index_select(0, Variable(top_idx))
            online_net.train()
        # forward
        pos_score = online_net.forward(batch_pos_feats, in_layer=in_layer)
        neg_score = online_net.forward(batch_neg_feats, in_layer=in_layer)
        loss = loss_fn(pos_score,neg_score)
        online_net.zero_grad() 
        loss.backward()
        torch.nn.utils.clip_grad_norm(online_net.parameters(), opts['grad_clip'])
        online_optimizer.step()
    
    # testing
    pos_score = online_net.forward(batch_pos_feats, in_layer=in_layer)
    neg_score = online_net.forward(batch_neg_feats, in_layer=in_layer)
    after_loss = loss_fn(pos_score,neg_score)
    #print("loss:%.4f, after loss:%.4f"%(loss.data[0], after_loss.data[0]))


def run_meta_tracker():
    handle = vot.VOT("rectangle")
    selection = handle.region()
    imagefile = handle.frame()
    if not imagefile:
        sys.exit(0)

    # Init bbox
    target_bbox = np.array([selection.x, selection.y, selection.width, selection.height])
    prev_result = target_bbox
    prev_result_bb = target_bbox

    ## Initialize Tracker Net ##
    init_net = MetaSDNet(opts['init_model_path'])
    if opts['use_gpu']:
        init_net = init_net.cuda()
    init_net.set_learnable_params(['features','fc'])
    states = torch.load(opts['meta_tracker_path'])
    meta_init = states['meta_init']
    meta_alpha = states['meta_alpha']
    init_net.copy_meta_weights(meta_init)
    
    # Init criterion
    criterion = BinaryLoss()
    evaluator = Accuracy() # or Precision

    # Load first image1
    image1 = Image.open(imagefile).convert('RGB')
    
    # Train bbox regressor
    bbreg_examples = gen_samples(SampleGenerator('uniform', image1.size, 0.3, 1.5, 1.1),
                                 target_bbox, opts['n_bbreg'], opts['overlap_bbreg'], opts['scale_bbreg'])
    bbreg_feats = forward_samples(init_net, image1, bbreg_examples, out_layer='features')
    bbreg = BBRegressor(image1.size)
    bbreg.train(bbreg_feats, bbreg_examples, target_bbox)

    # Initial training(for the first frame)
    pos_feats, neg_feats, init_acc, first_acc = train_init(init_net, meta_alpha, criterion, image1, target_bbox, evaluator)
    init_params = OrderedDict()
    for k, p in init_net.named_parameters():
            init_params[k] = p
    online_net = MetaSDNetOnline()
    online_net.copy_meta_weights(init_params)
    online_net.set_learnable_params(['fc'])
    if opts['use_gpu']:
        online_net = online_net.cuda()
    online_optimizer = set_optimizer(online_net, opts['lr_update'], opts['lr_mult'])
    #print("\t[Init] init_acc:%.4f, acc:%.4f\n"%(init_acc, first_acc))
    
    # Init sample generators
    sample_generator = SampleGenerator('gaussian', image1.size, opts['trans_f'], opts['scale_f'], valid=True)
    pos_generator = SampleGenerator('gaussian', image1.size, 0.1, 1.2)
    neg_generator = SampleGenerator('uniform', image1.size, 1.5, 1.2)

    # Init pos/neg features for update
    pos_feats_all = [pos_feats]
    neg_feats_all = [neg_feats]

    # Main loop
    iters = 1
    while True:
        # Load image
        imagefile = handle.frame()
        if not imagefile:
            break
        image = Image.open(imagefile).convert('RGB')

        # Estimate target bbox,
        # Using top-scored 5 bboxes, average bbox # regression
        samples = gen_samples(sample_generator, target_bbox, opts['n_samples'])
        sample_scores = forward_samples(online_net, image, samples)
        top_scores, top_idx = sample_scores[:,1].topk(5)
        top_idx = top_idx.cpu().numpy()
        target_score = top_scores.mean()
        target_bbox = samples[top_idx].mean(axis=0)

        success = target_score > opts['success_thr']
        
        # Expand search area at failure
        if success:
            sample_generator.set_trans_f(opts['trans_f'])
        else:
            sample_generator.set_trans_f(opts['trans_f_expand'])

        # Bbox regression
        if success:
            bbreg_samples = samples[top_idx]
            bbreg_feats = forward_samples(online_net, image, bbreg_samples, out_layer='features')
            bbreg_samples = bbreg.predict(bbreg_feats, bbreg_samples)
            bbreg_bbox = bbreg_samples.mean(axis=0)
        else:
            bbreg_bbox = target_bbox
        
        # Copy previous result at failure
        if not success:
            target_bbox = prev_result
            bbreg_bbox = prev_result_bb
        
        # Save result
        prev_result = target_bbox
        prev_result_bb = bbreg_bbox
        vot_result = vot.Rectangle(bbreg_bbox[0],bbreg_bbox[1],bbreg_bbox[2],bbreg_bbox[3])
        handle.report(vot_result)

        # Data collect
        if success:
            # Draw pos/neg samples
            pos_examples = gen_samples(pos_generator, target_bbox, 
                                       opts['n_pos_update'],
                                       opts['overlap_pos_update'])
            neg_examples = gen_samples(neg_generator, target_bbox, 
                                       opts['n_neg_update'],
                                       opts['overlap_neg_update'])
            # Extract pos/neg features
            pos_feats = forward_samples(online_net, image, pos_examples, out_layer='features')
            neg_feats = forward_samples(online_net, image, neg_examples, out_layer='features')
            pos_feats_all.append(pos_feats)
            neg_feats_all.append(neg_feats)
            if len(pos_feats_all) > opts['n_frames_long']:
                del pos_feats_all[0]
            if len(neg_feats_all) > opts['n_frames_short']:
                del neg_feats_all[0]

        # Short term update
        if not success:
            #print("Short term update!")
            nframes = min(opts['n_frames_short'],len(pos_feats_all))
            pos_feats_short = pos_feats_all[-nframes:]
            pos_data = pos_feats_short[0]
            for pos_feat in pos_feats_short[1:]:
                pos_data = torch.cat([pos_data,pos_feat],0)
            #neg_feats_short = neg_feats_all
            neg_feats_short = neg_feats_all[-nframes:]
            neg_data = neg_feats_short[0]
            for neg_feat in neg_feats_short[1:]:
                neg_data = torch.cat([neg_data,neg_feat],0)
            train_online(online_net, online_optimizer, criterion, pos_data, neg_data, opts['n_tracker_updates'])
        # Long term update
        elif iters % opts['long_interval'] == 0:
            #print("Long term update!")
            pos_data = pos_feats_all[0]
            for pos_feat in pos_feats_all[1:]:
                pos_data = torch.cat([pos_data,pos_feat],0)
            neg_data = neg_feats_all[0]
            for neg_feat in neg_feats_all[1:]:
                neg_data = torch.cat([neg_data,neg_feat],0)
            train_online(online_net, online_optimizer, criterion, pos_data, neg_data, opts['n_tracker_updates'])
        iters = iters+1
        
    return prev_result, prev_result_bb 

    
opts['lr_mult'] = {'fc.fc6':10}
opts['n_init_updates'] = 1
opts['n_tracker_updates'] = 15
opts['grad_clip'] = 1000
opts['lr_update'] = 0.00005
opts['init_model_path'] = '/home/eunbyung/Works2/src/tracking/meta_trackers/meta_sdnet/models/imagenet-vgg-m.mat'
opts['meta_tracker_path'] = '/home/eunbyung/Works2/src/tracking/meta_trackers/meta_sdnet/models/meta_init_otb_ilsvrc.pth'
   
# Run tracker
result, result_bb = run_meta_tracker()
