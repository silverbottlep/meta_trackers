import numpy as np
import os
import sys
import time
import argparse
import json
from PIL import Image
from matplotlib import pyplot as plt
from scipy import io
from sklearn.decomposition import PCA
import cv2

import torch
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable

sys.path.insert(0,'/home/eunbyung/Works2/src/tracking/meta_trackers/meta_crest')
from model import *
from utils import *

import vot

def model_update(tracker_net, online_optimizer, loss_fn, feat_update, label_update):
    # prepare index for random batch
    n_total = n_online_updates*n_update_batch
    n_cur_db = feat_update.size(0)
    if n_cur_db < n_total:
        indices = torch.from_numpy(np.random.permutation(n_cur_db))
        indices = indices.clone().repeat(int((n_total/n_cur_db)+1))
    else:
        indices = torch.from_numpy(np.random.permutation(n_total))
    if use_gpu:
        indices = indices.cuda()

    for it in range(n_online_updates):
        feat_update_batch = feat_update.index_select(0,indices[(it)*n_update_batch:(it+1)*n_update_batch])
        label_update_batch = label_update.index_select(0,indices[(it)*n_update_batch:(it+1)*n_update_batch])
        feat_update_batch = Variable(feat_update_batch)
        label_update_batch = Variable(label_update_batch)

        response = tracker_net.forward(feat_update_batch)
        loss = loss_fn(response,label_update_batch)
        tracker_net.zero_grad()
        loss.backward()
        online_optimizer.step()
        #print('loss: %.4f'%(loss.data[0]))

def train_init(meta_init, meta_alpha, target_cell_sz, loss_fn, feat, label_map):
    rw = int(np.ceil(target_cell_sz[0]/2))
    rh = int(np.ceil(target_cell_sz[1]/2))
    fw = int(rw*2 + 1)
    fh = int(rh*2 + 1)
    #print('cf_size(h,w): (18,12) -> (%d,%d)'%(fh,fw))

    # bilinear sampling for resizing correlation filters 
    cf_weight = OrderedDict()
    for k,p in meta_init.items():
        if 'cf.weight' in k:
            cf_weight[k] = bilinear_upsample(p, (fh, fw))
        else:
            cf_weight[k] = 1*p 
    cf_alpha = OrderedDict()
    for k,p in meta_alpha.items():
        if 'cf.weight' in k:
            cf_alpha[k] = bilinear_upsample(p, (fh, fw))
        else:
            cf_alpha[k] = 1*p
    for it in range(n_init_updates):
        temp = conv2d(feat, cf_weight['conv1.weight'], cf_weight['conv1.bias'], stride=1,padding=0)
        response = conv2d(temp, cf_weight['cf.weight'], cf_weight['cf.bias'], stride=1,padding=(rh,rw))
        loss = loss_fn(response,label_map)
        grads = torch.autograd.grad(loss, cf_weight.values(), create_graph=True)
        cf_weight = OrderedDict((name, param - torch.mul(alpha,grad))
                                      for ((name, param),(_,alpha),grad) in
                                      zip(cf_weight.items(),cf_alpha.items(), grads))
        #print('loss: %.4f'%(loss.data[0]))
    #temp = conv2d(feat, cf_weight['conv1.weight'], cf_weight['conv1.bias'], stride=1,padding=0)
    #response = conv2d(temp, cf_weight['cf.weight'], cf_weight['cf.bias'], stride=1,padding=(rh,rw))
    #loss = loss_fn(response,label_map)
    #print('loss: %.4f'%(loss.data[0]))
    #plt.imshow(response[0][0].data.cpu().numpy()); plt.colorbar(); plt.show()
    
    tracker_net = CFConv1NetOnline(fh, fw, rh,rw, vgg_num_channels, cf_num_channels)
    tracker_net.copy_meta_weights(cf_weight)
    return tracker_net
    

def scale_estimation(image, feature_net, tracker_net, \
                     target_sz, target_center, window_sz, cos_window):
    scales = [1, 0.95, 1.05]
    scores = np.zeros(len(scales))
    for j in range(len(scales)):
        scale = scales[j]
        scaled_target_sz = target_sz*scale
        scaled_window_sz = get_search_size(scaled_target_sz)
        patch = get_search_patch(image, target_center, scaled_window_sz)
        patch = cv2.resize(patch.transpose(1,2,0).astype('float32'),dsize=(int(window_sz),int(window_sz)))
        patch = patch.transpose(2,0,1)
        patch = Variable(torch.from_numpy(patch[None,:]))
        if use_gpu:
            patch = patch.cuda()
        feat = feature_net.forward(patch)
        feat = resize_feat_cos_window(feat, cos_window)
        
        response = tracker_net.forward(feat)

        scores[j] = response.max().data[0]
        if scores[0] < 0.15:
            return target_sz
    idx = scores.argmax()
    return 0.4*target_sz + 0.6*np.round(target_sz*scales[idx])

# resizing features as same size as cos_window
def resize_feat_cos_window(feat, cos_window):
    num_channels = feat.size(1)
    cos_window_sz = cos_window.size(3)
    feat_data = feat.data.clone().cpu().numpy()
    feat = cv2.resize(feat_data[0].transpose(1,2,0).astype('float32'),dsize=(cos_window_sz,cos_window_sz))
    feat = feat.transpose(2,0,1)
    feat = Variable(torch.from_numpy(feat).float(),requires_grad=False)
    if use_gpu:
        feat = feat.cuda()
    feat = feat*cos_window
    return feat
    
def set_optimizer(model, lr_base, w_decay):
    param_list = []
    for k,p in model.named_parameters():
        param_list.append({'params': [p], 'lr':lr_base, 'weight_decay':w_decay})
    optimizer = optim.Adam(param_list)
    return optimizer

def run_crest():
    handle = vot.VOT("rectangle")
    selection = handle.region()
    imagefile = handle.frame()
    if not imagefile:
        sys.exit(0)

    # Generate sequence config
    max_object_sz = 100

    target_bbox1 = np.array([selection.x, selection.y, selection.width, selection.height])

    ## initialize feature extractor ##
    feature_net = FeatureExtractor(feat_extractor_path)
    if use_gpu:
        feature_net = feature_net.cuda()
    feature_net.eval()

    ## resize target object size - computational issue ##
    scale = 1
    if target_bbox1[2:].prod() > max_object_sz**2:
        scale = max_object_sz/(target_bbox1[2:].max())

    # load first image1
    image1 = Image.open(imagefile).convert('RGB')
    image_sz_resized = (int(image1.size[0]*scale), int(image1.size[1]*scale))
    image1_resized = image1.resize(image_sz_resized,Image.BILINEAR)
    
    # scale target bbox
    resized_target_bbox1 = np.round(target_bbox1*scale)
    target_sz1 = resized_target_bbox1[2:]
    target_center1 = resized_target_bbox1[:2] + np.floor(target_sz1/2)

    # get cosine window(square shape)
    window_sz = get_search_size(target_sz1)
    window_cell_sz = np.ceil(window_sz/cell_sz)
    window_cell_sz = window_cell_sz - (window_cell_sz%2) + 1
    cos_window = np.outer(np.hanning(window_cell_sz),np.hanning(window_cell_sz))
    cos_window = cos_window[None,None,:,:].repeat(vgg_num_channels,axis=1)
    cos_window = Variable(torch.from_numpy(cos_window).float(),requires_grad=False)
    if use_gpu:
        cos_window = cos_window.cuda()

    # get search patch and extract the feature
    patch = get_search_patch(image1_resized, target_center1, window_sz)
    patch = Variable(torch.from_numpy(patch[None,:]))
    if use_gpu:
        patch = patch.cuda()
    feat = feature_net.forward(patch)

    # resizing features as same size as cos_window
    feat = resize_feat_cos_window(feat, cos_window)
    feat_sz=feat.size(3)

    ## get correlation filter label(gaussian shape) ##
    target_cell_sz1 = np.ceil(target_sz1/cell_sz)
    rw = int(np.ceil(target_cell_sz1[0]/2)) # padding for CF
    rh = int(np.ceil(target_cell_sz1[1]/2)) # padding for CF
    output_sigma = target_cell_sz1*output_sigma_factor
    label1_np = gaussian_shaped_labels(output_sigma, feat_sz, target_sz1)
    label1 = Variable(torch.from_numpy(label1_np).float())
    label1 = label1.view(1,1,label1.size(0),label1.size(1))
    if use_gpu:
        label1 = label1.cuda()
    
    ## MetaCrest init ##
    state = torch.load(tracker_path)
    meta_init = state['meta_init']
    meta_alpha = state['meta_alpha']
    criterion = L2normLoss()
    if use_gpu:
        criterion = criterion.cuda()
    
    ## training first frame ##
    tracker_net = train_init(meta_init, meta_alpha, target_cell_sz1, criterion, feat, label1)
    online_optimizer = set_optimizer(tracker_net, online_lr_base, online_w_decay)
    tracker_net.eval()
    
    ## online prediction ##
    target_center = target_center1
    target_sz = target_sz1
    feat_update = []
    label_update = []
    iters = 0
    while True:
        imagefile = handle.frame()
        if not imagefile:
            break
        image = Image.open(imagefile).convert('RGB')
        image_resized = image.resize(image_sz_resized,Image.BILINEAR)
        patch = get_search_patch(image_resized, target_center, window_sz)
        #plt.imshow(patch[0].data.cpu().numpy().transpose(1,2,0)); plt.colorbar(); plt.show()
        patch = Variable(torch.from_numpy(patch[None,:]))
        if use_gpu:
            patch = patch.cuda()
        feat = feature_net.forward(patch)
        feat = resize_feat_cos_window(feat, cos_window)
        
        response = tracker_net.forward(feat)

        score = response.max().data[0]
        #plt.imshow(response.data.cpu().numpy()[0][0]); plt.colorbar(); plt.show()

        motion_sigma = target_cell_sz1*motion_sigma_factor
        motion_map = gaussian_shaped_labels(motion_sigma, window_cell_sz, target_sz)
        #plt.imshow(motion_map); plt.colorbar(); plt.show()
        motion_map = Variable(torch.from_numpy(motion_map).float())
        motion_map = motion_map.view(1,1,motion_map.size(0),motion_map.size(1))
        if use_gpu:
            motion_map = motion_map.cuda()
        
        response = response*motion_map
        response_np = response.data[0][0].cpu().numpy()
        #plt.imshow(response_np); plt.colorbar(); plt.show()
        delta = np.argwhere(response_np.max()==response_np)[0]+1 #zero-index -> original-index
        delta = delta-np.ceil(window_cell_sz/2)
        target_center[0] = target_center[0] + delta[1]*cell_sz
        target_center[1] = target_center[1] + delta[0]*cell_sz
        target_center = valid_pos(target_center, image_sz_resized)
        target_sz = scale_estimation(image_resized, feature_net, tracker_net, \
                                     target_sz, target_center, window_sz, cos_window)
        
        resized_target_bbox = np.zeros(4)
        resized_target_bbox[:2] = target_center - target_sz/2
        resized_target_bbox[2:] = target_sz
        result = np.round(resized_target_bbox/scale)
        vot_result = vot.Rectangle(result[0],result[1],result[2],result[3])
        handle.report(vot_result)
      
        if score > score_thres:
            temp_label = np.roll(label1_np, int(delta[0]), axis=0) # circular shift y-axis
            temp_label = np.roll(temp_label, int(delta[1]), axis=1) # x-axis
            if len(label_update) == max_db_size:
                label_update.pop(0)
                feat_update.pop(0)
            temp_label = torch.from_numpy(temp_label[None,:]).float()
            if use_gpu:
                temp_label = temp_label.cuda()
            label_update.append(temp_label)
            feat_update.append(feat[0].data.clone().float())
        
        if (iters+1)%update_interval==0:
            if len(label_update) >= update_interval:
                tracker_net.train()
                model_update(tracker_net, online_optimizer, criterion, torch.stack(feat_update), torch.stack(label_update))
        iters = iters+1

    return result, gt

use_gpu = True

## options
cf_num_channels = 64
vgg_num_channels = 512
output_sigma_factor = 0.1
motion_sigma_factor = 0.6
cell_sz = 4.0

online_lr_base = 0.000007
online_w_decay = 0
n_init_updates = 1
n_online_updates = 2
update_interval = 5
n_update_batch = 5
max_db_size = n_online_updates*n_update_batch

score_thres = 0.0

np.random.seed(int(time.time()))
torch.manual_seed(int(time.time()))
torch.cuda.manual_seed(int(time.time()))
    
feat_extractor_path = '/home/eunbyung/Works2/src/tracking/meta_trackers/meta_crest/models/imagenet-vgg-verydeep-16.mat'
tracker_path = '/home/eunbyung/Works2/src/tracking/meta_trackers/meta_crest/models/meta_init_otb_ilsvrc.pth'

result_bb, gt = run_crest()
