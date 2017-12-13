import os
import sys
import pickle
import time
import scipy.io

import torch
import torch.optim as optim
from torch.autograd import Variable

from data_prov import *
from meta_model import *
from options import *

def meta_update(meta_init, meta_init_grads, meta_alpha, meta_alpha_grads, 
                meta_init_optimizer, meta_alpha_optimizer):
    # Unpack the list of grad dicts
    init_gradients = {k: sum(d[k] for d in meta_init_grads) for k in meta_init_grads[0].keys()}
    alpha_gradients = {k: sum(d[k] for d in meta_alpha_grads) for k in meta_alpha_grads[0].keys()}
    
    # dummy variable to mimic forward and backward
    dummy_x = Variable(torch.Tensor(np.random.randn(1)), requires_grad=False).cuda()
    
    # update meta_init(for initial weights)
    for k,init in meta_init.items():
        dummy_x = torch.sum(dummy_x*init)
    meta_init_optimizer.zero_grad()                                                               
    dummy_x.backward()
    for k,init in meta_init.items():
        init.grad = init_gradients[k]
    meta_init_optimizer.step()

    # update meta_alpha(for learning rate)
    dummy_y = Variable(torch.Tensor(np.random.randn(1)), requires_grad=False).cuda()
    for k,alpha in meta_alpha.items():
        dummy_y = torch.sum(dummy_y*alpha)
    meta_alpha_optimizer.zero_grad()
    dummy_y.backward()
    for k,alpha in meta_alpha.items():
        alpha.grad = alpha_gradients[k]
    meta_alpha_optimizer.step()

def train_init(tracker_net, meta_alpha, loss_fn, pos_regions, neg_regions, lh_pos_regions,
               lh_neg_regions, evaluator, train_all=False):
    if train_all:
        tracker_init_weights = OrderedDict((name, param) for (name, param) in tracker_net.named_parameters())
        tracker_keys = [name for (name, _) in tracker_net.named_parameters()]
    else:
        tracker_init_weights = OrderedDict((name, param) for (name, param) in tracker_net.named_parameters() if name.startswith('fc') )
        tracker_keys = [name for (name, _) in tracker_net.named_parameters() if name.startswith('fc')]

    # the first iteration
    pos_score = tracker_net.forward(pos_regions)
    neg_score = tracker_net.forward(neg_regions)
    loss = loss_fn(pos_score,neg_score)
    grads = torch.autograd.grad(loss, tracker_init_weights.values(), create_graph=True)
    tracker_weights = OrderedDict((name, param - torch.mul(meta_alpha,grad)) for
                                  ((name, param),(_,meta_alpha),grad) in
                                  zip(tracker_init_weights.items(),
                                      meta_alpha.items(), grads))

    for i in range(opts['n_init_updates']-1):
        pos_score = tracker_net.forward(pos_regions, tracker_weights)
        neg_score = tracker_net.forward(neg_regions, tracker_weights)
        loss = loss_fn(pos_score,neg_score)
        grads = torch.autograd.grad(loss, tracker_weights.values(), create_graph=True)
        tracker_weights = OrderedDict((name, param - torch.mul(meta_alpha,grad))
                                      for ((name, param),(_,meta_alpha),grad) in
                                      zip(tracker_weights.items(),meta_alpha.items(), grads))

    lh_pos_score = tracker_net.forward(lh_pos_regions, tracker_weights)
    lh_neg_score = tracker_net.forward(lh_neg_regions, tracker_weights)
    lh_loss = loss_fn(lh_pos_score,lh_neg_score)
    lh_acc,lh_acc_pos,lh_acc_neg = evaluator(lh_pos_score, lh_neg_score)
    
    pos_score = tracker_net.forward(pos_regions, tracker_weights)
    neg_score = tracker_net.forward(neg_regions, tracker_weights)
    loss = loss_fn(pos_score,neg_score)
    acc,acc_pos,acc_neg = evaluator(pos_score, neg_score)

    # compute meta grads for lookahead dataset
    grads = torch.autograd.grad(lh_loss, tracker_init_weights.values(), retain_graph=True)
    alpha_grads = torch.autograd.grad(lh_loss, meta_alpha.values())
    meta_init_grads = {}
    meta_alpha_grads = {}
    for i in range(len(tracker_keys)):
        meta_init_grads[tracker_keys[i]] = grads[i]
        meta_alpha_grads[tracker_keys[i]] = alpha_grads[i]
    return meta_init_grads, meta_alpha_grads, loss.data[0], lh_loss.data[0], acc, lh_acc

img_home = '../../dataset/'
ilsvrc_home = '/home/eunbyung/Works2/data/ILSVRC/Data/VID'
train_ilsvrc_data_path = 'data/ilsvrc_train.json'
val_data_path = 'data/ilsvrc_val.json'
val_metadata_path = 'data/ilsvrc_val_meta.json'

# for OTB experiment
#train_tracking_data_path = 'data/vot-otb.pkl'
#opts['output_path'] = '../models/meta_init_vot_ilsvrc_new.pth'

# for VOT experiment
train_tracking_data_path = 'data/otb-vot.pkl' #for VOT experiment
opts['output_path'] = '../models/meta_init_otb_ilsvrc_new.pth'

opts['init_model_path'] = '../models/imagenet-vgg-m.mat'
opts['n_init_updates'] = 1
opts['label_shuffling'] = True
opts['n_meta_updates'] = 10000

def train_meta_sdnet():
    # save file name
    train_dataset = ILSVRCDataset(train_ilsvrc_data_path, ilsvrc_home + '/train', opts)
    train_tracking_dataset = TrackingDataset(train_tracking_data_path, opts)

    ## Initialize Tracker Net ##
    tracker_net = MetaSDNet(opts['init_model_path'])
    if opts['use_gpu']:
        tracker_net = tracker_net.cuda()
    tracker_net.set_learnable_params(['fc'])

    ## Initialize Meta Alpha(learning rate) Net##
    meta_init= OrderedDict()
    for k,p in tracker_net.get_learnable_params().items():
        meta_init[k] = Variable(p.data.clone(), requires_grad=True)
    meta_init_params = [p for k,p in meta_init.items()]
    meta_init_optimizer = optim.Adam(meta_init_params, lr = opts['meta_init_lr'])

    ## Initialize Meta Alpha(learning rate) Net##
    meta_alpha = OrderedDict()
    for k,p in tracker_net.get_learnable_params().items():
        alpha = Variable(p.data.clone(), requires_grad=True)
        alpha.data.fill_(opts['tracker_init_lr'])
        meta_alpha[k]=alpha
    meta_alpha_params = [p for k,p in meta_alpha.items()]
    meta_alpha_optimizer = optim.Adam(meta_alpha_params, lr = opts['meta_alpha_lr'])
        
    criterion = BinaryLoss()
    evaluator = Accuracy()

    loss_list = np.zeros(opts['n_meta_updates'])
    lh_loss_list = np.zeros(opts['n_meta_updates'])
    acc_list = np.zeros(opts['n_meta_updates'])
    lh_acc_list = np.zeros(opts['n_meta_updates'])

    # first 5000 meta updates, only updates fc layers
    train_all = False
    for i in range(opts['n_meta_updates']):
        meta_init_grads = []
        meta_alpha_grads = []
        loss = np.zeros(opts['meta_batch_size'])
        lh_loss = np.zeros(opts['meta_batch_size'])
        acc = np.zeros(opts['meta_batch_size'])
        lh_acc = np.zeros(opts['meta_batch_size'])
        if (i+1)%1000==0:
            states = {'iter': i+1,
                      'meta_init': meta_init,
                      'meta_alpha': meta_alpha,
                      'loss_list':loss_list, 'lh_loss_list':lh_loss_list,
                      'acc_list':acc_list, 'lh_acc_list':lh_acc_list}
            print("Save model to %s"%opts['output_path'])
            torch.save(states, opts['output_path'])

        # train all layers after 5000 meta updates
        if (i+1)%10==0:
            train_all = True
            tracker_net = MetaSDNetFull(opts['init_model_path'])
            if opts['use_gpu']:
                tracker_net = tracker_net.cuda()
            tracker_net.set_learnable_params(['fc','features'])
            fc_trained_meta_init = meta_init.copy()
            fc_trained_meta_alpha = meta_alpha.copy()
            tracker_net.copy_meta_weights(fc_trained_meta_init)
            
            # we meta-train all layers
            meta_init = OrderedDict()
            for k,p in tracker_net.get_learnable_params().items():
                meta_init[k] = Variable(p.data.clone(), requires_grad=True)
            meta_init_params = [p for k,p in meta_init.items()]
            meta_init_optimizer = optim.Adam(meta_init_params, lr = opts['meta_init_lr'])
    
            ## Initialize Meta Alpha(learning rate) Net##
            meta_alpha = OrderedDict()
            for k,p in tracker_net.get_learnable_params().items():
                if k in fc_trained_meta_alpha:
                    meta_alpha[k]=fc_trained_meta_alpha[k]
                else:
                    alpha = Variable(p.data.clone(), requires_grad=True)
                    alpha.data.fill_(opts['tracker_init_lr'])
                    meta_alpha[k]=alpha
            meta_alpha_params = [p for k,p in meta_alpha.items()]
            meta_alpha_optimizer = optim.Adam(meta_alpha_params, lr = opts['meta_alpha_lr'])

        tic = time.time()
        for j in range(opts['meta_batch_size']):
            if np.random.randint(10)>=3:
                pos_regions, neg_regions, lh_pos_regions, lh_neg_regions, seq_id = train_dataset.next()
            else:
                pos_regions, neg_regions, lh_pos_regions, lh_neg_regions, seq_id = train_tracking_dataset.next()
            
            pos_regions = Variable(pos_regions)
            neg_regions = Variable(neg_regions)
            lh_pos_regions = Variable(lh_pos_regions)
            lh_neg_regions = Variable(lh_neg_regions)
        
            if opts['use_gpu']:
                pos_regions = pos_regions.cuda()
                neg_regions = neg_regions.cuda()
                lh_pos_regions = lh_pos_regions.cuda()
                lh_neg_regions = lh_neg_regions.cuda()
            
            tracker_net.copy_meta_weights(meta_init)
            init_g, alpha_g, loss[j], lh_loss[j], acc[j], lh_acc[j] = train_init(tracker_net, 
                    meta_alpha, criterion, pos_regions, neg_regions, lh_pos_regions, lh_neg_regions, evaluator, train_all)
            meta_init_grads.append(init_g)
            meta_alpha_grads.append(alpha_g)
            print("\tseq_id:%d meta_iter:%d, meta_batch:%d, loss:%.4f, lh_loss:%.4f, acc:%.4f, ls_acc:%.4f"
                  %(seq_id, i, j, loss[j], lh_loss[j], acc[j], lh_acc[j]))
        
        toc = time.time()-tic
        loss_list[i] = loss.mean()
        lh_loss_list[i] = lh_loss.mean()
        acc_list[i] = acc.mean()
        lh_acc_list[i] = lh_acc.mean()
        print("[meta_update]:%d, loss:%.4f, lh_loss:%.4f, acc:%.4f, lh_acc:%.4f Time %.3f"
              %(i,loss_list[i], lh_loss_list[i], acc_list[i],lh_acc_list[i],toc))
        meta_update(meta_init, meta_init_grads, meta_alpha, meta_alpha_grads,
                    meta_init_optimizer, meta_alpha_optimizer)

if __name__ == "__main__":
    train_meta_sdnet()
