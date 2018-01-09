import os
import sys
import pickle
import time
import scipy.io
import argparse

import torch
import torch.optim as optim
from torch.autograd import Variable
import cv2
from matplotlib import pyplot as plt

sys.path.insert(0,'../')
from data_prov import *
from model import *
from utils import *

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

def train_init(meta_init, meta_alpha, target_cell_sz, loss_fn, feat, lh_feat, label_map):
    rw = int(np.ceil(target_cell_sz[0]/2))
    rh = int(np.ceil(target_cell_sz[1]/2))
    fw = int(rw*2 + 1)
    fh = int(rh*2 + 1)

    #print('cf_size(h,w): (18,12) -> (%d,%d)'%(fh,fw))
    # bilinear sampling for resizing correlation filters 
    cf_init_weight = OrderedDict()
    for k,p in meta_init.items():
        if 'cf.weight' in k:
            cf_init_weight[k] = bilinear_upsample(p, (fh, fw))
        else:
            cf_init_weight[k] = 1*p 
    cf_alpha = OrderedDict()
    for k,p in meta_alpha.items():
        if 'cf.weight' in k:
            cf_alpha[k] = bilinear_upsample(p, (fh, fw))
        else:
            cf_alpha[k] = 1*p

    # the first iteration
    temp = conv2d(feat, cf_init_weight['conv1.weight'], cf_init_weight['conv1.bias'],
                      stride=1,padding=0)
    response = conv2d(temp, cf_init_weight['cf.weight'], cf_init_weight['cf.bias'],
                      stride=1,padding=(rh,rw))
    loss = loss_fn(response,label_map)
    print('Init loss: %.4f'%(loss.data[0]))
    grads = torch.autograd.grad(loss, cf_init_weight.values(), create_graph=True)
    cf_weight = OrderedDict((name, param - torch.mul(alpha,grad)) for
                                  ((name, param),(_,alpha),grad) in
                                  zip(cf_init_weight.items(), cf_alpha.items(), grads))

    # subsequent iterations
    for i in range(n_init_updates-1):
        temp = conv2d(feat, cf_weight['conv1.weight'], cf_weight['conv1.bias'],
                        stride=1,padding=0)
        response = conv2d(temp, cf_weight['cf.weight'], cf_weight['cf.bias'],
                          stride=1,padding=(rh,rw))
        loss = loss_fn(response,label_map)
        grads = torch.autograd.grad(loss, cf_weight.values(), create_graph=True)
        cf_weight = OrderedDict((name, param - torch.mul(alpha,grad))
                                      for ((name, param),(_,alpha),grad) in
                                      zip(cf_weight.items(),cf_alpha.items(), grads))
        print('loss: %.4f'%(loss.data[0]))

    # lookahead patch loss
    temp = conv2d(lh_feat, cf_weight['conv1.weight'], cf_weight['conv1.bias'],
                        stride=1,padding=0)
    lh_response = conv2d(temp, cf_weight['cf.weight'], cf_weight['cf.bias'],
                          stride=1,padding=(rh,rw))
    lh_loss = loss_fn(lh_response,label_map)
    # current patch loss
    temp = conv2d(feat, cf_weight['conv1.weight'], cf_weight['conv1.bias'],
                        stride=1,padding=0)
    response = conv2d(temp, cf_weight['cf.weight'], cf_weight['cf.bias'],
                          stride=1,padding=(rh,rw))
    loss = loss_fn(response,label_map)
    
    print('loss: %.4f, lh_loss %.4f'%(loss.data[0], lh_loss.data[0]))
    #plt.imshow(lh_response[0][0].data.cpu().numpy()); plt.colorbar(); plt.show()

    # compute meta grads for lookahead dataset
    grads = torch.autograd.grad(lh_loss, meta_init.values(), retain_graph=True)
    alpha_grads = torch.autograd.grad(lh_loss, meta_alpha.values())
    meta_init_grads = {}
    meta_alpha_grads = {}
    count = 0
    for k,_ in meta_init.items():
        meta_init_grads[k] = grads[count]
        meta_alpha_grads[k] = alpha_grads[count]
        count = count + 1
    return meta_init_grads, meta_alpha_grads, loss.data[0], lh_loss.data[0]

# resizing features as same size as cos_window
def resize_feat_cos_window(feat, cos_window):
    num_channels = feat.size(1)
    cos_window_sz = cos_window.size(3)
    feat_data = feat.data.clone().cpu().numpy()
    feat = cv2.resize(feat_data[0].transpose(1,2,0).astype('float32'),dsize=(cos_window_sz,cos_window_sz))
    feat = feat.transpose(2,0,1)
    feat = Variable(torch.from_numpy(feat[None,:]).float())
    if use_gpu:
        feat = feat.cuda()
    feat = feat*cos_window
    return feat

def train_meta_crest():
    train_dataset = ILSVRCDataset(train_ilsvrc_data_path, ilsvrc_home+'/train')
    train_tracking_dataset = TrackingDataset(train_tracking_data_path)

    ## initialize feature extractor ##
    feature_net = FeatureExtractor(feat_extractor_path)
    if use_gpu:
        feature_net = feature_net.cuda()
    feature_net.eval()

    ## CFConv1Net init ##
    template_net = CFConv1Net([12,18], vgg_num_channels, cf_num_channels)
    criterion = L2normLoss()
    if use_gpu:
        template_net = template_net.cuda()
        criterion = criterion.cuda()
    
    ## Initialize Meta Init(initial weights for tracker) Net##
    meta_init = OrderedDict()
    for k,p in template_net.named_parameters():
        meta_init[k] = Variable(p.data.clone(), requires_grad=True)
    meta_init_params = [p for k,p in meta_init.items()]
    meta_init_optimizer = optim.Adam(meta_init_params, lr = meta_init_lr)

    ## Initialize Meta Alpha(learning rate) Net##
    meta_alpha = OrderedDict()
    for k,p in template_net.named_parameters():
        alpha = Variable(p.data.clone(), requires_grad=True)
        alpha.data.fill_(meta_alpha_init)
        meta_alpha[k]=alpha
    meta_alpha_params = [p for k,p in meta_alpha.items()]
    meta_alpha_optimizer = optim.Adam(meta_alpha_params, lr = meta_alpha_lr)
        
    best_loss = 10000.0
    loss_list = np.zeros(n_meta_updates)
    lh_loss_list = np.zeros(n_meta_updates)
    val_lh_loss_list = []

    for i in range(n_meta_updates):
        meta_init_grads = []
        meta_alpha_grads = []
        loss = np.zeros(n_meta_batch)
        lh_loss = np.zeros(n_meta_batch)
        if (i+1)%1000==0:
            states = {'iter': i,
                      'meta_init': meta_init,
                      'meta_alpha': meta_alpha,
                      'loss_list':loss_list, 'lh_loss_list':lh_loss_list,
                      'val_lh_loss_list': val_lh_loss_list}
            print("Save model to %s"%output_path)
            torch.save(states, output_path+'iter%d'%(i+1))
        
        ## Adjust learning rate ##
        if (i+1)%5000==0:
            meta_init_params = [p for k,p in meta_init.items()]
            meta_init_optimizer = optim.Adam(meta_init_params, lr = meta_init_lr*0.1)
            meta_alpha_params = [p for k,p in meta_alpha.items()]
            meta_alpha_optimizer = optim.Adam(meta_alpha_params, lr = meta_alpha_lr*0.1)

        tic = time.time()
        for j in range(n_meta_batch):
            if np.random.randint(10)>=3:
                patch, lh_patch, target_cell_sz, label_map, cos_window, seq_id = train_dataset.next()
            else:
                patch, lh_patch, target_cell_sz, label_map, cos_window, seq_id = train_tracking_dataset.next()
            
            patch = Variable(patch)
            lh_patch = Variable(lh_patch)
            label_map = Variable(label_map, requires_grad=False)
            cos_window = Variable(cos_window, requires_grad=False)
            if use_gpu:
                patch = patch.cuda()
                lh_patch = lh_patch.cuda()
                cos_window = cos_window.cuda()
                label_map = label_map.cuda()
            feat = feature_net.forward(patch)
            feat = resize_feat_cos_window(feat, cos_window)
            lh_feat = feature_net.forward(lh_patch)
            lh_feat = resize_feat_cos_window(lh_feat, cos_window)
            
#            plt.imshow(un_preprocess_image(patch.data.cpu().numpy()[0])); plt.show()
#            plt.imshow(un_preprocess_image(lh_patch.data.cpu().numpy()[0])); plt.show()
#            plt.imshow(label_map.data.cpu().numpy()[0][0]); plt.colorbar(); plt.show()
#            plt.imshow(cos_window.data.cpu().numpy()[0][0]); plt.colorbar(); plt.show()

            init_g, alpha_g, loss[j], lh_loss[j] = train_init(meta_init, meta_alpha, target_cell_sz,
                                    criterion, feat, lh_feat, label_map)
            meta_init_grads.append(init_g)
            meta_alpha_grads.append(alpha_g)
            print("\tseq_id:%d meta_iter:%d, meta_batch:%d, loss:%.4f, lh_loss:%.4f"
                  %(seq_id, i, j, loss[j], lh_loss[j]))
        
        toc = time.time()-tic
        loss_list[i] = loss.mean()
        lh_loss_list[i] = lh_loss.mean()
        print("[meta_update]:%d, loss:%.4f, lh_loss:%.4f, Time %.3f"
              %(i, loss_list[i], lh_loss_list[i], toc))
        meta_update(meta_init, meta_init_grads, meta_alpha, meta_alpha_grads,
                    meta_init_optimizer, meta_alpha_optimizer)

ilsvrc_home = '../../dataset/VID'
feat_extractor_path = '../models/imagenet-vgg-verydeep-16.mat'
train_ilsvrc_data_path = '../../dataset/ilsvrc_train.json'

use_gpu = True

n_init_updates = 1
meta_init_lr = 1e-6
meta_alpha_init = 1e-6
meta_alpha_lr = 1e-6
n_meta_updates = 10000
n_meta_batch = 8

cf_num_channels = 64
vgg_num_channels = 512
output_sigma_factor = 0.1
motion_sigma_factor = 0.6
cell_sz = 4.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', default='OTB', help='OTB or VOT')
    args = parser.parse_args()
    if args.experiment=='OTB':
        # for OTB experiment
        train_tracking_data_path = '../../dataset/vot-otb.pkl'
        output_path = '../models/meta_init_vot_ilsvrc.pth'
        print('meta-training for OTB experiment')
    elif args.experiment=='VOT':
        # for VOT experiment
        train_tracking_data_path = '../../dataset/otb-vot.pkl' #for VOT experiment
        output_path = '../models/meta_init_otb_ilsvrc.pth'
        print('meta-training for VOT experiment')
    
    train_meta_crest()
