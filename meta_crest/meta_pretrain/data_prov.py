import os
import sys
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import json
import pickle

import torch
import torch.utils.data as data

from utils import *

# extension to get the regions of both current and future(lookahead) frames
class ILSVRCDataset(data.Dataset):
    def __init__(self, data_path, data_home):
        with open(data_path, 'r') as f:
          self.dataset = json.load(f)
        self.img_dir = data_home
        self.n_seq = len(self.dataset)
        self.seq_names = []
        self.gt = []
        self.ranges = []
        for i in range(self.n_seq):
            self.seq_names.append(self.dataset[i]['seq_name'])
        for i in range(self.n_seq):
            self.gt.append(np.array(self.dataset[i]['gt']))
        for i in range(self.n_seq):
            self.ranges.append((self.dataset[i]['start_frame'],self.dataset[i]['end_frame']))

        self.lookahead = 10
        self.max_object_sz = 100
        self.cell_sz = 4
        self.output_sigma_factor= 0.1
        self.feat_num_channels = 512
        
    def __iter__(self):
        return self

    def __next__(self):
        seq_id = np.random.randint(self.n_seq)
        start_frame = self.ranges[seq_id][0]
        end_frame = self.ranges[seq_id][1]
        idx = np.random.randint(end_frame-start_frame-self.lookahead)
        lh_idx = idx + np.random.randint(self.lookahead)+1
        bbox = np.copy(self.gt[seq_id][idx])
        lh_bbox = np.copy(self.gt[seq_id][lh_idx])
        img_path = self.img_dir + ('/%s/%06d.JPEG'%(self.seq_names[seq_id], (start_frame-1)+idx))
        lh_img_path = self.img_dir + ('/%s/%06d.JPEG'%(self.seq_names[seq_id],(start_frame-1)+lh_idx))
           
        # resize the image mainly for reducing computational cost
        scale = 1
        if bbox[2:].prod() > self.max_object_sz**2:
            scale = self.max_object_sz/(bbox[2:].max())
        image = Image.open(img_path).convert('RGB')
        lh_image = Image.open(lh_img_path).convert('RGB')
        if scale!=1:
            image = image.resize((int(image.size[0]*scale),int(image.size[1]*scale)), Image.BICUBIC)
            lh_image = lh_image.resize((int(lh_image.size[0]*scale),int(lh_image.size[1]*scale)), Image.BICUBIC)
            bbox = np.round(bbox*scale)
            lh_bbox = np.round(lh_bbox*scale)
       
        # calculate sizes
        target_sz = bbox[2:]
        target_center = bbox[:2] + np.floor(target_sz/2)
        target_cell_sz = np.ceil(target_sz/self.cell_sz)
        window_sz = get_search_size(target_sz)
        window_cell_sz = np.ceil(window_sz/self.cell_sz)
        window_cell_sz = window_cell_sz - (window_cell_sz%2) + 1
        
        # get center cropped patch for current image
        center = bbox[:2] + np.floor(target_sz/2)
        patch = get_search_patch(image, center, window_sz)
        
        # get center cropped patch for lookahead image
        # patch size is same
        lh_target_sz = lh_bbox[2:]
        lh_center = lh_bbox[:2] + np.floor(lh_target_sz/2)
        lh_patch = get_search_patch(lh_image, lh_center, window_sz)

        # get some windows
        cos_window = np.outer(np.hanning(window_cell_sz),np.hanning(window_cell_sz))
        cos_window = cos_window[None,None,:,:].repeat(self.feat_num_channels,axis=1)
        output_sigma = target_cell_sz*self.output_sigma_factor
        label_map = gaussian_shaped_labels(output_sigma, window_cell_sz, target_sz)

        # tensorize them
        patch = torch.from_numpy(patch[None,:]).float()
        lh_patch = torch.from_numpy(lh_patch[None,:]).float()
        label_map = torch.from_numpy(label_map[None,None,:]).float()
        cos_window = torch.from_numpy(cos_window).float()

        return patch, lh_patch, target_cell_sz, label_map, cos_window, seq_id

    def display_samples(self, image, bbox=None):
        dpi = 80.0
        figsize = (image.size[0]/dpi, image.size[1]/dpi)
        fig = plt.figure(frameon=False, figsize=figsize, dpi=dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        im = ax.imshow(image)
        if bbox is not None:
            gt_rect = plt.Rectangle(tuple(bbox[:2]),bbox[2],bbox[3], 
                    linewidth=3, edgecolor="#110000", zorder=1, fill=False)
            ax.add_patch(gt_rect)
        plt.pause(.01)
        plt.draw()

    next = __next__

class TrackingDataset(data.Dataset):
    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
            self.dataset = pickle.load(f)
        self.n_seq = len(self.dataset)
        self.seq_names = []
        self.img_list = []
        self.gt = []
        for seqname, seq in self.dataset.items():
            self.seq_names.append(seqname)
            self.img_list.append(seq['images'])
            self.gt.append(seq['gt'])
        
        self.img_dir = '../../dataset'
        self.img_size = []
        for i in range(self.n_seq):
            im_path = self.img_dir + '/%s/%s'%(self.seq_names[i],self.img_list[i][0])
            image = Image.open(im_path).convert('RGB')
            self.img_size.append(np.array(image.size))
        
        # adjust gt bboxes
        for i in range(self.n_seq):
            gt = self.gt[i]
            img_size = self.img_size[i] 
            for j in range(len(gt)):
                bbox = gt[j]
                if bbox[0]<0:
                    bbox[0]=0
                if bbox[1]<0:
                    bbox[1]=0
                if bbox[0] + bbox[2] > img_size[0]:
                    bbox[2] = img_size[0]-bbox[0]
                if bbox[1] + bbox[3] > img_size[1]:
                    bbox[3] = img_size[1]-bbox[1]

        self.lookahead = 20
        self.max_object_sz = 100
        self.cell_sz = 4
        self.output_sigma_factor= 0.1
        self.feat_num_channels = 512
        
    def __iter__(self):
        return self

    def __next__(self):
        seq_id = np.random.randint(self.n_seq)
        seq_len = len(self.img_list[seq_id])
        idx = np.random.randint(seq_len-self.lookahead)
        lh_idx = idx + np.random.randint(self.lookahead)+1
        bbox = self.gt[seq_id][idx]
        lh_bbox = self.gt[seq_id][lh_idx]
        img_path = self.img_dir + ('/%s/%s'%(self.seq_names[seq_id],self.img_list[seq_id][idx]))
        lh_img_path = self.img_dir + ('/%s/%s'%(self.seq_names[seq_id],self.img_list[seq_id][lh_idx]))
            
        # resize the image mainly for reducing computational cost
        scale = 1
        if bbox[2:].prod() > self.max_object_sz**2:
            scale = self.max_object_sz/(bbox[2:].max())
        image = Image.open(img_path).convert('RGB')
        lh_image = Image.open(lh_img_path).convert('RGB')
        if scale!=1:
            image = image.resize((int(image.size[0]*scale),int(image.size[1]*scale)), Image.BICUBIC)
            lh_image = lh_image.resize((int(lh_image.size[0]*scale),int(lh_image.size[1]*scale)), Image.BICUBIC)
            bbox = np.round(bbox*scale)
            lh_bbox = np.round(lh_bbox*scale)
        
        # calculate sizes
        target_sz = bbox[2:]
        target_center = bbox[:2] + np.floor(target_sz/2)
        target_cell_sz = np.ceil(target_sz/self.cell_sz)
        window_sz = get_search_size(target_sz)
        window_cell_sz = np.ceil(window_sz/self.cell_sz)
        window_cell_sz = window_cell_sz - (window_cell_sz%2) + 1
        
        # get center cropped patch for current image
        center = bbox[:2] + np.floor(target_sz/2)
        patch = get_search_patch(image, center, window_sz)

        # get center cropped patch for lookahead image
        # patch size is same
        lh_target_sz = lh_bbox[2:]
        lh_center = lh_bbox[:2] + np.floor(lh_target_sz/2)
        lh_patch = get_search_patch(lh_image, lh_center, window_sz)
        
        # get some windows
        cos_window = np.outer(np.hanning(window_cell_sz),np.hanning(window_cell_sz))
        cos_window = cos_window[None,None,:,:].repeat(self.feat_num_channels,axis=1)
        output_sigma = target_cell_sz*self.output_sigma_factor
        label_map = gaussian_shaped_labels(output_sigma, window_cell_sz, target_sz)
        
        # tensorize them
        patch = torch.from_numpy(patch[None,:]).float()
        lh_patch = torch.from_numpy(lh_patch[None,:]).float()
        label_map = torch.from_numpy(label_map[None,None,:]).float()
        cos_window = torch.from_numpy(cos_window).float()
    
        return patch, lh_patch, target_cell_sz, label_map, cos_window, seq_id

    next = __next__
    
    def display_samples(self, image, bbox, pos_samples, neg_samples):
        dpi = 80.0
        figsize = (image.size[0]/dpi, image.size[1]/dpi)
        fig = plt.figure(frameon=False, figsize=figsize, dpi=dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        im = ax.imshow(image)

        gt_rect = plt.Rectangle(tuple(bbox[:2]),bbox[2],bbox[3], 
                linewidth=3, edgecolor="#000000", zorder=1, fill=False)
        ax.add_patch(gt_rect)
        for i in range(min(5,len(pos_samples))):
          pos_rect = plt.Rectangle(tuple(pos_samples[i,:2]),pos_samples[i,2],pos_samples[i,3], 
                linewidth=1, edgecolor="#00ff00", zorder=1, fill=False)
          ax.add_patch(pos_rect)
        for i in range(min(10,len(neg_samples))):
          pos_rect = plt.Rectangle(tuple(neg_samples[i,:2]),neg_samples[i,2],neg_samples[i,3], 
                linewidth=1, edgecolor="#ff0000", zorder=1, fill=False)
          ax.add_patch(pos_rect)

        plt.pause(.01)
        plt.draw()
