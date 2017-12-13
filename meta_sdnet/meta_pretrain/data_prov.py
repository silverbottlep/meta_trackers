import os
import sys
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import json
import pickle

import torch
import torch.utils.data as data

sys.path.insert(0,'../modules')
from sample_generator import *

class ILSVRCDataset(data.Dataset):
    def __init__(self, data_path, data_home, opts):
        with open(data_path, 'r') as f:
          self.dataset = json.load(f)
        self.img_dir = data_home
        self.n_seq = len(self.dataset)
        self.seq_names = []
        self.gt = []
        self.pos_generators=[]
        self.neg_generators=[]
        self.ranges = []
        for i in range(self.n_seq):
            self.seq_names.append(self.dataset[i]['seq_name'])
        for i in range(self.n_seq):
            self.gt.append(np.array(self.dataset[i]['gt']))
        for i in range(self.n_seq):
            self.ranges.append((self.dataset[i]['start_frame'],self.dataset[i]['end_frame']))
        for i in range(self.n_seq):
            image_size = (self.dataset[i]['im_width'], self.dataset[i]['im_height'])
            self.pos_generators.append(SampleGenerator('gaussian', image_size, 0.1, 1.2, 1.1, True))
            self.neg_generators.append(SampleGenerator('uniform', image_size, 1, 1.2, 1.1, True))

        self.batch_pos = opts['batch_pos']
        self.batch_neg = opts['batch_neg']
        self.overlap_pos = opts['overlap_pos']
        self.overlap_neg = opts['overlap_neg']
        self.crop_size = opts['img_size']
        self.padding = opts['padding']
        self.shuffle = opts['label_shuffling']
        
        self.lookahead = 10

        self.display = False
        
    def __iter__(self):
        return self

    def __next__(self):
        seq_id = np.random.randint(self.n_seq)
        start_frame = self.ranges[seq_id][0]
        end_frame = self.ranges[seq_id][1]
        idx = np.random.randint(end_frame-start_frame-self.lookahead)
        lh_idx = idx + np.random.randint(self.lookahead)+1
        bbox = self.gt[seq_id][idx]
        lh_bbox = self.gt[seq_id][lh_idx]
        img_path = self.img_dir + ('/%s/%06d.JPEG'%(self.seq_names[seq_id], (start_frame-1)+idx))
        lh_img_path = self.img_dir + ('/%s/%06d.JPEG'%(self.seq_names[seq_id],(start_frame-1)+lh_idx))
            
        image = Image.open(img_path).convert('RGB')
        image_np = np.asarray(image)
        lh_image = Image.open(lh_img_path).convert('RGB')
        lh_image_np = np.asarray(lh_image)

        pos_examples = gen_samples(self.pos_generators[seq_id], bbox, self.batch_pos, overlap_range=self.overlap_pos)
        neg_examples = gen_samples(self.neg_generators[seq_id], bbox, self.batch_neg, overlap_range=self.overlap_neg)
        pos_regions = self.extract_regions(image_np, pos_examples)
        neg_regions = self.extract_regions(image_np, neg_examples)

        lh_pos_examples = gen_samples(self.pos_generators[seq_id], lh_bbox, self.batch_pos, overlap_range=self.overlap_pos)
        lh_neg_examples = gen_samples(self.neg_generators[seq_id], lh_bbox, self.batch_neg, overlap_range=self.overlap_neg)
        lh_pos_regions = self.extract_regions(lh_image_np, lh_pos_examples)
        lh_neg_regions = self.extract_regions(lh_image_np, lh_neg_examples)
        
        if self.display:
            self.display_samples(image, bbox, pos_examples, neg_examples)
            self.display_samples(lh_image, lh_bbox, lh_pos_examples, lh_neg_examples)
        
        pos_regions = torch.from_numpy(pos_regions).float()
        neg_regions = torch.from_numpy(neg_regions).float()
        lh_pos_regions = torch.from_numpy(lh_pos_regions).float()
        lh_neg_regions = torch.from_numpy(lh_neg_regions).float()
        # label shuffling
        if self.shuffle:
            if np.random.randint(2) > 0:
                return pos_regions, neg_regions, lh_pos_regions, lh_neg_regions, seq_id
            else:
                return neg_regions, pos_regions, lh_neg_regions, lh_pos_regions, seq_id
        else:
            return pos_regions, neg_regions, lh_pos_regions, lh_neg_regions, seq_id
    
    next = __next__

    def extract_regions(self, image, samples):
        regions = np.zeros((len(samples),self.crop_size,self.crop_size,3),dtype='uint8')
        for i, sample in enumerate(samples):
            regions[i] = crop_image(image, sample, self.crop_size, self.padding, True)
        
        regions = regions.transpose(0,3,1,2)
        regions = regions.astype('float32') - 128.
        return regions

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

class TrackingDataset(data.Dataset):
    def __init__(self, data_path, opts):
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
        self.pos_init_generators=[]
        self.neg_init_generators=[]
        self.pos_generators=[]
        self.neg_generators=[]
        self.img_size = []
        for i in range(self.n_seq):
            im_path = self.img_dir + '/%s/%s'%(self.seq_names[i],self.img_list[i][0])
            image = Image.open(im_path).convert('RGB')
            self.img_size.append(np.array(image.size))
            self.pos_init_generators.append(SampleGenerator('gaussian', image.size, 0.1, 1.2, 1.1, True))
            self.neg_init_generators.append(SampleGenerator('uniform', image.size, 1, 1.2, 1.1, True))
            self.pos_generators.append(SampleGenerator('gaussian', image.size, 0.1, 1.2))
            self.neg_generators.append(SampleGenerator('uniform', image.size, 1.5, 1.2))
        
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

        self.batch_pos = opts['batch_pos']
        self.batch_neg = opts['batch_neg']
        self.overlap_pos = opts['overlap_pos']
        self.overlap_neg = opts['overlap_neg']
        self.crop_size = opts['img_size']
        self.padding = opts['padding']
        self.shuffle = opts['label_shuffling']
        
        self.lookahead = 10

        self.display = False
        
    def __iter__(self):
        return self

    def __next__(self):
        okay = False
        while okay==False:
            seq_id = np.random.randint(self.n_seq)
            seq_len = len(self.img_list[seq_id])
            idx = np.random.randint(seq_len-self.lookahead)
            lh_idx = idx + np.random.randint(self.lookahead)+1
            bbox = self.gt[seq_id][idx]
            lh_bbox = self.gt[seq_id][lh_idx]

            img_path = self.img_dir + ('/%s/%s'%(self.seq_names[seq_id],self.img_list[seq_id][idx]))
            lh_img_path = self.img_dir + ('/%s/%s'%(self.seq_names[seq_id],self.img_list[seq_id][lh_idx]))
                
            image = Image.open(img_path).convert('RGB')
            image_np = np.asarray(image)
            lh_image = Image.open(lh_img_path).convert('RGB')
            lh_image_np = np.asarray(lh_image)

            pos_examples = gen_samples(self.pos_init_generators[seq_id], bbox, self.batch_pos, overlap_range=self.overlap_pos)
            neg_examples = gen_samples(self.neg_init_generators[seq_id], bbox, self.batch_neg, overlap_range=self.overlap_neg)
            pos_regions = self.extract_regions(image_np, pos_examples)
            neg_regions = self.extract_regions(image_np, neg_examples)

            lh_pos_examples = gen_samples(self.pos_init_generators[seq_id], lh_bbox, self.batch_pos, overlap_range=self.overlap_pos)
            lh_neg_examples = gen_samples(self.neg_init_generators[seq_id], lh_bbox, self.batch_neg, overlap_range=self.overlap_neg)
            lh_pos_regions = self.extract_regions(lh_image_np, lh_pos_examples)
            lh_neg_regions = self.extract_regions(lh_image_np, lh_neg_examples)
            
            if self.display:
                self.display_samples(image, bbox, pos_examples, neg_examples)
                self.display_samples(lh_image, lh_bbox, lh_pos_examples, lh_neg_examples)
            
            if pos_regions.shape[0] > 10 and neg_regions.shape[0] > 10 and lh_pos_regions.shape[0] > 10 and lh_neg_regions.shape[0] > 10:
                okay = True
        
        pos_regions = torch.from_numpy(pos_regions).float()
        neg_regions = torch.from_numpy(neg_regions).float()
        lh_pos_regions = torch.from_numpy(lh_pos_regions).float()
        lh_neg_regions = torch.from_numpy(lh_neg_regions).float()
        # label shuffling
        if self.shuffle:
            if np.random.randint(2) > 0:
                return pos_regions, neg_regions, lh_pos_regions, lh_neg_regions, seq_id
            else:
                return neg_regions, pos_regions, lh_neg_regions, lh_pos_regions, seq_id
        else:
            return pos_regions, neg_regions, lh_pos_regions, lh_neg_regions, seq_id
    
    next = __next__
    
    def extract_regions(self, image, samples):
        regions = np.zeros((len(samples),self.crop_size,self.crop_size,3),dtype='uint8')
        for i, sample in enumerate(samples):
            regions[i] = crop_image(image, sample, self.crop_size, self.padding, True)
        
        regions = regions.transpose(0,3,1,2)
        regions = regions.astype('float32') - 128.
        return regions

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
