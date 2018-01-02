import os
import scipy.io
import numpy as np
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch

def valid_pos(pos, img_sz):
    w = int(img_sz[0])
    h = int(img_sz[1])
    if pos[0]<0:
        pos[0] = 0
    if pos[0]>w-1:
        pos[0] = w-1
    if pos[1]<0:
        pos[1] = 0
    if pos[1]>h-1:
        pos[1] = h-1
    return pos

def overlap_ratio(rect1, rect2):
    '''
    Compute overlap ratio between two rects
    - rect: 1d array of [x,y,w,h] or 
            2d array of N x [x,y,w,h]
    '''

    if rect1.ndim==1:
        rect1 = rect1[None,:]
    if rect2.ndim==1:
        rect2 = rect2[None,:]

    left = np.maximum(rect1[:,0], rect2[:,0])
    right = np.minimum(rect1[:,0]+rect1[:,2], rect2[:,0]+rect2[:,2])
    top = np.maximum(rect1[:,1], rect2[:,1])
    bottom = np.minimum(rect1[:,1]+rect1[:,3], rect2[:,1]+rect2[:,3])

    intersect = np.maximum(0,right - left) * np.maximum(0,bottom - top)
    union = rect1[:,2]*rect1[:,3] + rect2[:,2]*rect2[:,3] - intersect
    iou = np.clip(intersect / union, 0, 1)
    return iou

def gaussian_shaped_labels(sigma, sz, target_size):
    v = np.linspace(1,sz,sz) - np.ceil(sz/2)
    xs,ys = np.meshgrid(v,v)

    size_max = max(target_size)
    size_min = min(target_size)
    if (size_max/size_min<1.025) and size_max>120:
        alpha=0.2
    else:
        alpha=0.3
    
    labels = np.exp(-alpha*((xs**2)/(sigma[0]**2) + (ys**2)/(sigma[1]**2)))
    return labels

# search window size is 5 times larger than current target size
def get_search_size(target_sz):
    ratio = target_sz[1]/target_sz[0]
    # if height is larger then width
    if ratio>1:
        window_sz = np.round(target_sz*np.array([5*ratio,5]))
    # otherwise
    else:
        window_sz = np.round(target_sz*np.array([5,5/ratio]))
    window_sz = window_sz - (window_sz%2) + 1
    return window_sz.max()

def get_search_patch(img, center, sz):
    img_array = np.asarray(img, dtype='float32')
    patch = crop_image(img_array, center, [sz,sz])
    patch = preprocess_image(patch)
    return patch

def crop_image(img, center, sz):
    center_x,center_y = np.array(center,dtype='float32')-1
    #center_x,center_y = np.array(center,dtype='float32')
    w,h = np.array(sz,dtype='float32')
    half_w, half_h = w/2, h/2

    img_h, img_w, _ = img.shape
    min_x = int(center_x - half_w + 0.5)
    min_y = int(center_y - half_h + 0.5)
    max_x = int(center_x + half_w + 0.5)
    max_y = int(center_y + half_h + 0.5)

    if min_x >=0 and min_y >= 0 and max_x <= img_w and max_y <= img_h:
        cropped = img[min_y:max_y, min_x:max_x, :]
    else:
        min_x_val = max(0, min_x)
        min_y_val = max(0, min_y)
        max_x_val = min(img_w, max_x)
        max_y_val = min(img_h, max_y)

        cropped = 128 * np.ones((max_y-min_y, max_x-min_x, 3), dtype='float32')
        cropped[min_y_val-min_y:max_y_val-min_y, min_x_val-min_x:max_x_val-min_x, :] \
           = img[min_y_val:max_y_val, min_x_val:max_x_val, :]
 
    return cropped

def preprocess_image(img):
    # RGB(from matconvnet)
    mean = np.array([123.68, 116.779, 103.939])
    for i in range(3):
        img[:,:,i] = img[:,:,i] - mean[i]
    img = img.transpose(2,0,1)
    return img

def un_preprocess_image(img):
    # RGB(from matconvnet)
    img = img.transpose(1,2,0)
    mean = np.array([123.68, 116.779, 103.939])
    for i in range(3):
        img[:,:,i] = img[:,:,i] + mean[i]
    img = img.astype(np.uint8)
    return img

def clip_grad_norm(grads, max_norm, norm_type=2):
    """Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Variable]): an iterable of Variables that will have
            gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for grad in grads)
    else:
        total_norm = 0
        for grad in grads:
            grad_norm = grad.data.norm(norm_type)
            total_norm += grad_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for grad in grads:
            grad.data.mul_(clip_coef)
    return total_norm
