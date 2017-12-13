import os
import sys
import numpy as np
import scipy.io
import json

file_name = 'ilsvrc_train.mat'
output_name = 'ilsvrc_train.json'
mat = scipy.io.loadmat(file_name)
n_seq = len(mat['dataset'][0])

threshold = 0.6

dataset=[]
for i in range(n_seq):
    im_width  = int(mat['dataset'][0][i][0][0][4][0][0])
    im_height  = int(mat['dataset'][0][i][0][0][4][0][1])
    bboxes = mat['dataset'][0][i][0][0][3]
    if sum(bboxes[:,2]<15)>0:
        continue
    if sum(bboxes[:,3]<15)>0:
        continue
    if (sum(im_width*threshold<bboxes[:,2]) == 0) and (sum(im_height*threshold<bboxes[:,3]) == 0):
        rec = {}
        rec['seq_name'] = str(mat['dataset'][0][i][0][0][0][0])
        rec['start_frame'] = int(mat['dataset'][0][i][0][0][1][0][0])
        rec['end_frame'] = int(mat['dataset'][0][i][0][0][2][0][0])
        rec['gt'] = bboxes.tolist()
        rec['im_width'] = im_width
        rec['im_height'] = im_height
        dataset.append(rec)

print(len(dataset))
with open(output_name, 'w') as f:
    json.dump(dataset, f)
