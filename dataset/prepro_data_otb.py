import os
import numpy as np
import pickle
from collections import OrderedDict

seqlist_path = 'otb-vot.txt'
output_path = 'otb-vot.pkl'

with open(seqlist_path,'r') as fp:
    seq_list = fp.read().splitlines()

data = {}
for i,seq in enumerate(seq_list):
    img_list = sorted(['img/'+p for p in os.listdir(seq+'/img') if os.path.splitext(p)[1] == '.jpg'])
    with open(seq+'/groundtruth_rect.txt') as f:
        gt = np.loadtxt((x.replace(',',' ') for x in f))

    assert len(img_list) == len(gt), "Lengths do not match!!"

    data[seq] = {'images':img_list, 'gt':gt}

with open(output_path, 'wb') as fp:
    pickle.dump(data, fp, -1)
