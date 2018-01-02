import json
from matplotlib import pyplot as plt
import argparse
import os
import sys
import numpy as np
from utils import *
        
def compute_success_overlap(gt_bb, result_bb):
    thresholds_overlap = np.arange(0,1.05,0.05)
    n_frame = len(gt_bb)
    success = np.zeros(len(thresholds_overlap))
    iou = np.zeros(n_frame)
    for i in range(n_frame):
        iou[i] = overlap_ratio(gt_bb[i], result_bb[i])
    for i in range(len(thresholds_overlap)):
        success[i] = sum(iou > thresholds_overlap[i])/n_frame
    return success

def compute_success_error(gt_center, result_center):
    thresholds_error = np.arange(0,51,1)
    n_frame = len(gt_center)
    success = np.zeros(len(thresholds_error))
    dist = np.zeros(n_frame)
    for i in range(n_frame):
        dist[i] = np.sqrt(np.power(gt_center[i]-result_center[i],2).sum())
    for i in range(len(thresholds_error)):
        success[i] = sum(dist <= thresholds_error[i])/n_frame
    return success

def get_gt_bb(seq):
    gt_path = os.path.join('../dataset/OTB/', seq, 'groundtruth_rect.txt')
    with open(gt_path) as f:
        gt = np.loadtxt((x.replace(',',' ') for x in f))
    return gt

def get_result_bb(seq, arch):
    result_path = './result/otb/' + seq + '_' + arch + '.json'
    with open(result_path, 'r') as f:
        temp = json.load(f)
    return np.array(temp['results'][0]['res'])

def convert_bb_to_center(bboxes):
    return np.array([(bboxes[:,0]+(bboxes[:,2]-1)/2),
                      (bboxes[:,1]+(bboxes[:,3]-1)/2)]).T

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seq', default='', help='input seq')
    args = parser.parse_args()

    data_dir = '../dataset/OTB'
    list_path = os.path.join(data_dir,'tb_100.txt')
    seqs = []
    with open(list_path) as f:
        content = f.readlines()
    for line in content:
        parsed_line = line.split()
        seqs.append(parsed_line[0])

    trackers = []
    trackers.append('init01-lr0.000007')

    n_seq = len(seqs)
    thresholds_overlap = np.arange(0,1.05,0.05)
    thresholds_error = np.arange(0,51,1)

    success_overlap = np.zeros((n_seq,len(trackers),len(thresholds_overlap)))
    success_error = np.zeros((n_seq,len(trackers),len(thresholds_error)))
    for i in range(n_seq):
        seq = seqs[i]
        gt_bb = get_gt_bb(seq)
        gt_center = convert_bb_to_center(gt_bb)
        print('processing %s'%seq)
        for j in range(len(trackers)):
            tracker = trackers[j]
            bb = get_result_bb(seq, tracker)
            center = convert_bb_to_center(bb)
            success_overlap[i][j] = compute_success_overlap(gt_bb,bb)
            success_error[i][j] = compute_success_error(gt_center,center)

    print('Success Overlap')
    for i in range(len(trackers)):
        print('%s(%.4f)'%(trackers[i],success_overlap[:,i,:].mean()))
    print('Success Error')
    for i in range(len(trackers)):
        print('%s(%.4f)'%(trackers[i],success_error[:,i,:].mean(0)[20]))
    print('Finished')
