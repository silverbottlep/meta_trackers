from collections import OrderedDict

opts = OrderedDict()
opts['use_gpu'] = True

opts['img_size'] = 107
opts['padding'] = 16

opts['batch_pos'] = 32
opts['batch_neg'] = 96
opts['batch_neg_cand'] = 1024
opts['batch_test'] = 256

opts['n_samples'] = 256
opts['trans_f'] = 0.6
opts['scale_f'] = 1.05
opts['trans_f_expand'] = 1.5

opts['n_bbreg'] = 1000
opts['overlap_bbreg'] = [0.6, 1]
opts['scale_bbreg'] = [1, 2]

# hyper-parameters for history management
opts['success_thr'] = 0
opts['n_frames_short'] = 20
opts['n_frames_long'] = 100
opts['long_interval'] = 10

opts['overlap_pos_update'] = [0.7, 1]
opts['overlap_neg_update'] = [0, 0.3]
opts['w_decay'] = 0.0005
opts['momentum'] = 0.9

opts['n_pos_update'] = 50 #5 #50
opts['n_neg_update'] = 200 #20 #200
opts['n_pos_init'] = 32 #500
opts['n_neg_init'] = 96 #5000
opts['overlap_pos_init'] = [0.7, 1]
opts['overlap_neg_init'] = [0, 0.5]
