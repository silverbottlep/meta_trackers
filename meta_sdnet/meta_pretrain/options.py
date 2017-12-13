from collections import OrderedDict

opts = OrderedDict()
opts['use_gpu'] = True

opts['batch_pos'] = 32 
opts['batch_neg'] = 96 
opts['overlap_pos'] = [0.7, 1]
opts['overlap_neg'] = [0, 0.5]
opts['img_size'] = 107
opts['padding'] = 16

opts['meta_init_lr'] = 0.0001
opts['meta_alpha_lr'] = 0.00001
opts['meta_beta_lr'] = 0.00001
opts['meta_batch_size'] = 8
opts['tracker_init_lr'] = 0.0001
opts['n_init_updates'] = 1
