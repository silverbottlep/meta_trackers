# MetaTrackers

## 0. Prerequisites
[PyTorch](http://pytorch.org/) >= v0.2.0

## 0. Dataset download
Download dataset([OTB](https://sites.google.com/site/benchmarkpami/), [VOT](http://www.votchallenge.net/)) and prepared the link to the dataset in $(meta_trackers_root)/dataset/ directory.
```bash
$(meta_trackers_root)/dataset/VID
$(meta_trackers_root)/dataset/OTB
$(meta_trackers_root)/dataset/vot2013
$(meta_trackers_root)/dataset/vot2015
$(meta_trackers_root)/dataset/vot2016
```

## 0. Prepare dataset meta files
I already prepared all necessary meta-files for ILSVRC VID dataset, OTB, VOT dataset. Either you use them or you could generate via scripts in $(meta_trackers_root)/dataset/ directory.
```bash
$(meta_trackers_root)/dataset/ilsvrc_train.json # meta file for loading ILSVRC VID dataset to meta-train.
$(meta_trackers_root)/dataset/vot-otb.pkl # meta file for loading VOT dataset to meta-train(for OTB experiments)
$(meta_trackers_root)/dataset/otb-vot.pkl # meta file for loading OTB dataset to meta-train(for VOT experiments)
```
Also need to download imagenet pretrained models(our base feature extractors) into $(meta_trackers_root)/meta_crest(and meta_sdnet)/models/. (We used the same networks that original trackers used. For meta_sdnet - [imagenet-vgg-m.mat](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-m.mat), and for meta_crest - [imagenet-vgg-verydeep-16.mat](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat))

## 1. Meta-Training
You can skip this step and download pretrain models, and use them to test the trackers. If you want to meta-train MetaCREST trackers,
```bash
$(meta_trackers_root)/meta_crest/meta_pretrain$> python train_meta_init.py -e OTB # for OTB experiments, for VOT use -e VOT
```
To meta-train MetaSDNet trackers,
```bash
$(meta_trackers_root)/meta_sdnet/meta_pretrain$> python train_meta_init.py -e OTB # for OTB experiments, for VOT use -e VOT
```

## 2. Downloading pretrained models
We provide pretrained models for both meta trackers for your convenience. You can download it from following links and locate them in models directory.

[$(meta_trackers_root)/meta_sdnet/models/meta_init_vot_ilsvrc.pth](https://drive.google.com/file/d/1hQm9pHO_FJDceAcX_DDqoiZvNpXUfJ5D/view?usp=sharing) (~35M)

[$(meta_trackers_root)/meta_sdnet/models/meta_init_otb_ilsvrc.pth](https://drive.google.com/file/d/1y5Iqd40G6CrRZTeY2zHybO49qfUmGjXE/view?usp=sharing) (~35M)

[$(meta_trackers_root)/meta_crest/models/meta_init_vot_ilsvrc.pth](https://drive.google.com/file/d/1pBiVFaoi1kjK_COnQxZehmLvA59o3PGt/view?usp=sharing) (~59K)

[$(meta_trackers_root)/meta_crest/models/meta_init_otb_ilsvrc.pth](https://drive.google.com/file/d/1THmp-FdUPu2lzueSJlXsMOPBpe7QYh2q/view?usp=sharing) (~59K)


## 3. Testing MetaTrackers
```bash
$(meta_trackers_root)/meta_crest/meta_tracking$>python run_tracker.py # meta_crest tracker for OTB experiments
$(meta_trackers_root)/meta_sdnet/meta_tracking$>python run_tracker.py # meta_sdnet tracker for OTB experiments
```
To run VOT2016 experiments, I provided following VOT integration files. You can use them and run it via VOT2016 toolkit. Please refer to [VOT homepage](http://votchallenge.net/howto/)
```bash
$(meta_trackers_root)/meta_crest/meta_tracking/run_tracker_vot.py
$(meta_trackers_root)/meta_sdnet/meta_tracking/run_tracker_vot.py
```

## 4. Evaluations
If you used pre-trained models, you should be able to get same results(or small variation due to randomness in trackers) reported in the papers. If you meta-trained the model, you should also be able to get similar results.
```bash
$(meta_trackers_root)/meta_crest$> python eval_otb.py 
$(meta_trackers_root)/meta_sdnet$> python eval_otb.py
```
Similarly, please refer to [VOT homepage](http://votchallenge.net/howto/) for VOT evaluations. I also provided all raw results for both OTB and VOT experiments that used in the paper([meta_crest_result](https://drive.google.com/file/d/18PfjMJ21ldKkBfUQpaa8TK4PHU5lh9mx/view?usp=sharing), [meta_sdnet_result](https://drive.google.com/file/d/10tTaiO2-hyggjKyuwlgTG34bFQkIyxgF/view?usp=sharing))


## Acknowledgments
Many parts of this code are adopted from other related works([pytorch-maml](https://github.com/katerakelly/pytorch-maml), [py-MDNet](https://github.com/HyeonseobNam/py-MDNet), [CREST](https://github.com/ybsong00/CREST-Release)).
