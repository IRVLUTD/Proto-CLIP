Configs for the pretrained checkpoint **[ FewSOL-198 ]**
```
# ------ root_path/dataset_name ------
root_path: 'DATA'

# ------ Basic Config ------
shots: 16
backbone: 'ViT-L/14'
dataset: 'fewsol'
only_test: True

lr: 0.0001
augment_epoch: 10
train_epoch: 2000

alpha: 0.2
beta: 12

adapter: 'fc'
train_vis_mem_only: True

losses: ['L1', 'L2', 'L3']
```