# Proto-CLIP: Vision-Language Prototypical Network for Few-Shot Learning

## This is the official implementation for [Proto-CLIP](https://irvlutd.github.io/Proto-CLIP).

![alt text](https://irvlutd.github.io/Proto-CLIP/assets/images/proto-clip/proto_clip.webp)


# Dataset
- To download the datasets, please follow the details in [DATASET.md](DATASET.md).
- To download the [FewSOL](https://irvlutd.github.io/FewSOL/) dataset variants [52 | 198], please use [this link](https://utdallas.box.com/v/proto-clip-fewsol-variants).
- **Note** : Please make sure to place all the datasets in `DATA/` directory.

# Install dependencies
```sh
conda create -n proto-clip
conda activate proto-clip

pip install -r requirements.txt

# Install the according versions of torch and torchvision
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

# Alias

 - For adapter aliases, please check the supplementary material # TODO: place arxiv supplementary link
 - For dataset aliases, please check [datasets/\_\_init\_\_.py](datasets/__init__.py)

# Run
```sh

CUDA_VISIBLE_DEVICES=<$gpu_id> python main.py \
--config <configs-file> --dataset <dataset-alias> \
--logs tb_logs --alpha <alpha> --beta <beta> \
--adapter <adapter-alias> <vl_flag> <test_flag>
```

- `config-file` : Configuration file path for the experiment. Default config files are in `configs/` directory.
- `dataset-alias` : Alias of the dataset to be used for the experiment
- `alpha` : alpha hyperparameter for the selected dataset
- `beta` : beta hyperparameter for the selected dataset
- `adapter-alias` : adapter alias for the experiment
- `vl-flag` : To train text memory use `""` else `"--train_vis_memory_only"`
- `test-flag` : To train/test use `""`/`"--only_test"`. 

Note: Please use `main.qt.py` for experiments involving <strong style="font-variant: small-caps">Proto-CLIP-F-Q<sup>T</sup></strong>.

# Tensorboard
```sh
tensorboard --logdir tb_logs
```

# Citation
Please cite the following if you incorporate our work.
```bibtex
Coming Soon. Thanks for your patience.
```

# Proto-CLIP Toolkit
<p align="center">
  <img src="media/real-world.gif">
</p>
<p align="center">
 Demo: User command oriented robot (<a href="https://fetchrobotics.borealtech.com/robotics-platforms/fetch-mobile-manipulator/?lang=en">Fetch</a>) grasping using <span style="font-variant: small-caps">Proto-CLIP</span> predictions. <br>For the real world demo, please use <a href="toolkit/"> proto-clip-toolkit</a>.
</p>

# Links
- [Project Page](https://irvlutd.github.io/Proto-CLIP)
- Please check the FAQs [here](#FAQs)
- Real World [Demo](https://irvlutd.github.io/Proto-CLIP#demo) | [Playlist](https://www.youtube.com/watch?v=CisrACRE7qE&list=PLgOC2wLNlACnuPrQV2Kxq2PtTAgUqdM-T)
- [Results](https://irvlutd.github.io/Proto-CLIP#jos-fsc) for Joint Object Segmentation and Few-Shot Classification in the Real World
- CLIP vs Proto-CLIP [t-SNE visualization](https://irvlutd.github.io/#clip-vs-proto-clip-t-sne)
- Barnes-Hut [t-SNE visualization](https://irvlutd.github.io/Proto-CLIP#t-sne) using fine-tuned Proto-CLIP trained on [FewSOL](https://irvlutd.github.io/FewSOL) [198 classes] dataset

# Contact
Following 3 options are available for any clarification, comments or suggestions
- Join the [discussion forum](https://github.com/IRVLUTD/Proto-CLIP/discussions).
- Inform an [issue](https://github.com/IRVLUTD/Proto-CLIP/issues).
- Contact [Jishnu](https://jishnujayakumar.github.io/).
