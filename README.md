# Proto-CLIP: Vision-Language Prototypical Network for Few-Shot Learning

Code release for **Proto-CLIP** [ [Arxiv](https://arxiv.org/abs/2307.03073) | [Project-Page](https://irvlutd.github.io/Proto-CLIP) ]


![alt text](https://irvlutd.github.io/Proto-CLIP/assets/images/proto-clip/proto_clip.webp)


# Dataset
- To download the datasets, please follow the details in [DATASET.md](DATASET.md).
- To download the [FewSOL](https://irvlutd.github.io/FewSOL/) dataset variants [52 | 198], please use [this link](https://utdallas.box.com/v/proto-clip-fewsol-variants).
- **Note** : Please make sure to place all the datasets in `DATA/` directory.

# Setup
```sh
# create conda environment
conda create -n proto-clip python=3.9

# activate the environment
conda activate proto-clip

# install dependencies
pip install -r requirements.txt

# Install the according versions of torch and torchvision
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

# Alias
 
- | **Adapter** | **Adapter-Alias** |
  |-------------|-----------------|
  | 3xConv      | conv-3x         |
  | 2xConv      | conv-2x         |
  | MLP         | fc              |
- For details about adapter aliases, please check the [supplementary material](https://arxiv.org/src/2307.03073v1/anc/Proto_CLIP_supp.pdf).

- For dataset aliases, please check [datasets/\_\_init\_\_.py](datasets/__init__.py)
 

# Run
```sh

CUDA_VISIBLE_DEVICES=<GPU_ID> \
python main.py \
--config <configs-file> \
--dataset <dataset-alias> \
--logs tb_logs \
--alpha <alpha> \
--beta <beta> \
--adapter <adapter-alias> \
<vl-flag> \
<test-flag>
```

- `config-file` : Configuration file path for the experiment. Default config files are in `configs/` directory.
- `dataset-alias` : Alias of the dataset to be used for the experiment
- `alpha` : alpha hyperparameter for the selected dataset
- `beta` : beta hyperparameter for the selected dataset
- `adapter-alias` : adapter alias for the experiment
- `vl-flag` : To train text memory use `""` else `"--train_vis_memory_only"`
- `test-flag` : To train/test use `""`/`"--only_test"`. 

**Note:** Please use `main.qt.py` for experiments involving <strong style="font-variant: small-caps">Proto-CLIP-*F-Q<sup>T</sup>*</strong>.

# Tensorboard
```sh
tensorboard --logdir tb_logs
```

# Proto-CLIP Toolkit
<p align="center">
  <img src="media/real-world.gif">
</p>
<p align="center">
 Demo: User command oriented (<a href="https://fetchrobotics.borealtech.com/robotics-platforms/fetch-mobile-manipulator/?lang=en">Fetch</a>) robot grasping using <span style="font-variant: small-caps">Proto-CLIP</span> predictions. <br>For the real world demo, please use <a href="./toolkit/"> proto-clip-toolkit</a> (<a href="./toolkit/README.md">sample codes</a>). Please check the pypi package <a href="https://pypi.org/project/proto-clip-toolkit/">here</a>.<br>
 Please check the <a href="./pretrained_ckpt/">pretrained checkpoints</a> to use/work with the proto-clip-toolkit.<br><b>NOTE:</b> Use appropriate dataset w.r.t. the checkpoint.
</p>

# Links
- [Project Page](https://irvlutd.github.io/Proto-CLIP)
- Please check the FAQs [here](https://irvlutd.github.io/Proto-CLIP/#FAQs)
- Real World [Demo](https://irvlutd.github.io/Proto-CLIP#demo) | [Playlist](https://www.youtube.com/watch?v=CisrACRE7qE&list=PLgOC2wLNlACnuPrQV2Kxq2PtTAgUqdM-T)
- [Results](https://irvlutd.github.io/Proto-CLIP#jos-fsc) for Joint Object Segmentation and Few-Shot Classification in the Real World
- CLIP vs Proto-CLIP [t-SNE visualization](https://irvlutd.github.io/Proto-CLIP#clip-vs-proto-clip-t-sne)
- Barnes-Hut [t-SNE visualization](https://irvlutd.github.io/Proto-CLIP#t-sne) using Proto-CLIP-F trained on [FewSOL](https://irvlutd.github.io/FewSOL) [198 classes] dataset

# Contact
Following 3 options are available for any clarification, comments or suggestions
- Join the [discussion forum](https://github.com/IRVLUTD/Proto-CLIP/discussions).
- Inform an [issue](https://github.com/IRVLUTD/Proto-CLIP/issues).
- Contact [Jishnu](https://jishnujayakumar.github.io/).

# Citation
Please cite <span style="font-variant: small-caps">Proto-CLIP</span> if it helps your research:
```bibtex
@INPROCEEDINGS{padalunkal2024protoclip,
  author={P, Jishnu Jaykumar and Palanisamy, Kamalesh and Chao, Yu-Wei and Du, Xinya and Xiang, Yu},
  title={{Proto-CLIP: Vision-Language Prototypical Network for Few-Shot Learning}}, 
  keywords={Training;Representation learning;Adaptation models;Three-dimensional displays;Prototypes;Benchmark testing;Object recognition;Few shot learning;Intelligent robots},
  booktitle={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  doi={10.1109/IROS58592.2024.10801660},
  pages={2594-2601},
  year={2024}
}
```
