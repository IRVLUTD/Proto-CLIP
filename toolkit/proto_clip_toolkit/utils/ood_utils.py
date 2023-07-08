from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import random
import clip
import numpy as np
import torch
import sys

from .model_utils import load_pretrained_mb_and_adapters
from torchvision.datasets import ImageFolder

from pathlib import Path
p = Path(__file__).parents[3]

sys.path.append(str(p))
from utils import *
from imagenetv2_pytorch import ImageNetV2Dataset
from PIL import Image
from torch.utils.data import Dataset



class ImageNetSketchDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.images = self._load_images()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path, label = self.images[index]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def _load_images(self):
        images = []
        for cls_name in self.classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            if os.path.isdir(cls_dir):
                label = self.class_to_idx[cls_name]
                for filename in os.listdir(cls_dir):
                    image_path = os.path.join(cls_dir, filename)
                    images.append((image_path, label))
        return images


def test_ood_performance(cfg, test_dataset_name, n_workers, test_bs, memory_bank_v_path=None, memory_bank_t_path=None, adapter_type=None, adapter_weights_path=None):

    # CLIP
    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model.eval()

    # Prepare dataset; SEED is fetched from utils.py
    seed = get_seed()
    random.seed(seed)
    np.random.seed(seed)
    g = torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print("Preparing dataset.")
    if test_dataset_name=="imagenet_v2":
        test_dataset = ImageNetV2Dataset("matched-frequency", transform=preprocess)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_bs, num_workers=n_workers, shuffle=False)
    elif test_dataset_name=="imagenet_sketch":
        test_dataset = ImageFolder("./DATA/sketch", transform=preprocess)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_bs, num_workers=n_workers, shuffle=False)


    # Pre-load test features
    # print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)

    with torch.no_grad():
        print("Testing...")

        embeddings_v, embeddings_t, adapter = load_pretrained_mb_and_adapters(memory_bank_v_path=memory_bank_v_path, 
                                                                            memory_bank_t_path=memory_bank_t_path, 
                                                                            adapter_type=adapter_type, 
                                                                            adapter_weights_path=adapter_weights_path)
        NxK, ndim = embeddings_v.shape
        K = cfg['shots']
        N = NxK//K
        print(K, N)

        zs_imgs = embeddings_v.view(-1, K, ndim)
        zs_imgs = zs_imgs / zs_imgs.norm(dim=-1, keepdim=True)
        z_img_proto = zs_imgs.mean(dim=1)
        z_img_proto = z_img_proto / z_img_proto.norm(dim=-1, keepdim=True)

        zs_text = embeddings_t
        z_text_proto = zs_text / zs_text.norm(dim=-1, keepdim=True)

        test_features = adapter(test_features)
        test_features = test_features / test_features.norm(dim=-1, keepdim=True)

        p = P(test_features, z_img_proto, z_text_proto, cfg['alpha'], cfg['beta'])

        test_acc = (p.max(1)[1] == test_labels).float().mean() * 100.0

        return test_acc