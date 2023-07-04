import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as TF
import argparse
import yaml
import random
import sys

sys.path.append("../../../../proto-clip")
from proto_datasets import build_dataset
from sklearn.manifold import TSNE
from PIL import Image
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from utils import build_cache_model, get_textual_memory_bank
from scipy.io import savemat
import json
import cv2
import matplotlib as mpl

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of Proto-CLIP in yaml format', required=True)
    parser.add_argument('--memory_bank_v_path', dest='memory_bank_v_path', help='path to the visual embeddings memory bank', required=True)
    parser.add_argument('--memory_bank_t_path', dest='memory_bank_t_path', help='path to the textual embeddings memory bank', required=True)
    parser.add_argument('--after_train', dest='after_train', help='save embeddings after training', action='store_true')
    args = parser.parse_args()
    return args


def parse_splits_file(config_path):
    """Returns the map class id to the class name using the splits json file."""

    class_id_mapping = {}
    f = open(config_path)
    data = json.load(f)

    for config_data in data["train"]:
        class_id_mapping[config_data[1]] = config_data[2]
    
    return class_id_mapping

def get_image_samples(txt_path, n_classes):
    """Returns a single representative image sample for each class."""
    f = open(txt_path, "r")
    data = f.readlines()

    data = [x.strip("\n") for x in data]
    output_image_locations = []

    for i in range(n_classes):
        #We pick the first sample of the support set for a given class.
        img_idx = i*16

        output_image_locations.append(data[img_idx])

    return output_image_locations

def get_tsne_coordinates(z_img_proto, z_text_proto, n_class):
    """Returns the 2-dimensional t-SNE coordinates for the image and the text embeddings."""
    X = torch.vstack((
        z_img_proto,
        z_text_proto,
        # zq_imgs.view(-1, zq_imgs.shape[-1])
    )).cpu().data.numpy()
    tsne_X = TSNE(n_components=2, perplexity=10, random_state=1).fit_transform(X)
    zi, zt = tsne_X[:n_class], tsne_X[n_class: ]

    return zi, zt

def plot_tsne_after(z_img_proto, z_text_proto, txt_prompts):
    """Returns the t-SNE plot for the visual and textual embeddings after training."""
    n_class = z_img_proto.shape[0]
    zi, zt = get_tsne_coordinates(z_img_proto, z_text_proto, n_class)

    image_locations = get_image_samples("./image_locations.txt", n_class)
    _, ax = plt.subplots(figsize=(50, 50))
    fontsize = 10
    for idx, (x, y) in enumerate(zip(zi[:, 0], zi[:, 1])):
        img = plt.imread(image_locations[idx])
        img = cv2.resize(img, (48, 48))

        imagebox = OffsetImage(img)  # Adjust the zoom level as desired
        ab = AnnotationBbox(imagebox, (x, y), frameon=False, zorder=1)
        
        # ax.set_aspect('equal')
        ax.scatter(x, y, zorder=4, s=32, c='cyan', marker=".")
        ax.add_artist(ab)
        ax.annotate(
            txt_prompts[idx], xy=(x, y + 1), ha='center', c="crimson", fontsize=fontsize)
    
    ax.scatter(zt[:, 0], zt[:, 1], c='aquamarine', zorder=3, marker="+", s=128)

    ax.axis('off')

    plt.savefig("./test_plot_after.png", dpi=300)

def plot_tsne_before(z_img_proto, z_text_proto, txt_prompts):
    """Returns the t-SNE plot for the visual and textual embeddings before training."""
    n_class = z_img_proto.shape[0]
    zi, zt = get_tsne_coordinates(z_img_proto, z_text_proto, n_class)

    image_locations = get_image_samples("./image_locations.txt", n_class)
    _, ax = plt.subplots(figsize=(50, 50))
    fontsize = 25
    for idx, (x, y) in enumerate(zip(zi[:, 0], zi[:, 1])):
        img = plt.imread(image_locations[idx])
        img = cv2.resize(img, (48, 48))

        imagebox = OffsetImage(img)  # Adjust the zoom level as desired
        ab = AnnotationBbox(imagebox, (x, y), frameon=False, zorder=1)
        
        ax.scatter(x, y, zorder=4, s=32, c='cyan', marker=".")
        ax.add_artist(ab)
    
    ax.scatter(zt[:, 0], zt[:, 1], c='lightseagreen', zorder=3, marker="P", s=128)


    for i in range(len(txt_prompts)):    
        ax.annotate(
            txt_prompts[i], (zt[i, 0], zt[i, 1] + 0.2), c='crimson', fontsize=fontsize)

    ax.axis('off')

    plt.savefig("./test_plot_before.png", dpi=300)

if __name__=="__main__":
    args = get_arguments()

    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    cache_dir = os.path.join('./caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    dataset = build_dataset(cfg['dataset'], cfg['root_path'], cfg['shots'])

    if not args.after_train:
        #This function will not work if you do not have the cache available.
        #You would need to create the cache and place the initial embeddings inside it. 
        #An easy way to do it is to run the main.py for 1 epoch.
        visual_memory_keys, visual_mem_values = build_cache_model(cfg, None, None)

        text_prompts, textual_memory_bank = get_textual_memory_bank(cfg, [], None, None)
        
        embeddings_v = visual_memory_keys.t()
        embeddings_t = textual_memory_bank.t()
    else:
        best_model_path_v = args.memory_bank_v_path
        best_model_path_t = args.memory_bank_t_path

        try:
            embeddings_v = torch.load(best_model_path_v)
            embeddings_t = torch.load(best_model_path_t)
        except:
            raise FileNotFoundError(f"File does not exist: {best_model_path_v} and {best_model_path_t}") 
    
    
    NxK, ndim= embeddings_v.shape
    K = cfg['shots']
    N = NxK//K

    zs_imgs = embeddings_v.view(-1, K, ndim)
    zs_imgs = zs_imgs / zs_imgs.norm(dim=-1, keepdim=True)
    z_img_proto = zs_imgs.mean(dim=1)
    z_img_proto = z_img_proto / z_img_proto.norm(dim=-1, keepdim=True)
    
    zs_text = embeddings_t
    z_text_proto = zs_text / zs_text.norm(dim=-1, keepdim=True)
        
    class_id_mapping = parse_splits_file("../../datasets/fewsol_splits_198.json")

    if args.after_train:
        plot_tsne_after(z_img_proto, z_text_proto, list(class_id_mapping.values()))
    else:
        plot_tsne_before(z_img_proto, z_text_proto, list(class_id_mapping.values()))