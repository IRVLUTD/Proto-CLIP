import torch
import os
import sys

from pathlib import Path
p = Path(__file__).parents[3]

sys.path.append(str(p))
from utils import *
from model import Adapter, Adapter_FC

def load_pretrained_mb_and_adapters(config=None, memory_bank_v_path=None, memory_bank_t_path=None, adapter_type=None, adapter_weights_path=None):
    """Returns the pretrained visual embeddings and textual embeddings from the memory bank and the pretrained query adapter"""
    if config:
        with torch.no_grad():
            model_dir_root = get_model_dir_root(config)
            model_dir = f"{model_dir_root}/alpha-beta/{config['alpha']}-{config['beta']}"
            model_prefix = f"best_lr_{config['lr']}_aug_{config['augment_epoch']}_epochs_{config['train_epoch']}"

            best_model_path_v = os.path.join(model_dir, f"{model_prefix}_v.pt")
            best_model_path_t = os.path.join(model_dir, f"{model_prefix}_t.pt")
            best_model_path_a = os.path.join(model_dir, f"{model_prefix}_a.pt")

            try:
                embeddings_v = torch.load(best_model_path_v)
                embeddings_t = torch.load(best_model_path_t)
            except:
                raise FileNotFoundError(f"File does not exist: {best_model_path_v} and {best_model_path_t}")

            NxK, ndim = embeddings_v.shape

            if 'conv' in config['adapter']:
                adapter = Adapter(ndim, c_type=config['adapter'], dtype=torch.half).cuda()
            elif config['adapter'] == 'fc':
                adapter = Adapter_FC(ndim, dtype=torch.half).cuda()

            try:
                adapter.load_state_dict(torch.load(best_model_path_a))
            except:
                raise FileNotFoundError(f"File does not exist: {best_model_path_a}")
    else:
        with torch.no_grad():
            try:
                embeddings_v = torch.load(memory_bank_v_path)
                embeddings_t = torch.load(memory_bank_t_path)
            except:
                raise FileNotFoundError(f"File does not exist: {memory_bank_v_path} and {memory_bank_t_path}")

            NxK, ndim = embeddings_v.shape

            if adapter_type==None:
                raise Exception("Please mention the adapter type in the args or in the config file.")
                
            if 'conv' in adapter_type:
                adapter = Adapter(ndim, c_type=config['adapter'], dtype=torch.half).cuda()
            elif adapter_type == 'fc':
                adapter = Adapter_FC(ndim, dtype=torch.half).cuda()

            try:

                adapter_state_dict = torch.load(adapter_weights_path)
            except:
                raise FileNotFoundError(f"File does not exist: {best_model_path_a}")
            
            adapter.load_state_dict(adapter_state_dict)
    
    return embeddings_v, embeddings_t, adapter

def pre_load_features_without_cache(clip_model, loader):

    """Returns the clip features for images without caching them."""
    features = []
    with torch.no_grad():
        for i, images in enumerate(tqdm(loader)):
            images= images.cuda()
            image_features = clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            features.append(image_features)

    features = torch.cat(features)

    return features

