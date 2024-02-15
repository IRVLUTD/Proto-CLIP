from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import random
import argparse
import yaml
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn

from datasets import build_dataset
from datasets.utils import build_data_loader
import clip
from utils import *
from datasets.imagenet import ImageNet, get_random_train_tfm
from torch.utils.tensorboard import SummaryWriter
from model import Adapter, Adapter_FC, Adapter_LoRA, ProtoCLIP
import loralib as lora


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs', dest='logs_dir_path',
                        help='log directory path', required=False)
    parser.add_argument('--config', dest='config',
                        help='settings of Proto-CLIP in yaml format', required=True)
    parser.add_argument('--alpha', dest='alpha',
                        help='alpha', type=float, required=False)
    parser.add_argument('--beta', dest='beta', help='beta',
                        type=float, required=False)
    parser.add_argument('--adapter', dest='adapter',
                        help=f"adapter to use: ['conv-3x', 'conv-2x', 'fc']", type=str, required=False)
    parser.add_argument('--train_vis_memory_only', dest='train_vis_mem_only',
                        help='train visual memory only', action='store_true')
    parser.add_argument('--only_test', dest='only_test',
                        help='flag to perorm only testing', action='store_true')
    parser.add_argument('--shots', dest='shots',
                        help='shots in few-shot setups', type=int, required=False)
    parser.add_argument('--losses', nargs='+', dest='losses',
                        help="List of loss aliases: {'L1', 'L2', 'L3'}", required=False)
    parser.add_argument('--backbone', dest='backbone',
                        help='backbones: [RN50, RN101, ViT-B/16, ViT-B/32, ViT-L/14]', type=str, required=False)
    parser.add_argument('--dataset', dest='dataset',
                        help='dataset alias: [ caltech101, dtd, eurosat, fgvc, food101, imagenet, oxford_flowers, oxford_pets, stanford_cars, sun397, ucf101 ]', required=False)
    args = parser.parse_args()
    return args


def populate_cfg_using_args(cfg, args):
    # Set command-line arguments into config object
    if args.logs_dir_path:
        cfg['logs_dir_path'] = args.logs_dir_path
    if args.alpha:
        cfg['alpha'] = args.alpha
    if args.beta:
        cfg['beta'] = args.beta
    if args.adapter:
        cfg['adapter'] = args.adapter
    if args.shots:
        cfg['shots'] = args.shots
    if args.losses:
        cfg['losses'] = args.losses
    if args.backbone:
        cfg['backbone'] = args.backbone
    if args.dataset:
        cfg['dataset'] = args.dataset

    return cfg


def search_scale_step(cfg):
    """
    Sets the search scale and search step values for a given dataset configuration.

    Args:
        cfg (dict): A dictionary containing the experiment configuration.

    Returns:
        dict: The input dictionary with the search_scale and search_step values set.
    """
    dataset = cfg['dataset']
    dataset_dict = {'caltech101': ([12, 5], [200, 20]),
                    'dtd': ([13, 13], [200, 20]),
                    'eurosat': ([12, 10], [200, 20]),
                    'fgvc': ([30, 30], [200, 20]),
                    'food101': ([10, 10], [200, 20]),
                    'imagenet': ([7, 3], [200, 20]),
                    'oxford_flowers': ([50, 50], [200, 20]),
                    'oxford_pets': ([7, 3], [200, 20]),
                    'stanford_cars': ([20, 10], [200, 20]),
                    'sun397': ([12, 10], [200, 20]),
                    'ucf101': ([7, 3], [200, 20]),
                    # numbers are kept same as from clip, tip-a results in fewsol paper settings
                    'fewsol': ([13, 13], [200, 20])
                    }
    search_scale, search_step = dataset_dict.get(dataset, (None, None))
    cfg['search_scale'] = search_scale
    cfg['search_step'] = search_step
    return cfg


def run_proto_clip(cfg, visual_memory_keys, visual_memory_values, val_features, val_labels, test_features, test_labels, textual_memory_bank, clip_model, text_prompts, train_loader_F):

    ndim, NxK = visual_memory_keys.shape
    K = cfg['shots']
    N = NxK//K

    cfg = search_scale_step(cfg)
    torch.autograd.set_detect_anomaly(True)

    model = ProtoCLIP(clip_model, visual_memory_keys, textual_memory_bank, N, K, ndim, clip_model.dtype)

    # params = list(model.visual_embeddings.parameters()) + \
    #     list(model.textual_embeddings.parameters()) + \
    #     list(model.adapter.parameters())

    params = model.parameters()

    optimizer = torch.optim.AdamW(
        params, lr=cfg['lr'], eps=1e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg['train_epoch'] * NxK)

    best_acc, best_epoch = 0.0, 0

    model_dir_root = get_model_dir_root(cfg)
    os.makedirs(model_dir_root, exist_ok=True)

    # Create a SummaryWriter object
    writer = SummaryWriter(
        log_dir=f"{cfg['logs_dir_path']}/{model_dir_root}/{'_'.join(cfg['losses'])}/aug_{cfg['augment_epoch']}/epochs_{cfg['train_epoch']}")
    train_labels = torch.argmax(visual_memory_values, dim=1)


    if not cfg['only_test']:

        # define class sampling upper bound and lower bound
        class_upper = int(N * 0.9)
        class_lower = max(int(N * 0.4), 1)

        for epoch in tqdm(range(cfg['train_epoch'])):
            # Train
            model.visual_embeddings.train()
            model.textual_embeddings.train()
            model.adapter.train()

            correct_samples, all_samples = 0, 0
            loss_list = []
            print('Train Epoch: {:} / {:}'.format(epoch, cfg['train_epoch']))

            # create a permutation of class indexes
            class_indexes = np.random.permutation(N)
            start = 0
            while start < N-1:
                num_class = np.random.randint(class_lower, class_upper)
                class_index = sorted(
                    class_indexes[start:min(start + num_class, N-1)])
                num_class = len(class_index)

                # sample support and query
                support_index = []
                query_index = []
                zq_labels = []
                for i in range(num_class):
                    cls = class_index[i]
                    # sample number of support
                    assert K > 0
                    item_indexes = np.random.permutation(K)
                    n = np.random.randint(1, K) if K > 1 else K
                    support = sorted(item_indexes[:n])
                    if K > 1:
                        query = sorted(item_indexes[n:])
                    else:
                        query = sorted(item_indexes[:n])
                    support_index.extend(cls * K + support)
                    query_index.extend(cls * K + query)
                    zq_labels.extend([cls] * len(query))

                zq_imgs = visual_memory_keys.t()[query_index]  # N_qxC
                p, z_img_proto, z_text_proto = model(zq_imgs)
                zq_labels = torch.as_tensor(zq_labels).cuda()
                matches, train_loss, neg_log_loss, img2txt_align_loss, txt2img_align_loss, img_inter_cluster_loss, txt_inter_cluster_loss = \
                    compute_loss_and_matches(
                        p, zq_labels, z_img_proto, z_text_proto, cfg)

                mode = 'train'
                if neg_log_loss is not None:
                    writer.add_scalar(
                        f'Loss/{mode}/L1-negLog', neg_log_loss, epoch)
                if img2txt_align_loss is not None:
                    writer.add_scalar(
                        f'Loss/{mode}/L2-img2txt_align', img2txt_align_loss, epoch)
                if txt2img_align_loss is not None:
                    writer.add_scalar(
                        f'Loss/{mode}/L3-txt2img_align', txt2img_align_loss, epoch)
                if img_inter_cluster_loss is not None:
                    writer.add_scalar(
                        f'Loss/{mode}/L4-img_inter_cluster', img_inter_cluster_loss, epoch)
                if txt_inter_cluster_loss is not None:
                    writer.add_scalar(
                        f'Loss/{mode}/L5-txt_inter_cluster', txt_inter_cluster_loss, epoch)

                correct_samples += matches
                all_samples += len(zq_labels)
                loss_list.append(train_loss.item())

                optimizer.zero_grad()
                train_loss.backward(retain_graph=True)
                optimizer.step()

                # advance class sampling
                start += len(class_index)

            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            train_acc = correct_samples / all_samples
            train_loss = sum(loss_list)/len(loss_list)
            print('LR: {:.6f}, Acc: {:.4f}% ({:}/{:}), Loss: {:.4f}'.format(
                current_lr, train_acc*100, correct_samples, all_samples, train_loss))
            # print(model.adapter.ratio, model.alpha, model.beta)
            
            with torch.no_grad():
                zs_imgs = model.visual_embeddings.weight.view(-1, K, ndim)
                zs_imgs = zs_imgs / zs_imgs.norm(dim=-1, keepdim=True)
                z_img_proto = zs_imgs.mean(dim=1)
                z_img_proto = z_img_proto / \
                    z_img_proto.norm(dim=-1, keepdim=True)

                zs_text = model.textual_embeddings(
                    torch.arange(N, requires_grad=False).cuda())
                z_text_proto = zs_text / zs_text.norm(dim=-1, keepdim=True)

                val_features_adapt = model.adapter(val_features)
                val_features_adapt = val_features_adapt / \
                    val_features_adapt.norm(dim=-1, keepdim=True)

                
                
                p, z_img_proto, z_text_proto = model(val_features_adapt)

                pred_p, y_hat = p.max(dim=1)
                matches = (y_hat == val_labels).float().sum()
                neg_log_loss_val = -torch.log(pred_p).mean()

                val_acc = (p.max(1)[1] == val_labels).float().mean()

                print("**** Proto-CLIP's val accuracy: {:.2f}% | loss: {:.2f}***\n".format(
                    val_acc*100, neg_log_loss_val))

                model_dir_root = get_model_dir_root(cfg)

                model_dir = f"{model_dir_root}/alpha-beta/learned"
                model_prefix = f"best_lr_{cfg['lr']}_aug_{cfg['augment_epoch']}_epochs_{cfg['train_epoch']}"

                os.makedirs(model_dir, exist_ok=True)
                model_save_path = os.path.join(model_dir, f"{model_prefix}.pt")
                if val_acc >= best_acc:
                    best_acc = val_acc
                    best_epoch = epoch                    
                    torch.save(model.state_dict(), model_save_path)

                # Log to tensorboard
                writer.add_scalar('Loss/val', neg_log_loss_val, epoch)
                writer.add_scalar('Accuracy/val', val_acc, epoch)

            # Log to tensorboard
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('HP/lr', current_lr, epoch)

        print(
            f"Best model: best_val_acc = {best_acc*100: .2f}, best_val_epoch = {best_epoch}")

    print("Testing...")
    model_dir = f"{model_dir_root}/alpha-beta/learned"
    model_prefix = f"best_lr_{cfg['lr']}_aug_{cfg['augment_epoch']}_epochs_{cfg['train_epoch']}"


    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    zs_imgs = model.visual_embeddings.weight.view(-1, K, ndim)
    zs_imgs = zs_imgs / zs_imgs.norm(dim=-1, keepdim=True)
    z_img_proto = zs_imgs.mean(dim=1)
    z_img_proto = z_img_proto / z_img_proto.norm(dim=-1, keepdim=True)

    zs_text = model.textual_embeddings.weight
    z_text_proto = zs_text / zs_text.norm(dim=-1, keepdim=True)

    test_features = model.adapter(test_features)
    test_features = test_features / \
        test_features.norm(dim=-1, keepdim=True)


    p, z_img_proto, z_text_proto = model(test_features)

    test_acc = (p.max(1)[1] == test_labels).float().mean()

    print(
        "**** Fixed-alp-beta: Proto-CLIP's test accuracy: {:.2f}% ****\n".format(test_acc*100))
    print('alpha-beta learned')

    plot_tsne(model_dir_root, z_img_proto, z_text_proto,
                test_features, text_prompts, cfg, writer)

    # Log to tensorboard
    writer.add_scalar(
        'Accuracy/zsval-zstestval-zstest-3F-test', test_acc, 7)

    # Close the SummaryWriter object
    writer.close()


def seed_worker(worker_id):
    worker_seed = get_seed()
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main():

    # Load config file
    args = get_arguments()
    assert (os.path.exists(args.config))

    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    if args.dataset is None:
        raise SystemExit("Please provide alias of dataset")

    cfg = populate_cfg_using_args(cfg, args)

    cache_dir = os.path.join('./caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    print("\nRunning configs.")
    print(cfg, "\n")

    # CLIP
    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model.eval()

    # Prepare dataset; SEED is fetched from utils.py
    seed = get_seed()
    random.seed(seed)
    np.random.seed(seed)
    g = torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    n_workers, train_bs, val_bs, test_bs = 8, 1024, 1024, 1024

    print("Preparing dataset.")
    if cfg['dataset'] == 'imagenet':
        dataset = ImageNet(cfg['root_path'], cfg['shots'], preprocess)
        train_loader_cache = torch.utils.data.DataLoader(
            dataset.train, batch_size=train_bs, num_workers=n_workers, shuffle=False, worker_init_fn=seed_worker, generator=g)
        val_loader = torch.utils.data.DataLoader(
            dataset.test, batch_size=val_bs, num_workers=n_workers, shuffle=False)
        test_loader = torch.utils.data.DataLoader(
            dataset.test, batch_size=test_bs, num_workers=n_workers, shuffle=False)
    else:
        dataset = build_dataset(cfg['dataset'], cfg['root_path'], cfg['shots'])
        train_tranform = get_random_train_tfm()
        train_loader_cache = build_data_loader(data_source=dataset.train_x, batch_size=train_bs,
                                               tfm=train_tranform, is_train=True, shuffle=False, worker_init_fn=seed_worker, generator=g)
        train_loader_F = build_data_loader(data_source=dataset.train_x, batch_size=256,
                                    tfm=train_tranform, is_train=True, shuffle=True, worker_init_fn=seed_worker, generator=g)
        val_loader = build_data_loader(
            data_source=dataset.val, batch_size=val_bs, is_train=False, tfm=preprocess, shuffle=False)
        test_loader = build_data_loader(
            data_source=dataset.test, batch_size=test_bs, is_train=False, tfm=preprocess, shuffle=False)

    # Construct the cache model by few-shot training set
    print("Constructing memory bank by few-shot visual and textual features.")

    visual_memory_keys, visual_memory_values = build_cache_model(
        cfg, clip_model, train_loader_cache)

    # Textual features
    text_prompts, textual_memory_bank = get_textual_memory_bank(
        cfg, dataset.classnames, dataset.template, clip_model)

    # Load/Pre-load val features
    print("Loading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(
        cfg, "val", clip_model, val_loader)

    # Load/Pre-load test features
    print("Loading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(
        cfg, "test", clip_model, test_loader)

    # ------------------------------------------ Proto-CLIP ------------------------------------------
    run_proto_clip(cfg, visual_memory_keys, visual_memory_values, val_features, val_labels,
                   test_features, test_labels, textual_memory_bank, clip_model, text_prompts, train_loader_F)


if __name__ == '__main__':
    main()
