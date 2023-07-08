
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
from model import Adapter, Adapter_FC


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


def run_proto_clip(cfg, visual_memory_keys, visual_memory_values, val_features, val_labels, test_features, test_labels, textual_memory_bank, clip_model, text_prompts, train_loader_F):

    # Enable the visual_memory_keys to be learnable
    ndim, NxK = visual_memory_keys.shape
    K = cfg['shots']
    N = NxK//K

    visual_embeddings = nn.Embedding(
        num_embeddings=NxK, embedding_dim=ndim).cuda().to(clip_model.dtype)
    visual_embeddings.weight = nn.Parameter(visual_memory_keys.t().clone())

    if 'conv' in cfg['adapter']:
        adapter = Adapter(ndim, c_type=cfg['adapter'], dtype=torch.half).cuda()
    elif cfg['adapter'] == 'fc':
        adapter = Adapter_FC(ndim, dtype=torch.half).cuda()

    textual_embeddings = nn.Embedding(
        num_embeddings=N, embedding_dim=ndim).cuda().to(clip_model.dtype)
    textual_embeddings.weight = nn.Parameter(textual_memory_bank.t().clone())

    if cfg['train_vis_mem_only']:
        params = list(adapter.parameters()) + \
            list(visual_embeddings.parameters())
    else:
        params = list(visual_embeddings.parameters(
        )) + list(textual_embeddings.parameters()) + list(adapter.parameters())

    optimizer = torch.optim.AdamW(
        params, lr=cfg['lr'], eps=1e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg['train_epoch'] * NxK)

    best_acc, best_epoch = 0.0, 0

    # alpha and beta search range
    step_size = 0.1
    alpha_list = np.arange(0, 1+step_size, step_size)
    beta_list = np.concatenate(
        (np.arange(0.1, 1, 0.1),  np.arange(1, 21, 1.0)))

    val_acc_list = []
    test_acc_list = []
    train_acc_list = []

    # search for best (alpha, beta) using zero shot valicdation accuracy on val set
    model_dir_root = get_model_dir_root(cfg)
    os.makedirs(model_dir_root, exist_ok=True)
    val_path = os.path.join(
        model_dir_root, f"zero_shot_hp_search_val_{beautify(cfg['backbone'])}_K_{cfg['shots']}.pkl")
    test_path = os.path.join(
        model_dir_root, f"zero_shot_hp_search_test_{beautify(cfg['backbone'])}_K_{cfg['shots']}.pkl")
    train_path = os.path.join(
        model_dir_root, f"zero_shot_hp_search_train_{beautify(cfg['backbone'])}_K_{cfg['shots']}.pkl")

    # Create a SummaryWriter object
    writer = SummaryWriter(
        log_dir=f"{cfg['logs_dir_path']}/{model_dir_root}/{'_'.join(cfg['losses'])}/aug_{cfg['augment_epoch']}/epochs_{cfg['train_epoch']}")

    train_labels = torch.argmax(visual_memory_values, dim=1)

    if os.path.exists(val_path) and os.path.exists(test_path) and os.path.exists(train_path):
        val_acc_list = load(val_path, 'hp based on val set')
        test_acc_list = load(test_path, 'hp based on test set')
        train_acc_list = load(train_path, 'hp based on test set')
    else:
        with torch.no_grad():
            z_img_proto = visual_memory_keys.t().view(-1, K, ndim).mean(dim=1)
            z_text_proto = textual_memory_bank.t()

            z_img_proto = z_img_proto / z_img_proto.norm(dim=-1, keepdim=True)
            z_text_proto = z_text_proto / \
                z_text_proto.norm(dim=-1, keepdim=True)
            train_features = visual_memory_keys.t(
            ) / visual_memory_keys.t().norm(dim=-1, keepdim=True)

            val_features = val_features / \
                val_features.norm(dim=-1, keepdim=True)
            test_features = test_features / \
                test_features.norm(dim=-1, keepdim=True)

            for alpha in tqdm(alpha_list):
                for beta in beta_list:
                    p = P(val_features, z_img_proto, z_text_proto, alpha, beta)
                    val_acc = (p.max(1)[1] == val_labels).float().mean()
                    val_acc_list.append([alpha, beta, val_acc.item()])
                    p = P(test_features, z_img_proto,
                          z_text_proto, alpha, beta)
                    test_acc = (p.max(1)[1] == test_labels).float().mean()
                    test_acc_list.append([alpha, beta, test_acc.item()])
                    p = P(train_features, z_img_proto,
                          z_text_proto, alpha, beta)
                    train_acc = (p.max(1)[1] == train_labels).float().mean()
                    train_acc_list.append([alpha, beta, train_acc.item()])

            val_acc_list = np.array(val_acc_list)
            test_acc_list = np.array(test_acc_list)
            train_acc_list = np.array(train_acc_list)

            save(val_acc_list, val_path, 'hp based on val set')
            save(test_acc_list, test_path, 'hp based on test set')
            save(train_acc_list, train_path, 'hp based on test set')

    _, best_alpha, best_beta, _, _ = \
        plot_zero_shot_alpha_beta(val_acc_list[:, 0], val_acc_list[:, 1], val_acc_list[:, 2],
                                  test_acc_list[:, 2], train_acc_list[:, 2], cfg, writer, 0)

    # use the cfg alpha and beta for training
    best_alpha = cfg['alpha']
    best_beta = cfg['beta']

    if not cfg['only_test']:
        input('Please enter to start training.')

        for epoch in range(cfg['train_epoch']):
            # Train
            visual_embeddings.train()
            textual_embeddings.train()
            adapter.train()

            correct_samples, all_samples = 0, 0
            loss_list = []
            print('Train Epoch: {:} / {:}'.format(epoch, cfg['train_epoch']))

            for i, (images, target) in tqdm(enumerate(train_loader_F)):
                images, zq_labels = images.cuda(), target.cuda()
                with torch.no_grad():
                    zq_imgs = clip_model.encode_image(images)

                # create support set from visual_embeddings, all classes
                zs_imgs = visual_embeddings.weight.view(-1, K, ndim)
                zs_imgs = zs_imgs / zs_imgs.norm(dim=-1, keepdim=True)
                z_img_proto = zs_imgs.mean(dim=1).float()
                z_img_proto = z_img_proto / \
                    z_img_proto.norm(dim=-1, keepdim=True)
                zq_imgs = adapter(zq_imgs).float()  # adapter

                # use all classes
                zs_text = textual_embeddings.weight

                # normalization
                zq_imgs = zq_imgs / zq_imgs.norm(dim=-1, keepdim=True)
                zs_text = zs_text / zs_text.norm(dim=-1, keepdim=True)

                # compute class prototypes
                z_text_proto = zs_text.float()

                p = P(zq_imgs, z_img_proto, z_text_proto, best_alpha, best_beta)

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

            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            # current_lr = cfg['lr']

            train_acc = correct_samples / all_samples
            train_loss = sum(loss_list)/len(loss_list)
            print('LR: {:.6f}, Acc: {:.4f}% ({:}/{:}), Loss: {:.4f}'.format(
                current_lr, train_acc*100, correct_samples, all_samples, train_loss))

            # test validation set
            with torch.no_grad():
                # zs_imgs = adapter(visual_embeddings.weight).view(-1, K, ndim)
                zs_imgs = visual_embeddings.weight.view(-1, K, ndim)
                zs_imgs = zs_imgs / zs_imgs.norm(dim=-1, keepdim=True)
                z_img_proto = zs_imgs.mean(dim=1)
                z_img_proto = z_img_proto / \
                    z_img_proto.norm(dim=-1, keepdim=True)

                zs_text = textual_embeddings(
                    torch.arange(N, requires_grad=False).cuda())
                z_text_proto = zs_text / zs_text.norm(dim=-1, keepdim=True)

                val_features_adapt = adapter(val_features)
                val_features_adapt = val_features_adapt / \
                    val_features_adapt.norm(dim=-1, keepdim=True)

                p = P(val_features_adapt, z_img_proto,
                      z_text_proto, best_alpha, best_beta)

                pred_p, y_hat = p.max(dim=1)
                matches = (y_hat == val_labels).float().sum()
                neg_log_loss_val = -torch.log(pred_p).mean()

                val_acc = (p.max(1)[1] == val_labels).float().mean()

                print("**** Proto-CLIP's val accuracy: {:.2f}% | loss: {:.2f}***\n".format(
                    val_acc*100, neg_log_loss_val))

                model_dir_root = get_model_dir_root(cfg)

                model_dir = f"{model_dir_root}/best-alpha-beta/{best_alpha}-{best_beta}"
                model_prefix = f"best_lr_{cfg['lr']}_aug_{cfg['augment_epoch']}_epochs_{cfg['train_epoch']}"

                os.makedirs(model_dir, exist_ok=True)

                best_model_path_v = os.path.join(
                    model_dir, f"{model_prefix}_v.pt")
                best_model_path_t = os.path.join(
                    model_dir, f"{model_prefix}_t.pt")
                best_model_path_a = os.path.join(
                    model_dir, f"{model_prefix}_a.pt")

                if val_acc >= best_acc:
                    best_acc = val_acc
                    best_epoch = epoch
                    torch.save(visual_embeddings.weight, best_model_path_v)
                    torch.save(textual_embeddings.weight, best_model_path_t)
                    torch.save(adapter.state_dict(), best_model_path_a)

                # Log to tensorboard
                writer.add_scalar('Loss/val', neg_log_loss_val, epoch)
                writer.add_scalar('Accuracy/val', val_acc, epoch)
                print(
                    f"Best model: best_val_acc = {best_acc*100: .2f}, best_val_epoch = {best_epoch}")

            # Log to tensorboard
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('HP/lr', current_lr, epoch)

        print(
            f"Best model: best_val_acc = {best_acc*100: .2f}, best_val_epoch = {best_epoch}")

    with torch.no_grad():
        print("Testing...")
        model_dir = f"{model_dir_root}/best-alpha-beta/{best_alpha}-{best_beta}"
        model_prefix = f"best_lr_{cfg['lr']}_aug_{cfg['augment_epoch']}_epochs_{cfg['train_epoch']}"

        best_model_path_v = os.path.join(model_dir, f"{model_prefix}_v.pt")
        best_model_path_t = os.path.join(model_dir, f"{model_prefix}_t.pt")
        best_model_path_a = os.path.join(model_dir, f"{model_prefix}_a.pt")

        try:
            embeddings_v = torch.load(best_model_path_v)
            embeddings_t = torch.load(best_model_path_t)
            adapter.load_state_dict(torch.load(best_model_path_a))
        except:
            raise FileNotFoundError(
                f"File does not exist: {best_model_path_v} and {best_model_path_t}")

        zs_imgs = embeddings_v.view(-1, K, ndim)
        zs_imgs = zs_imgs / zs_imgs.norm(dim=-1, keepdim=True)
        z_img_proto = zs_imgs.mean(dim=1)
        z_img_proto = z_img_proto / z_img_proto.norm(dim=-1, keepdim=True)

        zs_text = embeddings_t
        z_text_proto = zs_text / zs_text.norm(dim=-1, keepdim=True)

        test_features = adapter(test_features)
        test_features = test_features / \
            test_features.norm(dim=-1, keepdim=True)

        train_features = adapter(visual_memory_keys.t())
        train_features = train_features / \
            train_features.norm(dim=-1, keepdim=True)

        # hp search
        val_features_adapt = adapter(val_features)
        val_acc_list = []
        test_acc_list = []
        train_acc_list = []
        for alpha in tqdm(alpha_list):
            for beta in beta_list:
                p = P(val_features_adapt, z_img_proto,
                      z_text_proto, alpha, beta)
                val_acc = (p.max(1)[1] == val_labels).float().mean()
                val_acc_list.append([alpha, beta, val_acc.item()])
                p = P(test_features, z_img_proto, z_text_proto, alpha, beta)
                test_acc = (p.max(1)[1] == test_labels).float().mean()
                test_acc_list.append([alpha, beta, test_acc.item()])
                p = P(train_features, z_img_proto, z_text_proto, alpha, beta)
                train_acc = (p.max(1)[1] == train_labels).float().mean()
                train_acc_list.append([alpha, beta, train_acc.item()])

        val_acc_list = np.array(val_acc_list)
        test_acc_list = np.array(test_acc_list)
        train_acc_list = np.array(train_acc_list)

        p = P(test_features, z_img_proto, z_text_proto, best_alpha, best_beta)

        test_acc = (p.max(1)[1] == test_labels).float().mean()

        print(
            "**** Fixed-alp-beta: Proto-CLIP's test accuracy: {:.2f}% ****\n".format(test_acc*100))
        print('fixed_best_alpha', best_alpha, 'fixed_best_beta', best_beta)

        _, best_alpha, best_beta, _, _ = \
            plot_zero_shot_alpha_beta(val_acc_list[:, 0], val_acc_list[:, 1], val_acc_list[:, 2],
                                      test_acc_list[:, 2], train_acc_list[:, 2], cfg, writer, 0)

        p = P(test_features, z_img_proto, z_text_proto, best_alpha, best_beta)

        test_acc = (p.max(1)[1] == test_labels).float().mean()

        print(
            "**** HP-search: Proto-CLIP's test accuracy: {:.2f}% ****\n".format(test_acc*100))
        print('hp_search_best_alpha', best_alpha,
              'hp_search_best_beta', best_beta)

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
        train_loader_F = torch.utils.data.DataLoader(
            dataset.train, batch_size=256, num_workers=n_workers, shuffle=True, worker_init_fn=seed_worker, generator=g)
        val_loader = torch.utils.data.DataLoader(
            dataset.test, batch_size=val_bs, num_workers=n_workers, shuffle=False)
        test_loader = torch.utils.data.DataLoader(
            dataset.test, batch_size=test_bs, num_workers=n_workers, shuffle=False)
    else:
        dataset = build_dataset(cfg['dataset'], cfg['root_path'], cfg['shots'])
        train_tranform = get_random_train_tfm()
        train_loader_cache = build_data_loader(data_source=dataset.train_x, batch_size=train_bs,
                                               tfm=train_tranform, is_train=True, shuffle=False, worker_init_fn=seed_worker, generator=g)
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
                   test_features, test_labels, textual_memory_bank, clip_model, text_prompts)


if __name__ == '__main__':
    main()
