from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os
from PIL import Image
import clip
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torchvision.transforms.functional as TF
from info_nce import InfoNCE
from focal_loss.focal_loss import FocalLoss


def get_seed():
    """
    Returns a random number generator seed for reproducibility purposes
    """
    return 1


def dir_exists(path):
    """
    Returns boolean value depending on whether the given path exists
    """
    return os.path.exists(path)


def save(obj, filepath, msg):
    """
    Saves the input object as a pickle file
    """
    print(f"Saving {msg} to {filepath}")
    with open(filepath, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load(filepath, msg):
    """
    Loads a pickle file from disk
    """
    print(f"Loading {msg} from {filepath}")
    with open(filepath, 'rb') as handle:
        return pickle.load(handle)


def get_textual_memory_bank(cfg, classnames, template, clip_model):
    msg = "text_memory_bank"
    model_dir_root = get_model_dir_root(cfg)
    os.makedirs(model_dir_root, exist_ok=True)
    path = os.path.join(
        model_dir_root, f"text_mb_{beautify(cfg['backbone'])}_K_{cfg['shots']}.pkl")

    if dir_exists(path):
        text_prompts = classnames
        return text_prompts, load(path, msg)
    else:
        # Textual features
        text_prompts, textual_memory_bank = clip_classifier(
            classnames, template, clip_model)
        save(textual_memory_bank, path, msg)
        return text_prompts, textual_memory_bank


def InfoNCELoss(A, B):
    """
    Computes L2 and L3 losses
    """
    loss = InfoNCE()
    return loss(A, B)


def compute_loss_and_matches(p, target_inds, z_img_proto, z_text_proto, cfg):
    """
        Computes loss and accuracy for one episode
    """
    pred_p, y_hat = p.max(dim=1)
    matches = (y_hat == target_inds).float().sum()
    loss = 0
    neg_log_loss, img2txt_align_loss, txt2img_align_loss, img_inter_cluster_loss, txt_inter_cluster_loss = None, None, None, None, None

    if len(cfg['losses']) == 0 or 'L1' in cfg['losses']:

        # nloss = nn.NLLLoss()
#         # loss += nloss(torch.log(p), target_inds)

        focalLoss = FocalLoss(gamma=1)
        loss += focalLoss(p, target_inds)

    if 'L2' in cfg['losses']:
        # L2: img with all text alignment loss
        img2txt_align_loss = InfoNCELoss(z_img_proto, z_text_proto)
        loss += img2txt_align_loss

    if 'L3' in cfg['losses']:
        # L3: text with all img alignment loss
        txt2img_align_loss = InfoNCELoss(z_text_proto, z_img_proto)
        loss += txt2img_align_loss

    if 'L4' in cfg['losses']:
        img_inter_cluster_loss = InfoNCELoss(z_img_proto, z_img_proto)
        txt_inter_cluster_loss = InfoNCELoss(z_text_proto, z_text_proto)
        loss += img_inter_cluster_loss
        loss += txt_inter_cluster_loss
    return matches, loss, neg_log_loss, img2txt_align_loss, txt2img_align_loss, img_inter_cluster_loss, txt_inter_cluster_loss


def get_target_inds(info):
    """
        Forms the target ground truth class label for the episode
        using info: n_class, k_support, k_query
    """
    n_class, _, k_query = info
    # form the ground truth labels
    target_inds = torch.arange(0, n_class).view(
        n_class, 1, 1).expand(n_class, k_query, 1).long()
    target_inds = Variable(target_inds).cuda()
    return target_inds


def plot_tsne(model_dir_root, z_img_proto, z_text_proto, zq_imgs, txt_prompts, cfg, writer):
    X = torch.vstack((
        z_img_proto,
        z_text_proto,
        # zq_imgs.view(-1, zq_imgs.shape[-1])
    )).cpu().data.numpy()
    tsne_X = TSNE(n_components=2, perplexity=10).fit_transform(X)
    n_class = z_img_proto.shape[0]
    zi, zt = tsne_X[:n_class], tsne_X[n_class: 2*n_class]
    y = [i for i in range(n_class)]
    colors = np.arange(n_class)/10 + 0.05
    plt.clf()
    plt.scatter(zi[:, 0], zi[:, 1], c=colors, marker='s')
    plt.scatter(zt[:, 0], zt[:, 1], c=colors, marker='+')
    fontsize = 3
    for i in range(n_class):
        plt.annotate(
            txt_prompts[i], (zi[i, 0], zi[i, 1] + 0.2), fontsize=fontsize)
        plt.annotate(
            txt_prompts[i], (zt[i, 0], zt[i, 1] + 0.2), fontsize=fontsize)

    plot_prefix, ext = "tsne", "png"
    losses = '_'.join(cfg['losses'])
    plot_infix = get_model_dir_root(cfg).strip().replace(
        '/', '_').replace('._caches_', '').replace('_models', '')
    plot_filename = f"last_ckpt_{plot_prefix}_{plot_infix}_aug-{cfg['augment_epoch']}_alpha-{cfg['alpha']}-beta-{cfg['beta']}_{losses}_epochs_{cfg['train_epoch']}.{ext}"

    plot_img_path = os.path.join(
        model_dir_root, plot_filename)
    plt.axis('off')

    plt.savefig(plot_img_path, dpi=300)
    print(f"Saved t-SNE plot to {plot_img_path}")

    img = np.asarray(Image.open(plot_img_path))

    # Convert to a Tensor and normalize
    image_tensor = TF.to_tensor(img)

    writer.add_image(f't-SNE/{plot_filename}', image_tensor, 0)


def plot_zero_shot_alpha_beta(alpha, beta, val_accuracy, test_accuracy, train_accuracy, cfg, writer, counter):

    # Create a new figure and axis object
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the first curve
    ax.plot(alpha, beta, val_accuracy, label='Val')

    # Plot the second curve
    ax.plot(alpha, beta, test_accuracy, label='Test')

    # Set axis labels and title
    ax.set_xlabel('alpha')
    ax.set_ylabel('beta')
    ax.set_zlabel('zero-shot-accuracy')
    ax.set_title(f"Proto-CLIP | Dataset:{cfg['dataset']}")

    # Add a legend
    ax.legend()

    # Show the plot
    # plt.save()

    # Save the plot
    plot_dir = os.path.join(
        'plots', cfg['logs_dir_path'], 'alpha-beta', 'zero_shot')
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(f"{plot_dir}/zero_shot_{cfg['dataset']}.png")

    best_val_acc, best_val_acc_idx = val_accuracy.max(), val_accuracy.argmax()
    best_test_acc, best_test_acc_idx = test_accuracy.max(), test_accuracy.argmax()
    best_train_acc, best_train_acc_idx = train_accuracy.max(), train_accuracy.argmax()

    best_val_alpha, best_val_beta = alpha[best_val_acc_idx], beta[best_val_acc_idx]
    best_test_alpha, best_test_beta = alpha[best_test_acc_idx], beta[best_test_acc_idx]
    best_train_alpha, best_train_beta = alpha[best_train_acc_idx], beta[best_train_acc_idx]

    print(
        f"alpha: {best_val_alpha: .3f}, beta:{best_val_beta: .3f} | Max val-acc: {best_val_acc*100: .3f} | Max test-acc-using-val-alpha-beta: {test_accuracy[best_val_acc_idx]*100: .3f}")
    print(
        f"alpha: {best_test_alpha: .3f}, beta:{best_test_beta: .3f} | Max test-acc: {best_test_acc*100: .3f}")
    print(f"alpha: {best_train_alpha: .3f}, beta:{best_train_beta: .3f} | Max train-acc: {best_train_acc*100: .3f}")

    writer.add_scalar('Accuracy/zsval-zstestval-zstest-3F-test',
                      best_val_acc, counter+1)
    writer.add_scalar('Accuracy/zsval-zstestval-zstest-3F-test',
                      test_accuracy[best_val_acc_idx], counter+2)
    writer.add_scalar('Accuracy/zsval-zstestval-zstest-3F-test',
                      best_val_acc, counter+3)
    writer.add_scalar('HP/alpha-val-test', best_val_alpha, counter+1)
    writer.add_scalar('HP/beta-val-test', best_val_beta, counter+1)
    writer.add_scalar('HP/alpha-val-test', best_test_alpha, counter+2)
    writer.add_scalar('HP/beta-val-test', best_test_beta, counter+2)

    return best_val_acc, best_val_alpha, best_val_beta, best_test_alpha, best_test_beta


def P(zq_imgs_flat, z_img_proto, z_text_proto, alpha, beta):
    """
    Returns probability dist, p = alpha * p_i + (1-alpha) * p_t
    """
    # compute pairwise euclidean distances(query, prototypes)
    xq_img_proto_dists = torch.cdist(
        zq_imgs_flat.float(), z_img_proto.float(), p=2).pow(2)
    xq_text_proto_dists = torch.cdist(
        zq_imgs_flat.float(), z_text_proto.float(), p=2).pow(2)

    # P(y=k|query_image,support_images)
    p_i = F.softmax(beta*(-xq_img_proto_dists), dim=1)

    #  P(y=k|query_image,support_text)
    p_t = F.softmax(beta*(-xq_text_proto_dists), dim=1)

    # total probability = alpha * p_image + (1-alpha) - p_text
    p = alpha * p_i + (1-alpha) * p_t

    return p


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0,
                keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return classnames, clip_weights


def beautify(string):
    return string.strip().replace('/', '_').replace('-', '_')


def get_model_dir_root(cfg):
    return f"{cfg['cache_dir']}/models/{beautify(cfg['backbone'])}/K-{cfg['shots']}"


def build_cache_model(cfg, clip_model, train_loader_cache):
    model_dir_root = get_model_dir_root(cfg) + '/aug'
    os.makedirs(model_dir_root, exist_ok=True)

    def get_filename(cfg, type):
        return f"{model_dir_root}/visual_mb_{type}_aug_{cfg['augment_epoch']}_{cfg['shots']}_shots.pt"

    key_path = get_filename(cfg, 'keys')
    value_path = get_filename(cfg, 'values')

    if dir_exists(key_path) and dir_exists(value_path):
        cache_keys = torch.load(key_path)
        cache_values = torch.load(value_path)
    else:
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []

                print(
                    'Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.cuda()
                    image_features = clip_model.encode_image(images)
                    train_features.append(image_features)
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                cache_keys.append(
                    torch.cat(train_features, dim=0).unsqueeze(0))

        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)

        # sorting
        cache_values = torch.cat(cache_values, dim=0)
        index = torch.argsort(cache_values)
        cache_values = cache_values[index]
        cache_keys = cache_keys[:, index]
        cache_values = F.one_hot(cache_values)

        torch.save(cache_keys, key_path)
        torch.save(cache_values, value_path)

    return cache_keys, cache_values


def pre_load_features(cfg, split, clip_model, loader):
    root_dir_prefix = f"{get_model_dir_root(cfg)}/{split}"
    feature_path = f"{root_dir_prefix}_features.pt"
    label_path = f"{root_dir_prefix}_labels.pt"

    if dir_exists(feature_path) and dir_exists(label_path):
        print(f"Loading cached features and labels from {root_dir_prefix}")
        features = torch.load(feature_path)
        labels = torch.load(label_path)
    else:
        print(
            f"Creating cached (features, labels) and saving to {root_dir_prefix}")
        features, labels = [], []
        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader)):
                images, target = images.cuda(), target.cuda()
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features)
                labels.append(target)

        features, labels = torch.cat(features), torch.cat(labels)

        torch.save(features, feature_path)
        torch.save(labels, label_path)

    return features, labels
