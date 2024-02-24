from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init
import math
from utils import *
import loralib as lora
from metrics import ArcMarginProduct


class Adapter_FC(nn.Module):
    def __init__(self, c_in, reduction=1, dropout_prob=0.1,dtype=None):
        super(Adapter_FC, self).__init__()
        
        # Initialize linear layers with Xavier initialization
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
        
        self.fc1 = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False, dtype=dtype), # 44.48, fewsol ViT-B/32
            # nn.ReLU(),
            # nn.LayerNorm(c_in // reduction, dtype=dtype),
            # # nn.Dropout(dropout_prob),  # Add dropout layer
            # nn.Linear(c_in // reduction, c_in, bias=False, dtype=dtype), # 44.26, fewsol ViT-B/32 
            # nn.ReLU(),
            # nn.LayerNorm(c_in, dtype=dtype),
            # # nn.Dropout(dropout_prob),  # Add dropout layer
            # nn.Linear(c_in, c_in // reduction, bias=False, dtype=dtype),
            # nn.ReLU(),
            # nn.LayerNorm(c_in // reduction, dtype=dtype),
            # nn.Linear(c_in, c_in // reduction, bias=False, dtype=dtype),
            # nn.ReLU(),
            # nn.LayerNorm(c_in // reduction, dtype=dtype),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False, dtype=dtype), # 44.48, fewsol ViT-B/32
            # nn.ReLU(),
            # nn.LayerNorm(c_in // reduction, dtype=dtype),
            # # nn.Dropout(dropout_prob),  # Add dropout layer
            # nn.Linear(c_in // reduction, c_in, bias=False, dtype=dtype), # 44.26, fewsol ViT-B/32 
        )

        self.conv3x = Adapter(c_in, 'conv-3x')

        # Apply Xavier initialization to the linear layers
        self.apply(init_weights)

    def forward(self, image_features):
        x1 = self.fc1(image_features)#.exp()
        x2 = self.fc2(image_features)#.exp()
        x1 = x1 + image_features
        x2 = x2 + image_features
        x21 = self.fc2(x1) + x1
        x12 = self.fc1(x2) + x2
        return x21, x12

class ProtoCLIP(nn.Module):
    def __init__(self, clip_model, visual_memory_keys, visual_memory_values, textual_memory_bank, N, K, ndim, dtype):
        super().__init__()
        self.N, self.K, self.ndim = N, K, ndim
        self.visual_memory_keys = visual_memory_keys
        self.textual_memory_bank = textual_memory_bank
        self.frozen_textual_memory_bank = textual_memory_bank
        self.visual_memory_values = visual_memory_values
        self.zs_labels_one_hot = self.visual_memory_values.view(-1, self.K, self.N).float().mean(dim=1)
        self.visual_embeddings = nn.Embedding(num_embeddings=N*K, embedding_dim=ndim).cuda().to(dtype)
        self.visual_embeddings.weight = nn.Parameter(visual_memory_keys.t().clone())
        self.adapter_conv_3x = Adapter(ndim, 'conv-3x').cuda()
        self.adapter_conv_2x = Adapter(ndim, 'conv-2x').cuda()
        self.adapter_fc = Adapter_FC(ndim).cuda()
        self.textual_embeddings = nn.Embedding(num_embeddings=N, embedding_dim=ndim).cuda().to(dtype)
        self.textual_embeddings.weight = nn.Parameter(textual_memory_bank.t().clone())
        self.clip_model = clip_model
        self.freeze_models()


    def freeze_models(self):
        for param in self.clip_model.parameters():
            param.requires_grad = False
        # for param in self.frozen_textual_memory_bank.parameters():
        #     param.requires_grad = False
        # for param in self.visual_embeddings.parameters():
        #     param.requires_grad = False


    def get_memory_banks(self):
        zs_imgs = self.visual_embeddings.weight.view(-1, self.K, self.ndim).float()
        zs_text = self.textual_embeddings.weight.float()
        return zs_imgs, zs_text


    def compute_prototypes(self):
        # This is a simple average over support embeddings
        zs_imgs, zs_text = self.get_memory_banks()
        zs_imgs = zs_imgs / zs_imgs.norm(dim=-1, keepdim=True)
        z_img_proto = zs_imgs.mean(dim=1)
        z_img_proto = z_img_proto / z_img_proto.norm(dim=-1, keepdim=True)

        # use all classes, normalization, compute class prototypes
        z_text_proto = zs_text / zs_text.norm(dim=-1, keepdim=True)

        return z_img_proto, z_text_proto


    def get_memory_one_hot_labels(self):
        return self.zs_labels_one_hot


    def forward(self, zq_imgs, zq_labels, alpha=0.9, beta=1, do_zero_shot=False):

        # print(self.clip_model.encode_image(zq_imgs).shape, self.adapter(zq_imgs).shape)
        # zq_imgs = self.clip_model.encode_image(zq_imgs) + self.adapter(zq_imgs).float()  # adapter

        z_img_proto, z_text_proto =  self.compute_prototypes()
        # z_img_proto, z_text_proto =  self.compute_attention_prototypes(zq_imgs)
        zs_one_hot_labels = self.get_memory_one_hot_labels()
        # print(zs_one_hot_labels)

        # query adapter 3
        zq_imgs = zq_imgs.float()
        if not do_zero_shot:
            zq_imgs_i, zq_imgs_t = self.adapter_fc(zq_imgs)
            pass
        zq_imgs_i = zq_imgs_i / zq_imgs_i.norm(dim=-1, keepdim=True)
        zq_imgs_t = zq_imgs_t / zq_imgs_t.norm(dim=-1, keepdim=True)
        
        # compute pairwise euclidean distances(query, prototypes)
        pow = 2
        xq_img_proto_dists = torch.cdist(
            zq_imgs_i.float(), z_img_proto.float(), p=pow).pow(pow)
        xq_text_proto_dists = torch.cdist(
            zq_imgs_t.float(), z_text_proto.float(), p=pow).pow(pow)

        self.beta = 1 #torch.tensor(1.0*self.ndim).sqrt() 

        # logits_iq = (self.beta * -xq_img_proto_dists).exp() @ zs_one_hot_labels + xq_text_proto_dists
        # logits_tq = xq_img_proto_dists + (self.beta * -xq_text_proto_dists).exp() @ zs_one_hot_labels

        logits_iq = (self.beta * -xq_img_proto_dists).exp()
        logits_tq = (self.beta * -xq_text_proto_dists).exp()

        clip_logits = zq_imgs @ self.frozen_textual_memory_bank.float()

        # logits = (logits_iq + logits_tq + clip_logits)#.exp()
        # logits = (logits_iq * logits_tq + clip_logits).exp() #  works
        logits = ((logits_iq * logits_tq) + clip_logits)#.exp() #  without exp by far the best

        # logits = self.fc(logits)

        p = logits.softmax(-1)

        # print(xq_img_proto_dists, xq_text_proto_dists.shape)


        # P(y=k|query_image,support_images)
        # p_i = F.softmax(self.beta*(-xq_img_proto_dists), dim=1)

        # # # #  P(y=k|query_image,support_text)
        # p_t = F.softmax(self.beta*(-xq_text_proto_dists), dim=1)

        # # # # total probability = alpha * p_image + (1-alpha) - p_text
        # p = ( p_i + p_t).softmax(dim=-1)

        # total probability = self.alpha * p_image + (1-self.alpha) - p_text
        # p = self.alpha * p_i + (1-self.alpha) * p_t
        # p = (self.alpha * p_i + (1-self.alpha) - p_t).softmax(-1)
        # p = ((p_i * p_t).exp()).softmax(dim=-1)
        # p = ((p_i * p_t).exp() @ zs_one_hot_labels).softmax(dim=-1)
        # p = (p_i * p_t)

        # # label smoothing: https://www.sciencedirect.com/science/article/pii/S0893608022003689
        # eps = 0.5
        # p = ((1-eps) * p + eps / self.N).exp().softmax(dim=-1) # doesn;t help much

        return p, z_img_proto, z_text_proto


    # def forward(self, zq_imgs, alpha=0.9, beta=17, do_zero_shot=False):

    #     # print(self.clip_model.encode_image(zq_imgs).shape, self.adapter(zq_imgs).shape)
    #     # zq_imgs = self.clip_model.encode_image(zq_imgs) + self.adapter(zq_imgs).float()  # adapter

    #     self.set_alpha_beta(alpha, beta)

    #     z_img_proto, z_text_proto =  self.compute_prototypes()
    #     # z_img_proto, z_text_proto =  self.compute_attention_prototypes(zq_imgs)
    #     zs_one_hot_labels = self.get_memory_one_hot_labels()
    #     # print(zs_one_hot_labels)

    #     # query adapter
    #     # zq_imgs = self.adapter(zq_imgs) #+ zq_imgs
    #     # zq_imgs = zq + attention(zq.half(), z_img_proto.half(), z_text_proto.half(), self.ndim).float() + zq_imgs
    #     # zq_imgs = self.adapter_fc(self.ada1pter_conv_3x(self.adapter_conv_2x(zq_imgs) + zq_imgs)+ zq_imgs) + zq_imgs
    #     # zq_imgs = self.adapter_conv_3x(zq_imgs) + attention(zq_imgs.half(), z_img_proto.half(), z_text_proto.half(), self.ndim).float() + zq_imgs

    #     # query adapter 2
    #     # zq_imgs = zq_imgs.float()


    #     # zq_imgs = self.adapter_conv_3x(zq_imgs) + qkv + zq_imgs
    #     # zq_imgs = zq_imgs / zq_imgs.norm(dim=-1, keepdim=True)
        

    #     # query adapter 3
    #     zq_imgs = zq_imgs.float()
    #     if not do_zero_shot:
    #         # adapter-1: MHSA
    #         # zq_imgs = self.attn(zq_imgs, zq_imgs, zq_imgs) + zq_imgs # didn't work w/wo skip connection

    #         # adapter-2: SHSA
    #         # q = self.q_proj(zq_imgs)
    #         # k = self.k_proj(z_img_proto)
    #         # v = self.v_proj(z_text_proto)
    #         # qk = (q @ k.t() ).float().softmax(dim=-1)
    #         # qkv = qk @ v #+ zq_imgs
    #         # zq_imgs = qkv / qkv.norm(dim=-1, keepdim=True) + zq_imgs # didn't work w/wo skip connection; worse than adapter-1
            
    #         # adapter-3: adapter_conv3x
    #         zq_imgs = self.adapter_conv_3x(zq_imgs)

    #         # adapter-4: adapter_conv3x
    #         # zq_imgs = self.adapter_conv_2x(zq_imgs) # lower performance than conv3x adapter-3

    #         # adapter-5: adapter_fc
    #         # zq_imgs = self.adapter_fc(zq_imgs) # not helping

    #     zq_imgs = zq_imgs / zq_imgs.norm(dim=-1, keepdim=True)
        
    #     # zq_imgs = self.adapter_conv_3x(zq_imgs) + attention(zq_imgs.half(), z_img_proto.half(), z_img_proto.half(), self.ndim).float() + zq_imgs


    #     # z_img_txt_proto_concatenated = torch.cat((z_img_proto, z_text_proto), dim=1)

    #     # # Repeat tensor B along a new dimension to match the first dimension of tensor A
    #     # zq_imgs_expanded = zq_imgs.unsqueeze(1).expand(-1, z_img_txt_proto_concatenated.size(0), -1)

    #     # q_img_txt = torch.cat((z_img_txt_proto_concatenated.unsqueeze(0).expand(zq_imgs.size(0), -1, -1), zq_imgs_expanded), dim=2)

    #     # logits = self.relnet(q_img_txt.float())

    #     # p = logits.softmax(dim=-1)

    #     # # attn = (Q.K^T)/sqrt(D)
    #     bs, D = zq_imgs.shape
    #     # # # # qkv_attn = ((zq_imgs @ z_img_proto.T) / torch.sqrt(torch.tensor(D))).softmax(dim=-1) @ z_text_proto
    #     # # # # # qk = ((zq_imgs @ z_img_proto.T) / (torch.tensor(D)).sqrt()).softmax(dim=-1)
    #     # qk = ((zq_imgs @ z_img_proto.T)).softmax(dim=-1)
    #     # zq_imgs = (qk @ z_text_proto) + zq_imgs
    #     # zq_imgs = zq_imgs / zq_imgs.norm(dim=-1, keepdim=True)

    #     # q = self.q_proj(zq_imgs)
    #     # k = self.k_proj(z_img_proto)
    #     # v = self.v_proj(z_text_proto)

    #     # qk = (q @ k.t() ).softmax(dim=-1)
    #     # zq_imgs = qk @ v #+ zq_imgs

    #     # zq_imgs = attention(zq_imgs, z_img_proto, z_text_proto, self.ndim) + zq_imgs

    #     # zq_imgs = zq_imgs / zq_imgs.norm(dim=-1, keepdim=True)
        
    #     # compute pairwise euclidean distances(query, prototypes)
    #     pow = 2
    #     xq_img_proto_dists = torch.cdist(
    #         zq_imgs.float(), z_img_proto.float(), p=pow).pow(pow)
    #     xq_text_proto_dists = torch.cdist(
    #         zq_imgs.float(), z_text_proto.float(), p=pow).pow(pow)

        
        
    #     # print(xq_img_proto_dists.shape, xq_text_proto_dists.shape)

    #     # 34.65% # make sure to test the last ckpt as well along with best ckpt as it would give better performance w.r.t. best ckpt
    #     self.beta = torch.tensor(1.0*self.ndim).sqrt() 

    #     # P(y=k|query_image,support_images)
    #     # p_i = F.softmax(self.beta * (-xq_img_proto_dists), dim=-1)
    #     p_i = F.softmax(self.beta * (-xq_img_proto_dists), dim=-1) # exp() operation doesn't help here

    #     #  P(y=k|query_image,support_text)
    #     # p_t = F.softmax(self.beta * (-xq_text_proto_dists), dim=-1)
    #     p_t = F.softmax(self.beta * (-xq_text_proto_dists), dim=-1) # exp() operation doesn't help here


    #     # total probability = self.alpha * p_image + (1-self.alpha) - p_text
    #     # p = self.alpha * p_i + (1-self.alpha) * p_t
    #     # p = (self.alpha * p_i + (1-self.alpha) - p_t).softmax(-1)
    #     p = ((p_i * p_t).exp()).softmax(dim=-1)
    #     # p = ((p_i * p_t).exp() @ zs_one_hot_labels).softmax(dim=-1)
    #     # p = (p_i * p_t)

    #     # # label smoothing: https://www.sciencedirect.com/science/article/pii/S0893608022003689
    #     # eps = 0.5
    #     # p = ((1-eps) * p + eps / self.N).exp().softmax(dim=-1) # doesn;t help much

    #     return p, z_img_proto, z_text_proto


    # def forward(self, zq_imgs):
    #     zs_imgs = self.visual_embeddings.weight.view(-1, self.K, self.ndim)
    #     zs_imgs = zs_imgs / zs_imgs.norm(dim=-1, keepdim=True)
    #     z_img_proto = zs_imgs.mean(dim=1).float()
    #     z_img_proto = z_img_proto / \
    #         z_img_proto.norm(dim=-1, keepdim=True)

    #     # print(self.clip_model.encode_image(zq_imgs).shape, self.adapter(zq_imgs).shape)
    #     # zq_imgs = self.clip_model.encode_image(zq_imgs) + self.adapter(zq_imgs).float()  # adapter

    #     zq_imgs = self.adapter(zq_imgs)

    #     # use all classes
    #     zs_text = self.textual_embeddings.weight

    #     # normalization
    #     zq_imgs = zq_imgs / zq_imgs.norm(dim=-1, keepdim=True)
    #     zs_text = zs_text / zs_text.norm(dim=-1, keepdim=True)

    #     # compute class prototypes
    #     z_text_proto = zs_text.float()

    #     # compute pairwise euclidean distances(query, prototypes)
    #     xq_img_proto_dists = torch.cdist(
    #         zq_imgs.float(), z_img_proto.float(), p=2).pow(2)
    #     xq_text_proto_dists = torch.cdist(
    #         zq_imgs.float(), z_text_proto.float(), p=2).pow(2)

    #     # P(y=k|query_image,support_images)
    #     p_i = F.softmax(self.beta*(-xq_img_proto_dists), dim=1)

    #     #  P(y=k|query_image,support_text)
    #     p_t = F.softmax(self.beta*(-xq_text_proto_dists), dim=1)

    #     # total probability = alpha * p_image + (1-alpha) - p_text
    #     p = ( p_i + p_t ).softmax(dim=-1)

    #     return p, z_img_proto, z_text_proto


class Adapter(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    def __init__(
        self,
        c_in,
        c_type,
        width=16,   # best so far 16
        dtype=None
    ):
        super().__init__()
        self.c_in = c_in
        self.c_type = c_type
        # Round up to the nearest integer
        size = int(math.ceil(math.sqrt(self.c_in)))

        norm_layer = nn.LayerNorm

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(1, width, kernel_size=1,
                               stride=1, bias=False, dtype=dtype)
        self.bn1 = norm_layer([width, size, size], dtype=dtype)

        self.conv2 = nn.Conv2d(width, width, kernel_size=3,
                               stride=1, padding=1, bias=False, dtype=dtype)
        self.bn2 = norm_layer([width, size, size], dtype=dtype)

        self.conv3 = nn.Conv2d(width, 1, kernel_size=1,
                               stride=1, bias=False, dtype=dtype)
        self.bn3 = norm_layer([1, size, size], dtype=dtype)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """    
        size = int(math.sqrt(self.c_in))
        x = x.view(-1, 1, size, size) #sqrt(768) is not a perfect integer for ViT-L/14
        """
        size = int(math.ceil(math.sqrt(self.c_in))
                   )  # Round up to the nearest integer
        pad_size = size**2 - self.c_in  # Compute the padding size
        # Pad the input tensor with zeros if necessary
        x = torch.nn.functional.pad(x, (0, pad_size))
        x = x.view(-1, 1, size, size)  # Reshape the tensor to a square shape

        identity = x

        out = self.conv1(x)
        out = self.bn1(out) 

        if self.c_type == 'conv-3x':
            out = self.conv2(out)
            out = self.bn2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity

        out = out.view(-1, 1, size*size)
        out = out[:, :, :self.c_in].view(-1, self.c_in)

        return out



    
class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model, weights, dtype=None):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, dtype=dtype)
        self.embed.weight = nn.Parameter(weights.clone())

    def forward(self, x):
        return self.embed(x)


def attention(q, k, v, d_k, mask=None, dropout=None):

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model, dtype=dtype)
        self.v_linear = nn.Linear(d_model, d_model, dtype=dtype)
        self.k_linear = nn.Linear(d_model, d_model, dtype=dtype)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model, dtype=dtype)

    def forward(self, q, k, v, mask=None):

        bs = q.size(0)

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, self.d_model)

        output = self.out(concat)

        return output


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, heads, weights, dtype=None):
        super().__init__()
        self.embed = Embedder(vocab_size, d_model, weights, dtype=dtype)
        self.attn = MultiHeadAttention(heads, d_model, dtype=dtype)

    def forward(self, src):
        x1 = self.embed(src)
        return self.attn(x1, x1, x1)