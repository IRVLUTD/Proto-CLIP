from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from utils import *
import loralib as lora

class Adapter_LoRA(nn.Module):
    def __init__(self, c_in, reduction=4, dtype=None):
        super(Adapter_LoRA, self).__init__()
        self.fc = nn.Sequential(
            lora.Linear(c_in, c_in, bias=False, dtype=dtype, r=reduction),
            # nn.LayerNorm(c_in // reduction, dtype=dtype),
            # lora.Linear(c_in // reduction, c_in, bias=False, dtype=dtype),
            # nn.LayerNorm(c_in, dtype=dtype),
        )
        self.ratio = nn.Parameter(torch.tensor(0.5))   # Initialize ratio as 0.5

    def forward(self, image_features):
        x = self.fc(image_features)
        image_features = self.ratio * x + (1 - self.ratio) * image_features
        return image_features


class AdapterLoRAConv(nn.Module):
    def __init__(self, ndim):
        super(AdapterLoRAConv, self).__init__()
        self.ndim = ndim
        # Define the convolutional layers
        self.conv1 = lora.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1)
        # self.conv2 = lora.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = lora.Conv2d(in_channels=128, out_channels=ndim, kernel_size=3, stride=1, padding=1)
        
        # Define pooling layers (for downsampling)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, images):
        B = images.shape[0]
        output_tensor = self.conv1(images)
        output_tensor = self.pool(output_tensor)
        # output_tensor = self.conv2(output_tensor)
        # output_tensor = self.pool(output_tensor)
        output_tensor = self.conv3(output_tensor)
        output_tensor = self.global_pool(output_tensor)

        # Check the shape after global average pooling
        print("Output tensor shape after global avg pooling:", output_tensor.shape)

        # Reshape the output tensor to have shape [B, ndim]
        output_tensor = output_tensor.view(B, self.ndim)
        return output_tensor


class ProtoCLIP(nn.Module):
    def __init__(self, clip_model, visual_memory_keys, textual_memory_bank, N, K, ndim, dtype):
        super().__init__()
        self.N, self.K, self.ndim = N, K, ndim
        self.visual_memory_keys = visual_memory_keys
        self.textual_memory_bank = textual_memory_bank
        self.visual_embeddings = lora.Embedding(num_embeddings=N*K, embedding_dim=ndim).cuda().to(dtype)
        self.visual_embeddings.weight = nn.Parameter(visual_memory_keys.t().clone())
        self.adapter = Adapter(ndim, 'fc', dtype=torch.half).cuda()
        # self.adapter = Adapter_LoRA(ndim, dtype=torch.half).cuda()
        # self.adapter = AdapterLoRAConv(ndim).cuda()
        # self.alpha = nn.Parameter(torch.tensor(0.5))  # Initialize alpha as 0.5
        # self.beta = nn.Parameter(torch.tensor(1.0))   # Initialize beta as 0.5
        self.beta = 17 #torch.tensor(1.0*ndim).sqrt()
        self.textual_embeddings = lora.Embedding(num_embeddings=N, embedding_dim=ndim).cuda().to(dtype)
        self.textual_embeddings.weight = nn.Parameter(textual_memory_bank.t().clone())
        self.clip_model = clip_model
        self.clip_model.eval()

    def forward(self, zq_imgs):
        zs_imgs = self.visual_embeddings.weight.view(-1, self.K, self.ndim)
        zs_imgs = zs_imgs / zs_imgs.norm(dim=-1, keepdim=True)
        z_img_proto = zs_imgs.mean(dim=1).float()
        z_img_proto = z_img_proto / \
            z_img_proto.norm(dim=-1, keepdim=True)

        # print(self.clip_model.encode_image(zq_imgs).shape, self.adapter(zq_imgs).shape)
        # zq_imgs = self.clip_model.encode_image(zq_imgs) + self.adapter(zq_imgs).float()  # adapter

        zq_imgs = self.adapter(zq_imgs)

        # use all classes
        zs_text = self.textual_embeddings.weight

        # normalization
        zq_imgs = zq_imgs / zq_imgs.norm(dim=-1, keepdim=True)
        zs_text = zs_text / zs_text.norm(dim=-1, keepdim=True)

        # compute class prototypes
        z_text_proto = zs_text.float()

        # compute pairwise euclidean distances(query, prototypes)
        xq_img_proto_dists = torch.cdist(
            zq_imgs.float(), z_img_proto.float(), p=2).pow(2)
        xq_text_proto_dists = torch.cdist(
            zq_imgs.float(), z_text_proto.float(), p=2).pow(2)

        # P(y=k|query_image,support_images)
        p_i = F.softmax(self.beta*(-xq_img_proto_dists), dim=1)

        #  P(y=k|query_image,support_text)
        p_t = F.softmax(self.beta*(-xq_text_proto_dists), dim=1)

        # total probability = alpha * p_image + (1-alpha) - p_text
        p = F.softmax(p_i + p_t, dim=-1)

        return p, z_img_proto, z_text_proto


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


class Adapter_FC(nn.Module):
    def __init__(self, c_in, reduction=4, dtype=None):
        super(Adapter_FC, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False, dtype=dtype),
            nn.LayerNorm(c_in // reduction, dtype=dtype),
            nn.Linear(c_in // reduction, c_in, bias=False, dtype=dtype),
            nn.LayerNorm(c_in, dtype=dtype),
        )

    def forward(self, image_features):
        x = self.fc(image_features)
        ratio = 0.2  # to prevent overfitting
        image_features = ratio * x + (1 - ratio) * image_features
        return image_features

    
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
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

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
