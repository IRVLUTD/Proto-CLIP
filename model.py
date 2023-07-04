from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from utils import *


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
