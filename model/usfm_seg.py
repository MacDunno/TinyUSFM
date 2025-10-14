# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
import torchvision.models as models

logger = logging.getLogger(__name__)


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        window_size=None,
        attn_head_dim=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (
                2 * window_size[1] - 1
            ) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads)
            )

            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = (
                coords_flatten[:, :, None] - coords_flatten[:, None, :]
            )
            relative_coords = relative_coords.permute(
                1, 2, 0
            ).contiguous()
            relative_coords[:, :, 0] += window_size[0] - 1
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = torch.zeros(
                size=(window_size[0] * window_size[1] + 1,) * 2,
                dtype=relative_coords.dtype,
            )
            relative_position_index[1:, 1:] = relative_coords.sum(-1)
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index", relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (
                    self.q_bias,
                    torch.zeros_like(self.v_bias, requires_grad=False),
                    self.v_bias,
                )
            )
        qkv = torch.nn.functional.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        if self.relative_position_bias_table is not None:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1,
                -1,
            )
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1
            ).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        init_values=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        window_size=None,
        attn_head_dim=None,
    ):
        super().__init__()
        from timm.models.layers import DropPath
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            window_size=window_size,
            attn_head_dim=attn_head_dim,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if init_values is not None:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones(dim), requires_grad=True
            )
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones(dim), requires_grad=True
            )
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, rel_pos_bias=None):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(
                self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias)
            )
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class RelativePositionBias(nn.Module):
    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (
            2 * window_size[1] - 1
        ) + 3
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads)
        )

        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = torch.zeros(
            size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype
        )
        relative_position_index[1:, 1:] = relative_coords.sum(-1)
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self):
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1] + 1,
            self.window_size[0] * self.window_size[1] + 1,
            -1,
        )
        return relative_position_bias.permute(2, 0, 1).contiguous()


class HybridResNetV2(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet50(pretrained=pretrained)

        layers = list(resnet.children())[:-2]  
        
        self.conv1 = layers[0]  # 7x7 conv
        self.bn1 = layers[1]
        self.relu = layers[2]  
        self.maxpool = layers[3]
        
        self.layer1 = layers[4]  # stage1: 64->256, stride=1
        self.layer2 = layers[5]  # stage2: 256->512, stride=2
        self.layer3 = layers[6]  # stage3: 512->1024, stride=2
        
        self.width = 64 
        
        self.channel_adapters = nn.ModuleList([
            nn.Conv2d(256, 64, 1),     # layer1: 256->64 (for 56x56)
            nn.Conv2d(512, 256, 1),    # layer2: 512->256 (for 28x28)
        ])
        
        self.dummy_16ch_adapter = nn.Conv2d(64, 16, 1)

    def forward(self, x):
        # Stage 0: stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # /4, 64 channels
        
        # Stage 1
        x1 = self.layer1(x)  # /4, 256 channels, 56x56
        
        # Stage 2  
        x2 = self.layer2(x1)  # /8, 512 channels, 28x28
        
        # Stage 3 
        x3 = self.layer3(x2)  # /16, 1024 channels, 14x14
        
        skip_28x28 = self.channel_adapters[1](x2)  # 512->256, 28x28
        skip_56x56 = self.channel_adapters[0](x1)  # 256->64, 56x56
        
        skip_112x112 = nn.functional.interpolate(skip_56x56, scale_factor=2, mode='bilinear', align_corners=False)
        skip_112x112 = self.dummy_16ch_adapter(skip_112x112)  # 64->16
        
        adapted_features = [skip_28x28, skip_56x56, skip_112x112, None]
        
        return x3, adapted_features


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings with ResNet hybrid."""
    def __init__(self, img_size, in_channels=3, pretrained=True):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)
        
        resnet_feature_size = (img_size[0] // 16, img_size[1] // 16) 
        
        patch_size = (1, 1)
        n_patches = resnet_feature_size[0] * resnet_feature_size[1]
        
        self.hybrid = True

        if self.hybrid:
            self.hybrid_model = HybridResNetV2(pretrained=pretrained)
            in_channels = 1024
        
        hidden_size = 768
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, hidden_size))

        dropout_rate = 0.1
        self.dropout = Dropout(dropout_rate)

    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


class Transformer(nn.Module):
    def __init__(self, img_size, vis=False, pretrained=True):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(img_size=img_size, pretrained=pretrained)
        
        hidden_size = 768
        from timm.models.layers import DropPath
        
        num_heads = 12
        num_layers = 12
        mlp_ratio = 4.0
        dropout_rate = 0.1
        drop_path_rate = 0.0
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.blocks = nn.ModuleList([
            Block(
                dim=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                qk_scale=None,
                drop=dropout_rate,
                attn_drop=0.0,
                drop_path=dpr[i],
                norm_layer=nn.LayerNorm,
                init_values=None,
                window_size=None,
                attn_head_dim=None,
            )
            for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        
        x = embedding_output
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=None)
        
        encoded = self.norm(x)
        
        return encoded, None, features 


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        if out_channels == 2:
            out_channels = 1
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_size = 768
        head_channels = 512
        self.conv_more = Conv2dReLU(
            hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        
        decoder_channels = (256, 128, 64, 16)
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        skip_channels = [256, 64, 16, 0]
        
        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)  # 14x14x768 -> 14x14x512
        
        for i, decoder_block in enumerate(self.blocks):
            skip = None
            if features is not None and i < len(features) and features[i] is not None:
                skip = features[i]
            x = decoder_block(x, skip=skip)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, num_classes=2, zero_head=False, vis=False, pretrained=True):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = 'seg'
        self.transformer = Transformer(img_size, vis, pretrained=pretrained)
        self.decoder = DecoderCup()
        
        # R50+B16配置
        decoder_channels = (256, 128, 64, 16)
        n_classes = num_classes
        
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=n_classes,
            kernel_size=3,
        )

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits