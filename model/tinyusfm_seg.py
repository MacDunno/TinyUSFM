import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

import timm.models.vision_transformer
from timm.models.vision_transformer import DropPath, Mlp, Attention as BaseAttn


class Attention(BaseAttn):
    def __init__(self, *args, **kwargs):
        super(Attention, self).__init__(*args, **kwargs)
        self.identity = nn.Identity()

    def forward(self, x, return_latent=False):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.identity(attn)
        if return_latent:
            return
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
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
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_latent=False):
        if return_latent:
            x = self.attn(self.norm1(x), return_latent=return_latent)
            return x
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer with support for global average pooling and distillation"""
    def __init__(
        self,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        act_layer=None,
        global_pool=False,
        distilled=False,
        **kwargs
    ):
        super(VisionTransformer, self).__init__(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            act_layer=act_layer,
            **kwargs
        )
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                for i in range(depth)
            ]
        )
        self.num_heads = num_heads
        self.depth = depth
        self.global_pool = global_pool
        self.distilled = distilled
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)
            del self.norm
            if self.distilled:
                self.dist_norm = norm_layer(embed_dim)
                if self.num_classes > 0:
                    self.head_dist = nn.Linear(self.embed_dim, self.num_classes)
                else:
                    self.head_dist = nn.Identity()
        self.init_weights("")

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            if self.distilled:
                x_dist, x = x[:, 0, :], x[:, 1:, :].mean(dim=1)
                return self.fc_norm(x), self.dist_norm(x_dist)
            else:
                x = x[:, 1:, :].mean(dim=1)
                return self.fc_norm(x)
        else:
            x = self.norm(x)
            return x[:, 0]


class MAEBackbone(VisionTransformer):
    def __init__(self, img_size=(512, 512), patch_size=16, in_channels=3, embed_dims=192,
                 num_layers=12, num_heads=12, mlp_ratio=4, out_indices=(3, 5, 7, 11),
                 attn_drop_rate=0.0, drop_path_rate=0.1, norm_eval=False, init_values=None,
                 **kwargs):
        
        super().__init__(
            img_size=img_size[0] if isinstance(img_size, tuple) else img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dims,
            depth=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            global_pool=False, 
            distilled=False,
            num_classes=0,
            **kwargs
        )
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dims = embed_dims
        self.out_indices = out_indices
        self.norm_eval = norm_eval
        
    def forward(self, x):
        B, C, H, W = x.shape
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Extract features at specified layers
        features = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.out_indices:
                # Remove CLS token and reshape to spatial format
                feat = x[:, 1:, :]  # Remove CLS token
                H_out = H // self.patch_size
                W_out = W // self.patch_size
                feat = feat.transpose(1, 2).reshape(B, self.embed_dims, H_out, W_out)
                features.append(feat)
                
        return features

    def train(self, mode=True):
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.LayerNorm):
                    m.eval()


class Feature2Pyramid_Scale(nn.Module):
    """Feature2Pyramid with scale"""
    def __init__(self, embed_dim=192, rescales=[4, 2, 1, 0.5], scale_factor=0.25):
        super().__init__()
        self.embed_dim = embed_dim
        self.rescales = rescales
        self.scale_factor = scale_factor
        
        out_channels = int(embed_dim * scale_factor)

        # Create projection layers for each scale
        self.projections = nn.ModuleList([
            nn.Conv2d(embed_dim, out_channels, kernel_size=1)
            for _ in rescales
        ])
        
    def forward(self, features):
        outputs = []
        
        for i, (feat, rescale) in enumerate(zip(features, self.rescales)):
            # Project to 256 channels
            proj_feat = self.projections[i](feat)
            
            # Scale the feature map
            if rescale != 1.0:
                proj_feat = F.interpolate(
                    proj_feat, scale_factor=rescale, 
                    mode='bilinear', align_corners=False
                )
            
            outputs.append(proj_feat)
            
        return outputs


class FPNHead(nn.Module):
    """FPN Decode Head"""
    def __init__(self, in_channels=[48, 48, 48, 48], in_index=[0, 1, 2, 3],
                 feature_strides=[4, 8, 16, 32], channels=128, dropout_ratio=0.1,
                 num_classes=2, align_corners=False):
        super().__init__()
        
        self.in_channels = in_channels
        self.in_index = in_index
        self.feature_strides = feature_strides
        self.channels = channels
        self.dropout_ratio = dropout_ratio
        self.num_classes = num_classes
        self.align_corners = align_corners
        
        # Lateral convolutions
        self.lateral_convs = nn.ModuleList()
        for in_ch in in_channels:
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, channels, kernel_size=1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # FPN convolutions
        self.fpn_convs = nn.ModuleList()
        for _ in range(len(in_channels)):
            self.fpn_convs.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Final classifier
        if num_classes == 2:
            self.conv_seg = nn.Conv2d(channels, 1, kernel_size=1)
        else:
            self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
            
    def forward(self, features):
        # Apply lateral convolutions
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            laterals.append(lateral_conv(features[self.in_index[i]]))
        
        # Build FPN (top-down pathway)
        for i in range(len(laterals) - 1, 0, -1):
            upsampled = F.interpolate(
                laterals[i], size=laterals[i-1].shape[2:],
                mode='bilinear', align_corners=self.align_corners
            )
            laterals[i-1] = laterals[i-1] + upsampled
        
        # Apply FPN convolutions
        fpn_outs = []
        for i, fpn_conv in enumerate(self.fpn_convs):
            fpn_outs.append(fpn_conv(laterals[i]))
        
        # Fuse all levels
        target_size = fpn_outs[0].shape[2:]
        fused = fpn_outs[0]
        
        for i in range(1, len(fpn_outs)):
            upsampled = F.interpolate(
                fpn_outs[i], size=target_size,
                mode='bilinear', align_corners=self.align_corners
            )
            fused = fused + upsampled
        
        # Apply dropout
        if self.dropout is not None:
            fused = self.dropout(fused)
            
        # Final classification
        output = self.conv_seg(fused)
        
        return output


class SegmentationModel(nn.Module):
    """Complete segmentation model using the provided VisionTransformer"""
    def __init__(self, num_classes=2):
        super().__init__()
        
        self.backbone = MAEBackbone(
            img_size=(224, 224),
            patch_size=16,
            in_channels=3,
            embed_dims=192,
            num_layers=12,
            num_heads=12,
            mlp_ratio=4,
            out_indices=(3, 5, 7, 11),
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            norm_eval=False
        )
        
        # Neck
        self.neck = Feature2Pyramid_Scale(
            embed_dim=192,
            rescales=[4, 2, 1, 0.5],
            scale_factor=0.25
        )
        
        # Decode head
        self.decode_head = FPNHead(
            in_channels=[48, 48, 48, 48], 
            in_index=[0, 1, 2, 3],
            feature_strides=[4, 8, 16, 32],
            channels=128,
            dropout_ratio=0.1,
            num_classes=num_classes,
            align_corners=False
        )
        
    def forward(self, x):
        # Extract features from backbone
        features = self.backbone(x)
        
        # Transform features through neck
        neck_features = self.neck(features)
        
        # Decode to segmentation map
        seg_logits = self.decode_head(neck_features)
        
        # Upsample to input size
        seg_logits = F.interpolate(
            seg_logits, size=x.shape[2:],
            mode='bilinear', align_corners=False
        )
        
        return seg_logits