import torch
import torch.nn as nn

from models import register
from torch.nn import TransformerEncoderLayer, TransformerEncoder, LayerNorm
from torch.nn.init import xavier_uniform_
from einops import rearrange
# from mmcv.runner.base_module import BaseModule, ModuleList
# from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, PatchEmbed
# from mmcv.cnn.utils.weight_init import trunc_normal_
# from mmcv.cnn.bricks.registry import DROPOUT_LAYERS


@register('transformer_neck')
class TransformerNeck(nn.Module):
    def __init__(self,
                 in_dim,
                 d_dim=256,
                 downsample=True,
                 has_pe=True,
                 has_norm=True,
                 class_token=True,
                 num_encoder_layers=3,
                 dim_feedforward=512,
                 drop_rate=0.1
                 ):
        super().__init__()
        self.input_proj = nn.Conv2d(in_dim, d_dim, kernel_size=1)
        self.downsample = downsample

        if self.downsample:
            self.sampler = nn.Conv2d(d_dim, d_dim, kernel_size=3, stride=2, padding=1)

        encoder_layer = TransformerEncoderLayer(
            d_model=d_dim,
            nhead=8,
            dim_feedforward=dim_feedforward,
            dropout=drop_rate,
            activation='gelu',
            batch_first=True
        )
        if has_norm:
            encoder_norm = LayerNorm(d_dim)
        else:
            encoder_norm = None
        self.trans_encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        if self.downsample:
            self.uplayer = nn.Sequential(
                nn.Conv2d(d_dim, d_dim*4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.Conv2d(d_dim, d_dim, kernel_size=3, padding=1)
            )

        # for p in self.parameters():
        #     if p.dim() > 1:
        #         xavier_uniform_(p)

        self.d_dim = d_dim
        self.class_token = class_token
        self.has_pe = has_pe

        if self.class_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_dim))

        if self.has_pe:
            self.pos_embed = nn.Parameter(
                torch.randn(1, d_dim, 24, 24) * 0.02
            )
            self.drop_after_pos = nn.Dropout(p=drop_rate)

    def forward(self, x):
        # x: List
        x = [self.input_proj(x_tmp) for x_tmp in x]
        if self.downsample:
            x[0] = self.sampler(x[0])
        B, C, H, W = x[0].shape

        if self.has_pe:
            assert W == self.pos_embed.shape[-1]
            x_with_pe = []
            for x_tmp in x:
                b_tmp, c_tmp, h_tmp, w_tmp = x_tmp.shape
                pe = nn.functional.interpolate(self.pos_embed, size=[h_tmp, w_tmp], mode='bicubic', align_corners=True)
                x_tmp = x_tmp + pe
                x_with_pe.append(x_tmp)
        else:
            x_with_pe = x

        x_flatten = []
        for x in x_with_pe:
            x = rearrange(x, 'b c h w -> b (h w) c')
            x_flatten.append(x)
        x_flatten = torch.cat(x_flatten, dim=1)

        if self.has_pe:
            x_flatten = self.drop_after_pos(x_flatten)

        if self.class_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x_flatten), dim=1)
        else:
            x = x_flatten

        x = self.trans_encoder(x)

        if self.class_token:
            global_content = x[:, 0]
            func_map = x[:, 1:(1 + H * W), :]
        else:
            global_content = x.mean(dim=1)  # global pool without cls token
            func_map = x[:, :(H * W), :]

        func_map = rearrange(func_map, 'b (h w) c -> b c h w', h=H)
        if self.downsample:
            func_map = self.uplayer(func_map)
        return global_content, func_map
