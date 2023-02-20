import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

import models
from models import register
from utils import make_coord, to_coordinates

from mmcv.cnn import ConvModule
from .blocks.CSPLayer import CSPLayer


@register('funsr')
class FUNSR(nn.Module):
    def __init__(self,
                 encoder_spec,
                 has_multiscale=False,
                 neck=None,
                 decoder=None,
                 global_decoder=None,
                 encoder_rgb=False,
                 n_forward_times=1,
                 encode_hr_coord=False,
                 has_bn=True,
                 encode_scale_ratio=False,
                 local_unfold=False,
                 weight_gen_func='nearest-exact',
                 return_featmap=False,
                 ):
        super().__init__()
        self.weight_gen_func = weight_gen_func  # 'bilinear', 'nearest-exact'
        self.encoder = models.make(encoder_spec)
        self.encoder_out_dim = self.encoder.out_dim
        self.encode_scale_ratio = encode_scale_ratio
        self.has_multiscale = has_multiscale
        self.encoder_rgb = encoder_rgb
        self.encode_hr_coord = encode_hr_coord
        self.local_unfold = local_unfold
        self.return_featmap = return_featmap

        self.multiscale_layers = nn.ModuleList()

        if self.has_multiscale:
            # 48->24->12->6
            conv_cfg = None
            if has_bn:
                norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)
            else:
                norm_cfg = None
            act_cfg = dict(type='ReLU')
            num_blocks = [2, 4, 6]
            for n_idx in range(3):
                conv_layer = ConvModule(
                    self.encoder_out_dim,
                    self.encoder_out_dim*2,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg
                )
                csp_layer = CSPLayer(
                    self.encoder_out_dim*2,
                    self.encoder_out_dim,
                    num_blocks=num_blocks[n_idx],
                    add_identity=True,
                    use_depthwise=False,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
                self.multiscale_layers.append(nn.Sequential(conv_layer, csp_layer))

        if neck is not None:
            self.neck = models.make(neck, args={'in_dim': self.encoder_out_dim})
            modulation_dim = self.neck.d_dim
        else:
            modulation_dim = self.encoder_out_dim

        self.n_forward_times = n_forward_times

        decoder_in_dim = 2
        if self.encode_scale_ratio:
            decoder_in_dim += 2
        if self.encode_hr_coord:
            decoder_in_dim += 2
        if self.encoder_rgb:
            decoder_in_dim += 3

        if decoder is not None:
            if self.local_unfold:
                self.down_dim_layer = nn.Conv2d(modulation_dim * 9, modulation_dim, 1)
            self.decoder = models.make(decoder, args={'modulation_dim': modulation_dim, 'in_dim': decoder_in_dim})

        if global_decoder is not None:
            decoder_in_dim = 2
            if self.encode_scale_ratio:
                decoder_in_dim += 2
            if self.encoder_rgb:
                decoder_in_dim += 3

            self.decoder_is_proj = global_decoder.get('is_proj', False)

            self.global_decoder = models.make(global_decoder, args={'modulation_dim': modulation_dim, 'in_dim': decoder_in_dim})

            if self.decoder_is_proj:
                self.input_proj = nn.Linear(modulation_dim, modulation_dim)
                # self.output_proj = nn.Conv2d(6, 3, kernel_size=3, padding=1)
                self.output_proj = nn.Conv2d(6, 3, kernel_size=1)

    def forward_step(self,
                     lr_img,
                     func_map,
                     global_func,
                     rel_coord,
                     lr_coord,
                     hr_coord,
                     scale_ratio_map=None,
                     pred_rgb_value=None
                     ):
        # Expand funcmap
        if self.local_unfold:
            b, c, h, w = func_map.shape
            func_map = F.unfold(func_map, 3, padding=1).view(b, c * 9, h, w)
            func_map = self.down_dim_layer(func_map)
        local_func_map = F.interpolate(func_map, size=hr_coord.shape[-2:], mode=self.weight_gen_func)

        rel_coord = repeat(rel_coord, 'b c h w -> (B b) c h w', B=lr_img.size(0))
        hr_coord = repeat(hr_coord, 'c h w -> B c h w', B=lr_img.size(0))
        local_input = rel_coord
        if self.encode_scale_ratio:
            local_input = torch.cat([local_input, scale_ratio_map], dim=1)
        if self.encode_hr_coord:
            local_input = torch.cat([local_input, hr_coord], dim=1)
        if self.encoder_rgb:
            if pred_rgb_value is None:
                pred_rgb_value = F.interpolate(lr_img, size=hr_coord.shape[-2:], mode='bicubic', align_corners=True)
            local_input = torch.cat((local_input, pred_rgb_value), dim=1)

        decoder_output = self.decoder(local_input, local_func_map)

        if hasattr(self, 'global_decoder'):
            if self.decoder_is_proj:
                global_func = self.input_proj(global_func)  # B C
            global_func = repeat(global_func, 'B C -> B C H W', H=hr_coord.shape[2], W=hr_coord.shape[3])

            global_input = hr_coord
            if self.encode_scale_ratio:
                global_input = torch.cat([global_input, scale_ratio_map], dim=1)
            if self.encoder_rgb:
                if pred_rgb_value is None:
                    pred_rgb_value = F.interpolate(lr_img, size=hr_coord.shape[-2:], mode='bicubic',
                                                   align_corners=True)
                global_input = torch.cat((global_input, pred_rgb_value), dim=1)
            global_decoder_output = self.global_decoder(global_input, global_func)

            returned_featmap = None
            if self.decoder_is_proj:
                if self.return_featmap:
                    returned_featmap = torch.cat((global_decoder_output, decoder_output), dim=1)
                decoder_output = self.output_proj(torch.cat((global_decoder_output, decoder_output), dim=1))
            else:
                decoder_output = global_decoder_output + decoder_output

        return decoder_output, returned_featmap

    def forward_backbone(self, x, keep_ori_feat=True):
        # x: img-BxCxHxW
        x = self.encoder(x)
        output_feats = []
        if keep_ori_feat:
            output_feats.append(x)
        for layer in self.multiscale_layers:
            x = layer(x)
            output_feats.append(x)
        return output_feats

    def get_coordinate_map(self, x, hr_size):
        B, C, H, W = x.shape
        H_up, W_up = hr_size
        x_coord = to_coordinates(x.shape[-2:], return_map=True).to(x.device).permute(2, 0, 1)
        hr_coord = to_coordinates(hr_size, return_map=True).to(x.device).permute(2, 0, 1)
        # important! mode='nearest' gives inconsistent results
        # import pdb
        # pdb.set_trace()
        rel_grid = hr_coord - F.interpolate(x_coord.unsqueeze(0), size=hr_size, mode='nearest-exact')
        rel_grid[:, 0, :, :] *= H
        rel_grid[:, 1, :, :] *= W

        return rel_grid.contiguous().detach(), x_coord.contiguous().detach(), hr_coord.contiguous().detach()

    def forward(self, x, out_size):
        B, C, H_lr, W_lr = x.shape
        output_feats = self.forward_backbone(x)  # List
        if hasattr(self, 'neck'):
            global_content, func_map = self.neck(output_feats)
        else:
            global_content = None
            func_map = output_feats[0]
        rel_coord, lr_coord, hr_coord = self.get_coordinate_map(x, out_size)
        scale_ratio_map = None
        if self.encode_scale_ratio:
            h_ratio = x.shape[2] / out_size[0]
            w_ratio = x.shape[3] / out_size[1]
            scale_ratio_map = torch.tensor([h_ratio, w_ratio]).view(1, -1, 1, 1).expand(B, -1, *out_size).to(x.device)

        pred_rgb_value = None
        return_pred_rgb_value = []

        for n_time in range(self.n_forward_times):
            pred_rgb_value, returned_featmaps = self.forward_step(
                x,
                func_map,
                global_content,
                rel_coord,
                lr_coord,
                hr_coord,
                scale_ratio_map,
                pred_rgb_value
            )
            return_pred_rgb_value.append(pred_rgb_value)
        if self.return_featmap:
            return return_pred_rgb_value, returned_featmaps
        return return_pred_rgb_value


