import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

import models
from models import register
from utils import make_coord, to_coordinates

from mmcv.cnn import ConvModule
from .blocks.CSPLayer import CSPLayer


@register('rs_multiscale_super')
class RSMultiScaleSuper(nn.Module):
    def __init__(self,
                 encoder_spec,
                 multiscale=False,
                 neck=None,
                 decoder=None,
                 has_bn=True,
                 input_rgb=False,
                 n_forward_times=1,
                 global_decoder=None,
                 encode_scale_ratio=False
                 ):
        super().__init__()
        self.encoder = models.make(encoder_spec)
        self.multiscale = multiscale
        self.encoder_out_dim = self.encoder.out_dim
        self.encode_scale_ratio = encode_scale_ratio

        conv_cfg = None
        if has_bn:
            norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)
        else:
            norm_cfg = None
        act_cfg = dict(type='ReLU')

        if self.multiscale:
            self.multiscale_layers = nn.ModuleList()
            # 32->16->8->4
            num_blocks = [2, 4, 6]
            for n_idx in range(3):
                conv_layer = ConvModule(
                    self.encoder.out_dim,
                    self.encoder.out_dim*2,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg
                )
                csp_layer = CSPLayer(
                    self.encoder.out_dim*2,
                    self.encoder.out_dim,
                    num_blocks=num_blocks[n_idx],
                    add_identity=True,
                    use_depthwise=False,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
                self.multiscale_layers.append(nn.Sequential(conv_layer, csp_layer))

        if neck is not None:
            self.neck = models.make(neck, args={'in_dim': self.encoder.out_dim})
            modulation_dim = self.neck.d_dim
        else:
            modulation_dim = self.encoder.out_dim

        self.n_forward_times = n_forward_times

        self.input_rgb = input_rgb
        decoder_in_dim = 5 if self.input_rgb else 2
        if encode_scale_ratio:
            decoder_in_dim += 2

        if decoder is not None:
            self.decoder = models.make(decoder, args={'modulation_dim': modulation_dim, 'in_dim': decoder_in_dim})

        if global_decoder is not None:
            decoder_in_dim = 5 if self.input_rgb else 2
            if encode_scale_ratio:
                decoder_in_dim += 2

            self.decoder_is_proj = global_decoder.get('is_proj', False)
            self.grid_global = global_decoder.get('grid_global', False)

            self.global_decoder = models.make(global_decoder, args={'modulation_dim': modulation_dim, 'in_dim': decoder_in_dim})

            if self.decoder_is_proj:
                self.input_proj = nn.Sequential(
                    nn.Linear(modulation_dim, modulation_dim)
                    )
                self.output_proj = nn.Sequential(
                    nn.Linear(3, 3)
                    )

    def query_rgb(self, coord, cell=None):
        feat = self.feat

        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret

        if self.feat_unfold:
            # if kernel_size=2 and stride=2, x should has shape (B, 4*C, H/2*W/2)
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret

    def forward_step(self,
                     ori_img,
                     coord,
                     func_map,
                     global_content,
                     pred_rgb_value=None,
                     scale_ratio=None
                     ):
        weight_gen_func = 'bilinear'  # 'bilinear'
        #  grid： 先x再y
        coord_ = coord.clone().unsqueeze(1).flip(-1)  # Bx1xNxC
        funcs = F.grid_sample(
            func_map, coord_, padding_mode='border', mode=weight_gen_func, align_corners=True).squeeze(2)  # B C N
        funcs = rearrange(funcs, 'B C N -> (B N) C')

        feat_coord = to_coordinates(func_map.shape[-2:], return_map=True).to(func_map.device)
        feat_coord = repeat(feat_coord, 'H W C -> B C H W', B=coord.size(0))  # 坐标是[y, x]
        nearest_coord = F.grid_sample(feat_coord, coord_, mode='nearest', align_corners=True).squeeze(2)  # B 2 N
        nearest_coord = rearrange(nearest_coord, 'B C N -> B N C')  # B N 2

        relative_coord = coord - nearest_coord
        relative_coord[:, :, 0] *= func_map.shape[-2]
        relative_coord[:, :, 1] *= func_map.shape[-1]
        relative_coord = rearrange(relative_coord, 'B N C -> (B N) C')
        decoder_input = relative_coord

        interpolated_rgb = None
        if self.input_rgb:
            if pred_rgb_value is not None:
                interpolated_rgb = rearrange(pred_rgb_value, 'B N C -> (B N) C')
            else:
                interpolated_rgb = F.grid_sample(
                    ori_img, coord_, padding_mode='border', mode='bilinear', align_corners=True).squeeze(2)  # B 3 N
                interpolated_rgb = rearrange(interpolated_rgb, 'B C N -> (B N) C')
            decoder_input = torch.cat((decoder_input, interpolated_rgb), dim=-1)
        if self.encode_scale_ratio:
            scale_ratio = rearrange(scale_ratio, 'B N C -> (B N) C')
            decoder_input = torch.cat((decoder_input, scale_ratio), dim=-1)

        decoder_output = self.decoder(decoder_input, funcs)
        decoder_output = rearrange(decoder_output, '(B N) C -> B N C', B=func_map.size(0))

        if hasattr(self, 'global_decoder'):
            # coord: BxNx2
            # global_content: Bx1xC
            if self.decoder_is_proj:
                global_content = self.input_proj(global_content)  # B 1 C
            global_funcs = repeat(global_content, 'B C -> B N C', N=coord.size(1))
            global_funcs = rearrange(global_funcs, 'B N C -> (B N) C')

            if self.grid_global:
                global_decoder_input = decoder_input
            else:
                global_decoder_input = rearrange(coord, 'B N C -> (B N) C')
                if self.input_rgb:
                    global_decoder_input = torch.cat((global_decoder_input, interpolated_rgb), dim=-1)
                if self.encode_scale_ratio:
                    global_decoder_input = torch.cat((global_decoder_input, scale_ratio), dim=-1)

            global_decoder_output = self.global_decoder(global_decoder_input, global_funcs)
            global_decoder_output = rearrange(global_decoder_output, '(B N) C -> B N C', B=func_map.size(0))

            if self.decoder_is_proj:
                decoder_output = self.output_proj(global_decoder_output + decoder_output)
            else:
                decoder_output = global_decoder_output + decoder_output

        return decoder_output

    def forward_backbone(self, inp):
        # inp: img-BxCxHxW
        return self.encoder(inp)

    def forward_multiscale(self, feats, keep_ori_featmap=False):
        if keep_ori_featmap:
            output_feats = feats
        else:
            output_feats = []
        x = feats[0]
        for layer in self.multiscale_layers:
            x = layer(x)
            output_feats.append(x)
        return output_feats

    def forward(self, inp, coord, scale_ratio=None):
        output_feats = [self.forward_backbone(inp)]
        if self.multiscale:
            output_feats = self.forward_multiscale(output_feats)
        if hasattr(self, 'neck'):
            global_content, func_maps = self.neck(output_feats)
        else:
            global_content = None
            func_maps = output_feats[0]

        pred_rgb_value = None
        return_pred_rgb_value = []

        for n_time in range(self.n_forward_times):
            pred_rgb_value = self.forward_step(inp, coord, func_maps, global_content, pred_rgb_value, scale_ratio)
            return_pred_rgb_value.append(pred_rgb_value)
        return return_pred_rgb_value


