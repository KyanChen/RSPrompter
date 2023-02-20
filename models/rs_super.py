import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

import models
from models import register
from utils import make_coord, to_coordinates


@register('rs_super')
class RSSuper(nn.Module):
    def __init__(self,
                 encoder_spec,
                 neck=None,
                 decoder=None,
                 input_rgb=True,
                 n_forward_times=1,
                 global_decoder=None
                 ):
        super().__init__()
        self.n_forward_times = n_forward_times
        self.encoder = models.make(encoder_spec)
        if neck is not None:
            self.neck = models.make(neck, args={'in_dim': self.encoder.out_dim})

        self.input_rgb = input_rgb
        decoder_in_dim = 5 if self.input_rgb else 2
        if decoder is not None:
            self.decoder = models.make(decoder, args={'modulation_dim': self.neck.out_dim, 'in_dim': decoder_in_dim})

        if global_decoder is not None:
            decoder_in_dim = 5 if self.input_rgb else 2
            self.decoder_is_proj = global_decoder.get('is_proj', False)
            self.grid_global = global_decoder.get('grid_global', False)

            self.global_decoder = models.make(global_decoder, args={'modulation_dim': self.neck.out_dim, 'in_dim': decoder_in_dim})

            if self.decoder_is_proj:
                self.input_proj = nn.Sequential(
                    nn.Linear(self.neck.out_dim, self.neck.out_dim)
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

    def forward_backbone_neck(self, inp, coord):
        # inp: 64x3x32x32
        # coord: BxNx2
        feat = self.encoder(inp)  # 64x64x32x32
        global_content, x_rep = self.neck(feat)  # Bx1xC; BxCxHxW
        return feat, x_rep, global_content

    def forward_step(self, inp, coord, feat, x_rep, global_content, pred_rgb_value=None):
        weight_gen_func = 'bilinear'  # 'bilinear'
        #  grid： 先x再y
        coord_ = coord.clone().unsqueeze(1).flip(-1)  # Bx1xNxC
        modulations = F.grid_sample(x_rep, coord_, padding_mode='border', mode=weight_gen_func,
                                    align_corners=True).squeeze(2)  # B C N
        modulations = rearrange(modulations, 'B C N -> (B N) C')

        feat_coord = to_coordinates(feat.shape[-2:], return_map=True).to(inp.device)
        feat_coord = repeat(feat_coord, 'H W C -> B C H W', B=inp.size(0))  # 坐标是[y, x]
        nearest_coord = F.grid_sample(feat_coord, coord_, mode='nearest', align_corners=True).squeeze(2)  # B 2 N
        nearest_coord = rearrange(nearest_coord, 'B C N -> B N C')  # B N 2

        relative_coord = coord - nearest_coord
        relative_coord[:, :, 0] *= feat.shape[-2]
        relative_coord[:, :, 1] *= feat.shape[-1]
        relative_coord = rearrange(relative_coord, 'B N C -> (B N) C')
        decoder_input = relative_coord

        interpolated_rgb = None
        if self.input_rgb:
            if pred_rgb_value is not None:
                interpolated_rgb = rearrange(pred_rgb_value, 'B N C -> (B N) C')
            else:
                interpolated_rgb = F.grid_sample(inp, coord_, padding_mode='border', mode='bilinear', align_corners=True).squeeze(2)  # B 3 N
                interpolated_rgb = rearrange(interpolated_rgb, 'B C N -> (B N) C')
            decoder_input = torch.cat((decoder_input, interpolated_rgb), dim=-1)

        decoder_output = self.decoder(decoder_input, modulations)
        decoder_output = rearrange(decoder_output, '(B N) C -> B N C', B=inp.size(0))

        if hasattr(self, 'global_decoder'):
            # coord: BxNx2
            # global_content: Bx1xC
            if self.decoder_is_proj:
                global_content = self.input_proj(global_content)  # B 1 C
            global_modulations = repeat(global_content, 'B N C -> B (N S) C', S=coord.size(1))
            global_modulations = rearrange(global_modulations, 'B N C -> (B N) C')

            if self.grid_global:
                # import pdb
                # pdb.set_trace()
                global_decoder_input = decoder_input
            else:
                global_decoder_input = rearrange(coord, 'B N C -> (B N) C')
                if self.input_rgb:
                    global_decoder_input = torch.cat((global_decoder_input, interpolated_rgb), dim=-1)

            global_decoder_output = self.global_decoder(global_decoder_input, global_modulations)
            global_decoder_output = rearrange(global_decoder_output, '(B N) C -> B N C', B=inp.size(0))

            if self.decoder_is_proj:
                decoder_output = self.output_proj(global_decoder_output + decoder_output)
            else:
                decoder_output = global_decoder_output + decoder_output

        return decoder_output

    def forward(self, inp, coord):
        # import pdb
        # pdb.set_trace()
        pred_rgb_value = None
        feat, x_rep, global_content = self.forward_backbone_neck(inp, coord)
        return_pred_rgb_value = []
        for n_time in range(self.n_forward_times):
            pred_rgb_value = self.forward_step(inp, coord, feat, x_rep, global_content, pred_rgb_value)
            return_pred_rgb_value.append(pred_rgb_value)
        return return_pred_rgb_value


