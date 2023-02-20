import torch
import torch.nn as nn
import torch.nn.functional as F
import models
from models import register


@register('diinn')
class DIINN(nn.Module):
    def __init__(self,
                 encoder_spec,
                 mode=3, init_q=False):
        super().__init__()

        self.encoder = models.make(encoder_spec)
        self.decoder = ImplicitDecoder(mode=mode, init_q=init_q)

    def forward(self, x, size, bsize=None):
        x = self.encoder(x)
        x = self.decoder(x, size, bsize)
        return x


class SineAct(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)


def patch_norm_2d(x, kernel_size=3):
    # B, C, H, W = x.shape
    # var, mean = torch.var_mean(F.unfold(x, kernel_size=kernel_size, padding=padding).view(B, C,kernel_size**2, H, W), dim=2, keepdim=False)
    # return (x - mean) / torch.sqrt(var + 1e-6)
    mean = F.avg_pool2d(x, kernel_size=kernel_size, padding=kernel_size // 2)
    mean_sq = F.avg_pool2d(x ** 2, kernel_size=kernel_size, padding=kernel_size // 2)
    var = mean_sq - mean ** 2
    return (x - mean) / (var + 1e-6)


class ImplicitDecoder(nn.Module):
    def __init__(self, in_channels=64, hidden_dims=[256, 256, 256, 256], mode=3, init_q=False):
        super().__init__()

        self.mode = mode
        self.init_q = init_q

        last_dim_K = in_channels * 9

        if self.init_q:
            self.first_layer = nn.Sequential(nn.Conv2d(3, in_channels * 9, 1),
                                             SineAct())
            last_dim_Q = in_channels * 9
        else:
            last_dim_Q = 3

        self.K = nn.ModuleList()
        self.Q = nn.ModuleList()
        if self.mode == 1:
            for hidden_dim in hidden_dims:
                self.K.append(nn.Sequential(nn.Conv2d(last_dim_K, hidden_dim, 1),
                                            nn.ReLU()))
                self.Q.append(nn.Sequential(nn.Conv2d(last_dim_Q, hidden_dim, 1),
                                            SineAct()))
                last_dim_K = hidden_dim
                last_dim_Q = hidden_dim
        elif self.mode == 2:
            for hidden_dim in hidden_dims:
                self.K.append(nn.Sequential(nn.Conv2d(last_dim_K, hidden_dim, 1),
                                            nn.ReLU()))
                self.Q.append(nn.Sequential(nn.Conv2d(last_dim_Q, hidden_dim, 1),
                                            SineAct()))
                last_dim_K = hidden_dim + in_channels * 9
                last_dim_Q = hidden_dim
        elif self.mode == 3:
            for hidden_dim in hidden_dims:
                self.K.append(nn.Sequential(nn.Conv2d(last_dim_K, hidden_dim, 1),
                                            nn.ReLU()))
                self.Q.append(nn.Sequential(nn.Conv2d(last_dim_Q, hidden_dim, 1),
                                            SineAct()))
                last_dim_K = hidden_dim + in_channels * 9
                last_dim_Q = hidden_dim
        elif self.mode == 4:
            for hidden_dim in hidden_dims:
                self.K.append(nn.Sequential(nn.Conv2d(last_dim_K, hidden_dim, 1),
                                            nn.ReLU()))
                self.Q.append(nn.Sequential(nn.Conv2d(last_dim_Q, hidden_dim, 1),
                                            SineAct()))
                last_dim_K = hidden_dim + in_channels * 9
                last_dim_Q = hidden_dim
        if self.mode == 4:
            self.last_layer = nn.Conv2d(hidden_dims[-1], 3, 3, padding=1, padding_mode='reflect')
        else:
            self.last_layer = nn.Conv2d(hidden_dims[-1], 3, 1)

    def _make_pos_encoding(self, x, size):
        B, C, H, W = x.shape
        H_up, W_up = size

        h_idx = -1 + 1 / H + 2 / H * torch.arange(H, device=x.device).float()
        w_idx = -1 + 1 / W + 2 / W * torch.arange(W, device=x.device).float()
        in_grid = torch.stack(torch.meshgrid(h_idx, w_idx, indexing='ij'), dim=0)

        h_idx_up = -1 + 1 / H_up + 2 / H_up * torch.arange(H_up, device=x.device).float()
        w_idx_up = -1 + 1 / W_up + 2 / W_up * torch.arange(W_up, device=x.device).float()
        up_grid = torch.stack(torch.meshgrid(h_idx_up, w_idx_up, indexing='ij'), dim=0)

        rel_grid = (up_grid - F.interpolate(in_grid.unsqueeze(0), size=(H_up, W_up),
                                            mode='nearest-exact'))  # important! mode='nearest' gives inconsistent results
        rel_grid[:, 0, :, :] *= H
        rel_grid[:, 1, :, :] *= W

        return rel_grid.contiguous().detach()

    def step(self, x, syn_inp):
        if self.init_q:
            syn_inp = self.first_layer(syn_inp)
            x = syn_inp * x
        if self.mode == 1:
            k = self.K[0](x)
            q = k * self.Q[0](syn_inp)

            for i in range(1, len(self.K)):
                k = self.K[i](k)
                q = k * self.Q[i](q)

            q = self.last_layer(q)
            return q
        elif self.mode == 2:
            k = self.K[0](x)
            q = k * self.Q[0](syn_inp)
            for i in range(1, len(self.K)):
                k = self.K[i](torch.cat([k, x], dim=1))
                q = k * self.Q[i](q)
            q = self.last_layer(q)
            return q
        elif self.mode == 3:
            k = self.K[0](x)
            q = k * self.Q[0](syn_inp)
            # q = k + self.Q[0](syn_inp)
            for i in range(1, len(self.K)):
                k = self.K[i](torch.cat([q, x], dim=1))
                q = k * self.Q[i](q)
                # q = k + self.Q[i](q)
            q = self.last_layer(q)
            return q
        elif self.mode == 4:
            k = self.K[0](x)
            q = k * self.Q[0](syn_inp)
            for i in range(1, len(self.K)):
                k = self.K[i](torch.cat([q, x], dim=1))
                q = k * self.Q[i](q)
            q = self.last_layer(q)
            return q

    def batched_step(self, x, syn_inp, bsize):
        with torch.no_grad():
            h, w = syn_inp.shape[-2:]
            ql = 0
            preds = []
            while ql < w:
                qr = min(ql + bsize // h, w)
                pred = self.step(x[:, :, :, ql: qr], syn_inp[:, :, :, ql: qr])
                preds.append(pred)
                ql = qr
            pred = torch.cat(preds, dim=-1)
        return pred

    def forward(self, x, size, bsize=None):
        B, C, H_in, W_in = x.shape
        rel_coord = self._make_pos_encoding(x, size).expand(B, -1, *size)  # 2
        ratio = x.new_tensor([(H_in * W_in) / (size[0] * size[1])]).view(1, -1, 1, 1).expand(B, -1, *size)  # 2
        syn_inp = torch.cat([rel_coord, ratio], dim=1)
        x = F.interpolate(F.unfold(x, 3, padding=1).view(B, C * 9, H_in, W_in), size=syn_inp.shape[-2:],
                          mode='nearest-exact')
        if bsize is None:
            pred = self.step(x, syn_inp)
        else:
            pred = self.batched_step(x, syn_inp, bsize)
        return pred
