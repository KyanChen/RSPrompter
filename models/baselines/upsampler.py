import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as wn

################
# Upsampler
################

def make_coord(shape, ranges=None, flatten=True):
    """Make coordinates at grid centers."""
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


class UPLayer_MS_V9(nn.Module):
    # Up-sampling net
    def __init__(self, n_feats, kSize, out_channels, interpolate_mode, levels=4):
        super().__init__()
        self.interpolate_mode = interpolate_mode
        self.levels = levels

        self.UPNet_x2_list = []

        for _ in range(levels - 1):
            self.UPNet_x2_list.append(
                nn.Sequential(
                    *[
                        nn.Conv2d(
                            n_feats,
                            n_feats * 4,
                            kSize,
                            padding=(kSize - 1) // 2,
                            stride=1,
                        ),
                        nn.PixelShuffle(2),
                    ]
                )
            )

        self.scale_aware_layer = nn.Sequential(
            *[nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, levels), nn.Sigmoid()]
        )

        self.UPNet_x2_list = nn.Sequential(*self.UPNet_x2_list)

        self.fuse = nn.Sequential(
            *[
                nn.Conv2d(n_feats * levels, 256, kernel_size=1, padding=0, stride=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1),
                nn.ReLU(),
                nn.Conv2d(256, out_channels, kernel_size=1, padding=0, stride=1),
            ]
        )

    def forward(self, x, out_size):

        if type(out_size) == int:
            out_size = [out_size, out_size]

        if type(x) == list:
            return self.forward_list(x, out_size)

        r = torch.tensor([x.shape[2] / out_size[0]], device="cuda")

        scale_w = self.scale_aware_layer(r.unsqueeze(0))[0]

        # scale_in = x.new_tensor(np.ones([x.shape[0], 1, out_size[0], out_size[1]])*r)

        x_list = [x]
        for l in range(1, self.levels):
            x_list.append(self.UPNet_x2_list[l - 1](x_list[l - 1]))

        x_resize_list = []
        for l in range(self.levels):
            x_resize = F.interpolate(
                x_list[l], out_size, mode=self.interpolate_mode, align_corners=False
            )
            x_resize *= scale_w[l]
            x_resize_list.append(x_resize)

        # x_resize_list.append(scale_in)
        out = self.fuse(torch.cat(tuple(x_resize_list), 1))
        return out

    def forward_list(self, h_list, out_size):
        assert (
            len(h_list) == self.levels
        ), "The Length of input list must equal to the number of levels"
        device = h_list[0].device
        r = torch.tensor([h_list[0].shape[2] / out_size[0]], device=device)
        scale_w = self.scale_aware_layer(r.unsqueeze(0))[0]

        x_resize_list = []
        for l in range(self.levels):
            h = h_list[l]
            for i in range(l):
                h = self.UPNet_x2_list[i](h)
            x_resize = F.interpolate(
                h, out_size, mode=self.interpolate_mode, align_corners=False
            )
            x_resize *= scale_w[l]
            x_resize_list.append(x_resize)

        out = self.fuse(torch.cat(tuple(x_resize_list), 1))
        return out


class UPLayer_MS_WN(nn.Module):
    # Up-sampling net
    def __init__(self, n_feats, kSize, out_channels, interpolate_mode, levels=4):
        super().__init__()
        self.interpolate_mode = interpolate_mode
        self.levels = levels
        self.UPNet_x2_list = []

        for _ in range(levels - 1):
            self.UPNet_x2_list.append(
                nn.Sequential(
                    *[
                        wn(
                            nn.Conv2d(
                                n_feats,
                                n_feats * 4,
                                kSize,
                                padding=(kSize - 1) // 2,
                                stride=1,
                            )
                        ),
                        nn.PixelShuffle(2),
                    ]
                )
            )

        self.scale_aware_layer = nn.Sequential(
            *[wn(nn.Linear(1, 64)), nn.ReLU(), wn(nn.Linear(64, levels)), nn.Sigmoid()]
        )

        self.UPNet_x2_list = nn.Sequential(*self.UPNet_x2_list)

        self.fuse = nn.Sequential(
            *[
                wn(
                    nn.Conv2d(n_feats * levels, 256, kernel_size=1, padding=0, stride=1)
                ),
                nn.ReLU(),
                wn(nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1)),
                nn.ReLU(),
                wn(nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1)),
                nn.ReLU(),
                wn(nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1)),
                nn.ReLU(),
                wn(nn.Conv2d(256, out_channels, kernel_size=1, padding=0, stride=1)),
            ]
        )

        assert self.interpolate_mode in (
            "bilinear",
            "bicubic",
            "nearest",
            "MLP",
        ), "Interpolate mode must be bilinear/bicubic/nearest/MLP"
        if self.interpolate_mode == "MLP":
            self.feature_interpolater = MLP_Interpolate(n_feats, radius=3)
        elif self.interpolate_mode == "nearest":
            self.feature_interpolater = lambda x, out_size: F.interpolate(
                x, out_size, mode=self.interpolate_mode
            )
        else:
            self.feature_interpolater = lambda x, out_size: F.interpolate(
                x, out_size, mode=self.interpolate_mode, align_corners=False
            )

    def forward(self, x, out_size):
        if type(out_size) == int:
            out_size = [out_size, out_size]

        if type(x) == list:
            return self.forward_list(x, out_size)

        r = torch.tensor([x.shape[2] / out_size[0]], device="cuda")

        scale_w = self.scale_aware_layer(r.unsqueeze(0))[0]

        x_list = [x]
        for l in range(1, self.levels):
            x_list.append(self.UPNet_x2_list[l - 1](x_list[l - 1]))

        x_resize_list = []
        for l in range(self.levels):
            x_resize = self.feature_interpolater(x_list[l], out_size)
            x_resize *= scale_w[l]
            x_resize_list.append(x_resize)

        out = self.fuse(torch.cat(tuple(x_resize_list), 1))
        return out

    def forward_list(self, h_list, out_size):
        assert (
            len(h_list) == self.levels
        ), "The Length of input list must equal to the number of levels"
        device = h_list[0].device
        r = torch.tensor([h_list[0].shape[2] / out_size[0]], device=device)
        scale_w = self.scale_aware_layer(r.unsqueeze(0))[0]

        x_resize_list = []
        for l in range(self.levels):
            h = h_list[l]
            for i in range(l):
                h = self.UPNet_x2_list[i](h)
            x_resize = self.feature_interpolater(h, out_size)
            x_resize *= scale_w[l]
            x_resize_list.append(x_resize)

        out = self.fuse(torch.cat(tuple(x_resize_list), 1))
        return out


class UPLayer_MS_WN_woSA(UPLayer_MS_WN):
    def __init__(self, n_feats, kSize, out_channels, interpolate_mode, levels=4):
        super().__init__(n_feats, kSize, out_channels, interpolate_mode, levels)

    def forward(self, x, out_size):
        if type(out_size) == int:
            out_size = [out_size, out_size]

        if type(x) == list:
            return self.forward_list(x, out_size)

        x_list = [x]
        for l in range(1, self.levels):
            x_list.append(self.UPNet_x2_list[l - 1](x_list[l - 1]))

        x_resize_list = []
        for l in range(self.levels):
            x_resize = self.feature_interpolater(x_list[l], out_size)
            x_resize_list.append(x_resize)

        out = self.fuse(torch.cat(tuple(x_resize_list), 1))
        return out

    def forward_list(self, h_list, out_size):
        assert (
            len(h_list) == self.levels
        ), "The Length of input list must equal to the number of levels"

        x_resize_list = []
        for l in range(self.levels):
            h = h_list[l]
            for i in range(l):
                h = self.UPNet_x2_list[i](h)
            x_resize = self.feature_interpolater(h, out_size)
            x_resize_list.append(x_resize)

        out = self.fuse(torch.cat(tuple(x_resize_list), 1))
        return out


class OSM(nn.Module):
    def __init__(self, n_feats, overscale):
        super().__init__()
        self.body = nn.Sequential(
            wn(nn.Conv2d(n_feats, 1600, 3, padding=1)),
            nn.PixelShuffle(overscale),
            wn(nn.Conv2d(64, 3, 3, padding=1)),
        )

    def forward(self, x, out_size):
        h = self.body(x)
        return F.interpolate(h, out_size, mode="bicubic", align_corners=False)


class MLP_Interpolate(nn.Module):
    def __init__(self, n_feat, radius=2):
        super().__init__()
        self.radius = radius

        self.f_transfer = nn.Sequential(
            *[
                nn.Linear(n_feat * self.radius * self.radius + 2, n_feat),
                nn.ReLU(True),
                nn.Linear(n_feat, n_feat),
            ]
        )

    def forward(self, x, out_size):
        x_unfold = F.unfold(x, self.radius, padding=self.radius // 2)
        x_unfold = x_unfold.view(
            x.shape[0], x.shape[1] * (self.radius ** 2), x.shape[2], x.shape[3]
        )

        in_shape = x.shape[-2:]
        in_coord = (
            make_coord(in_shape, flatten=False)
            .cuda()
            .permute(2, 0, 1)
            .unsqueeze(0)
            .expand(x.shape[0], 2, *in_shape)
        )

        if type(out_size) == int:
            out_size = [out_size, out_size]

        out_coord = make_coord(out_size, flatten=True).cuda()
        out_coord = out_coord.expand(x.shape[0], *out_coord.shape)

        q_feat = F.grid_sample(
            x_unfold,
            out_coord.flip(-1).unsqueeze(1),
            mode="nearest",
            align_corners=False,
        )[:, :, 0, :].permute(0, 2, 1)
        q_coord = F.grid_sample(
            in_coord,
            out_coord.flip(-1).unsqueeze(1),
            mode="nearest",
            align_corners=False,
        )[:, :, 0, :].permute(0, 2, 1)

        rel_coord = out_coord - q_coord
        rel_coord[:, :, 0] *= x.shape[-2]
        rel_coord[:, :, 1] *= x.shape[-1]

        inp = torch.cat([q_feat, rel_coord], dim=-1)

        bs, q = out_coord.shape[:2]
        pred = self.f_transfer(inp.view(bs * q, -1)).view(bs, q, -1)
        pred = (
            pred.view(x.shape[0], *out_size, x.shape[1])
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        return pred


class LIIF_Upsampler(nn.Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError

    def forward(self):
        pass
