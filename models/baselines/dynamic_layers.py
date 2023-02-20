import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm

class ScaleAwareAttention2d(nn.Module):
    def __init__(self, in_channels, ratios, K, temperature, init_weight=True):
        super().__init__()
        assert temperature % 3 == 1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if in_channels != 3:
            hidden_channels = int(in_channels * ratios) + 1
        else:
            hidden_channels = K
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_channels)
        self.fc2 = nn.Conv2d(hidden_channels + 2, K, 1, bias=True)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature != 1:
            self.temperature -= 3
            # print('Change temperature to:', str(self.temperature))

    def forward(self, x, scale):
        if not self.training:
            temperature = 1
        else:
            temperature = self.temperature

        batch_size = x.shape[0]
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = torch.cat(
            [x, torch.ones([batch_size, 2, 1, 1], device=x.device) * scale], dim=1
        )
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x / temperature, 1)


class ScaleAwareDynamicConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        ratio=0.25,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        K=4,
        temperature=34,
        init_weight=True,
    ):
        super().__init__()
        assert in_channels % groups == 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = ScaleAwareAttention2d(in_channels, ratio, K, temperature)

        self.weight = nn.Parameter(
            torch.randn(
                K, out_channels, in_channels // groups, kernel_size, kernel_size
            ),
            requires_grad=True,
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(K, out_channels))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x, scale):
        softmax_attention = self.attention(x, scale)
        batch_size, _, height, width = x.size()
        x = x.view(1, -1, height, width)
        weight = self.weight.view(self.K, -1)

        aggregate_weight = torch.mm(softmax_attention, weight).view(
            -1, self.in_channels, self.kernel_size, self.kernel_size
        )
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
        else:
            aggregate_bias = None
        output = F.conv2d(
            x,
            weight=aggregate_weight,
            bias=aggregate_bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups * batch_size,
        )
        output = output.view(
            batch_size, self.out_channels, output.size(-2), output.size(-1)
        )
        return output
