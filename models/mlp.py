import torch
import torch.nn as nn

from models import register


@register('mlp_pw')
class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        self.relu_0 = nn.ReLU(inplace=True)
        self.relu_1 = nn.ReLU(inplace=True)
        self.relu_2 = nn.ReLU(inplace=True)
        self.relu_3 = nn.ReLU(inplace=True)
        self.hidden=hidden_list[0]

    def forward(self, x,Coeff,basis,bias):
        # x(b*h*w,580)
        # Coeff(b*h*w,10)
        # basis[0](10,16*580)
        # basis[1](10,16*16)
        # basis[2](10,16*16)
        # basis[3](10,16*16)
        # basis[4](10,3*16)
        # bias[0](10,16)
        # bias[1](10,16)
        # bias[2](10,16)
        # bias[3](10,16)
        # bias[4](10,3)
        device=x.device
        # Applies a linear transformation to the incoming data: :math:`y = xA^T + b
        # Layer0
        x = x.unsqueeze(1)
        #  sum(  (b*h*w,1,580)*(b*h*w,16,580)  ,  dim=2  )  ->  (b*h*w,16)
        x = torch.sum(x*torch.matmul(Coeff.to(device),basis[0].to(device)).view(-1,self.hidden,580),dim=2)
        #  (b*h*w,16) + (b*h*w,16)  ->  (b*h*w,16)
        x = x + torch.matmul(Coeff.to(device),bias[0].to(device))
        x = self.relu_0(x)
        # Layer1
        x = x.unsqueeze(1)
        x = torch.sum(x*torch.matmul(Coeff.to(device),basis[1].to(device)).view(-1,self.hidden,self.hidden),dim=2)
        x = x + torch.matmul(Coeff.to(device),bias[1].to(device))
        x = self.relu_1(x)
        # Layer2
        x = x.unsqueeze(1)
        x = torch.sum(x*torch.matmul(Coeff.to(device),basis[2].to(device)).view(-1,self.hidden,self.hidden),dim=2)
        x = x + torch.matmul(Coeff.to(device),bias[2].to(device))
        x = self.relu_2(x)
        # Layer3
        x = x.unsqueeze(1)
        x = torch.sum(x*torch.matmul(Coeff.to(device),basis[3].to(device)).view(-1,self.hidden,self.hidden),dim=2)
        x = x + torch.matmul(Coeff.to(device),bias[3].to(device))
        x = self.relu_3(x)
        # Layer4
        x = x.unsqueeze(1)
        x = torch.sum(x*torch.matmul(Coeff.to(device),basis[4].to(device)).view(-1,3,self.hidden),dim=2)
        x = x + torch.matmul(Coeff.to(device),bias[4].to(device))

        return x


@register('mlp')
class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)
