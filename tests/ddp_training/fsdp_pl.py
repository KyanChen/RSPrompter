import torch
import torchvision
from lightning.pytorch.strategies import FSDPStrategy
import sys
sys.path.append(sys.path[0] + '/../..')
import torch
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from torch.distributed.fsdp.wrap import wrap
from module.segment_anything import sam_model_registry


class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # torchvision.models.ResNet50_Weights()
        self.res = torchvision.models.resnet50().requires_grad_(True)
        # self.sam = sam_model_registry['default']().eval().requires_grad_(False)
        # self.img_encoder = sam_model_registry['default']().image_encoder.eval().requires_grad_(False)

    # def configure_sharded_model(self):
    #     self.res = wrap(self.res)
        # self.img_encoder = wrap(self.img_encoder)

    def configure_optimizers(self):
        # params = filter(lambda p: p.requires_grad, self.trainer.model.parameters())
        params = self.trainer.model.parameters()
        return torch.optim.AdamW(params, lr=1e-3)

    def training_step(self, *args, **kwargs):
        # if self.local_rank == 0:
        #     import ipdb;
        #     ipdb.set_trace()

        x = torch.rand(16, 3, 1024, 1024).cuda()
        # self.trainer.strategy.barrier()
        y = self.res(x)
        # x = x[:, [2, 1, 0], :, :]  # BGR -> RGB
        # x = x.contiguous()
        # x = (x - self.img_encoder.pixel_mean) / self.img_encoder.pixel_std
        # image_embeddings, inner_states = self.img_encoder(x)
        # y = image_embeddings
        y = y.sum()
        return y

train_dataloaders = torch.utils.data.DataLoader(torch.rand(1, 3, 1024, 1024), batch_size=1)

model = MyModel()
strategy = FSDPStrategy(cpu_offload=True)
trainer = Trainer(accelerator='gpu', devices=[2, 3], strategy='fsdp', precision=32, max_epochs=100)
trainer.fit(model, train_dataloaders=train_dataloaders)
# 单卡18G，使用
# 18042MiB
