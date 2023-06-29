import lightning
import torch
import torch.nn as nn
from torchmetrics.detection import MeanAveragePrecision


class ToyNet(lightning.LightningModule):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.metric = MeanAveragePrecision(iou_type="segm")

    def training_step(self, batch, batch_idx):
        return None

    def validation_step(self, batch, batch_idx):
        preds = []
        targets = []
        for _ in range(200):
            result = dict()
            num_preds = torch.randint(0, 100, (1,)).item()
            result['boxes'] = torch.rand((num_preds, 4), device=self.device)
            result['scores'] = torch.rand((num_preds,), device=self.device)
            result['labels'] = torch.randint(0, 10, (num_preds,),  device=self.device)
            result['masks'] = torch.randint(0, 2, (num_preds, 10, 10),  device=self.device).bool()
            preds.append(result)
            # parse gt
            gt = dict()
            num_gt = torch.randint(0, 10, (1,)).item()
            gt['boxes'] = torch.rand((num_gt, 4), device=self.device)
            gt['labels'] = torch.randint(0, 10, (num_gt,),  device=self.device)
            gt['masks'] = torch.randint(0, 2, (num_gt, 10, 10),  device=self.device).bool()
            targets.append(gt)
        self.metric.update(preds, targets)

    def on_validation_epoch_end(self) -> None:
        res = self.metric.compute()
        # if self.local_rank == 0:
        #     import ipdb; ipdb.set_trace()
        #
        # self.strategy.barrier()
        res.pop('classes')
        self.log_dict(res, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


if __name__ == '__main__':
    model = ToyNet()
    trainer = lightning.Trainer(accelerator='gpu', devices=2, max_epochs=1)
    val_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.rand((10, 1)), torch.rand((10, 1))))
    trainer.validate(model, val_dataloader)
