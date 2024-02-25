from tuSeg.model.nets.SBCNet.SBCNet import SBCNet


import pytorch_lightning as pl
import hydra
import torch
import torch.nn as nn
from tuSeg.utils import losses,pytorch_msssim

from tuSeg.utils.lr_scheduler import LinearWarmupCosineAnnealingLR


class SegModel(pl.LightningModule):
    def __init__(self, optim, net='SBCNet', loss_fun='Dice', type='Edge', patch_size = [96, 96, 96], **kwargs):
        super().__init__()
        self.save_hyperparameters()

        if net=='SBCNet':
            self.net = SBCNet(n_classes=3)

        if loss_fun=='Dice':
            self.loss_fun = losses.DiceLoss()

        self.val_loss_fun = losses.DiceLoss()
        self.patch_size = patch_size
        self.ce = nn.BCEWithLogitsLoss(reduction="mean")

        self.type = type
        self.ssim = pytorch_msssim.SSIM(data_range=1,channel=1, spatial_dims=3,nonnegative_ssim=True)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['labels']

        if self.type=="normal":
            y_hat, up, down = self(x)
            loss1,_ = self.loss_fun(y_hat, y)
            loss2,_ = self.loss_fun(up, y)
            loss3,_ = self.loss_fun(down, y)

            loss = loss1 + 0.5*loss2 + 0.5*loss3

            self.log('loss/train', loss1, on_step=False, on_epoch=True, sync_dist=True)
            self.log('loss/up', loss2, on_step=False, on_epoch=True, sync_dist=True)
            self.log('loss/down', loss3, on_step=False, on_epoch=True, sync_dist=True)
            self.log('loss/total', loss, on_step=False, on_epoch=True, sync_dist=True)
        
        elif self.type == 'Edge':
            x, y, edges = batch['image'], batch['labels'], batch['edges']
            y_hat,up, down, up_mid, down_mid, edges_hat = self(x)
            loss1, dice = self.loss_fun(y_hat, y)
            loss2, _ = self.loss_fun(up, y)
            loss3, _ = self.loss_fun(down, y)

            up_mid = torch.sigmoid(up_mid)
            down_mid = torch.sigmoid(down_mid)
            loss_ssim = 1 - self.ssim(up_mid, down_mid)

            loss_edge = self.ce(edges_hat, edges)
            loss = loss1 + 0.5*loss2 + 0.5*loss3 + 0.2*loss_ssim + 0.2*loss_edge

            self.log('loss/ssim', loss_ssim, on_step=False, on_epoch=True, sync_dist=True)

            self.log('loss/train', loss1, on_step=False, on_epoch=True, sync_dist=True)
            self.log('loss/up', loss2, on_step=False, on_epoch=True, sync_dist=True)
            self.log('loss/down', loss3, on_step=False, on_epoch=True, sync_dist=True)
            self.log('loss/total', loss, on_step=False, on_epoch=True, sync_dist=True)

            self.log('dice/liver_train', dice[0], on_step=False, on_epoch=True, sync_dist=True)
            self.log('dice/tumor_train', dice[1], on_step=False, on_epoch=True, sync_dist=True)

            self.log('loss/train_edge', loss_edge, on_step=False, on_epoch=True, sync_dist=True)

        else:
            y_hat = self(x)
            loss, dice = self.loss_fun(y_hat, y)
            self.log('loss/train', loss, on_step=False, on_epoch=True, sync_dist=True)
            self.log('dice/liver_train', dice[0], on_step=False, on_epoch=True, sync_dist=True)
            self.log('dice/tumor_train', dice[1], on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['labels']

        if self.type=="normal":
            y_hat, _, _ = self(x)
        elif self.type == 'Edge':
            y_hat,_,_,_,_,_ = self(x)
        else:
            y_hat = self(x)
        loss, dice = self.val_loss_fun(y_hat, y)

        self.log('loss/val', loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('dice/liver_val', dice[0], on_step=False, on_epoch=True, sync_dist=True)
        self.log('dice/tumor_val', dice[1], on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.hparams.optim, params=self.parameters())

        scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                                   warmup_epochs=50,
                                                   max_epochs=3000)

        return [optimizer], [scheduler]
