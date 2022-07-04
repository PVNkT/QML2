from typing import List, Optional, Dict, Any

import torch
from torch import optim
from torch.optim import lr_scheduler
import torch.nn.functional as F

from pytorch_lightning import LightningModule
from torchmetrics import (
    MetricCollection,
    Accuracy,
    Precision,
    Recall,
)
import torchmetrics.functional as MF

from src import models


class ClassificationTask(LightningModule):
    prefix = ""

    def __init__(self, opt: Dict = None, net: Dict = None, inputs: Any = None) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = getattr(models, net.model)(net)
        self.get_metrics = MetricCollection(
            [
                Accuracy(num_classes=self.hparams.net.num_classes),
                # Precision(average="macro", num_classes=self.hparams.net.num_classes),
                # Recall(average="macro", num_classes=self.hparams.net.num_classes),
            ]
        )

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, batch_idx):
        x, targets = batch
        preds = self.model(x)
        loss = F.cross_entropy(preds, targets)
        # loss = F.nll_loss(preds, targets)
        return loss, preds, targets

    def _shared_epoch_end(self, outputs, loss_name, task=None):
        all_preds = torch.cat([x["preds"] for x in outputs])
        all_y = torch.cat([x["targets"] for x in outputs])
        avg_loss = torch.stack([x[loss_name] for x in outputs]).mean().cpu().detach()
        metrics = self.get_metrics(all_preds, all_y)
        logger_log = {
            f"{self.prefix.upper()}Loss/{task}": avg_loss.item(),
            **{
                f"{self.prefix.upper()}{key}/{task}": value.cpu().detach()
                for key, value in metrics.items()
            },
            "step": torch.tensor(self.current_epoch, dtype=torch.float32),
        }
        return avg_loss, logger_log

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self._shared_step(batch, batch_idx)
        return {"loss": loss, "preds": preds.detach(), "targets": targets}

    def training_epoch_end(self, outputs):
        avg_loss, logger_log = self._shared_epoch_end(outputs, "loss", "train")
        self.log_dict(logger_log)

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self._shared_step(batch, batch_idx)
        return {"val_loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs):
        avg_loss, logger_log = self._shared_epoch_end(outputs, "val_loss", "val")
        self.log_dict(logger_log)

    def test_step(self, batch, batch_idx):
        loss, preds, targets = self._shared_step(batch, batch_idx)
        return {"test_loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs):
        avg_loss, logger_log = self._shared_epoch_end(outputs, "test_loss", "test")
        self.log_dict(logger_log)

    def configure_optimizers(self):
        opt = getattr(optim, self.hparams.opt.optimizer)(
            self.model.parameters(), lr=self.hparams.opt.lr, weight_decay=5e-4, betas=(0.9, 0.999)
        )
        # sch = lr_scheduler.StepLR(opt, step_size=1, gamma=0.99)
        sch = lr_scheduler.StepLR(opt, step_size=1, gamma=0.99)
        return [opt]
        # return [opt], [sch]
