from typing import List
import matplotlib.pyplot as plt
import seaborn as sn
import torch
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LoggerCollection, TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score

def get_tensorboard_logger(trainer: Trainer) -> TensorBoardLogger:
    """Safely get Weights&Biases logger from Trainer."""
    """
    if trainer.fast_dev_run:
        raise Exception(
            "Cannot use Tensorboard callbacks since pytorch lightning disables loggers in `fast_dev_run=true` mode."
        )
    """
    if isinstance(trainer.logger, TensorBoardLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, TensorBoardLogger):
                return logger

    raise Exception(
        "You are using wandb related callback, but TensorboardLogger was not found for some reason..."
    )

class WatchModel(Callback):
    """Make Tensorboard watch model at each epoch."""

    def on_train_start(self, trainer, pl_module):
        logger = get_tensorboard_logger(trainer=trainer)
        experiment = logger.experiment
        for name, param in trainer.model.named_parameters():
            experiment.add_histogram(f"{name}", param.clone().cpu().detach().numpy(), 0)

    def on_train_epoch_end(self, trainer, pl_module):
        logger = get_tensorboard_logger(trainer=trainer)
        experiment = logger.experiment
        for name, param in trainer.model.named_parameters():
            experiment.add_histogram(f"{name}", param.clone().cpu().detach().numpy(), trainer.current_epoch+1)
        

class LogConfusionMatrix(Callback):
    """Generate confusion matrix every epoch and send it to wandb.
    Expects validation step to return predictions and targets.
    """

    def __init__(self):
        self.preds = []
        self.targets = []
        self.ready = True

    def on_sanity_check_start(self, trainer, pl_module) -> None:
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Gather data from single batch."""
        if self.ready:
            self.preds.append(outputs["preds"])
            self.targets.append(outputs["targets"])

    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate confusion matrix."""
        if self.ready:
            logger = get_tensorboard_logger(trainer)
            experiment = logger.experiment

            preds = torch.cat(self.preds).cpu().numpy().argmax(axis=1)
            targets = torch.cat(self.targets).cpu().numpy()

            confusion_matrix = metrics.confusion_matrix(y_true=targets, y_pred=preds)

            # set figure size
            plt.figure(figsize=(14, 8))

            # set labels size
            sn.set(font_scale=1.4)

            # set font size
            sn.heatmap(confusion_matrix, annot=True, annot_kws={"size": 8}, fmt="g")

            # names should be unique or else charts from different experiments in wandb will overlap
            experiment.add_figure(f'{trainer.current_epoch}', plt.gcf(), trainer.current_epoch)

            # according to wandb docs this should also work but it crashes
            # experiment.log(f{"confusion_matrix/{experiment.name}": plt})

            # reset plot
            plt.clf()

            self.preds.clear()
            self.targets.clear()

            
class LogF1PrecRecHeatmap(Callback):
    """Generate f1, precision, recall heatmap every epoch and send it to wandb.
    Expects validation step to return predictions and targets.
    """

    def __init__(self, class_names: List[str] = None):
        self.preds = []
        self.targets = []
        self.ready = True

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Gather data from single batch."""
        if self.ready:
            self.preds.append(outputs["preds"])
            self.targets.append(outputs["targets"])

    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate f1, precision and recall heatmap."""
        if self.ready:
            logger = get_tensorboard_logger(trainer=trainer)
            experiment = logger.experiment

            preds = torch.cat(self.preds).cpu().numpy().argmax(axis=1)
            targets = torch.cat(self.targets).cpu().numpy()
            f1 = f1_score(targets, preds, average=None)
            r = recall_score(targets, preds, average=None)
            p = precision_score(targets, preds, average=None)
            data = [f1, p, r]

            # set figure size
            plt.figure(figsize=(14, 3))

            # set labels size
            sn.set(font_scale=1.2)

            # set font size
            sn.heatmap(
                data,
                annot=True,
                annot_kws={"size": 10},
                fmt=".3f",
                yticklabels=["F1", "Precision", "Recall"],
            )

            # names should be unique or else charts from different experiments in wandb will overlap
            experiment.add_figure(f'f1_p_r_heatmap/{trainer.current_epoch}', plt.gcf(), trainer.current_epoch)

            # reset plot
            plt.clf()

            self.preds.clear()
            self.targets.clear()