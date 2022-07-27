from typing import List
import matplotlib.pyplot as plt
import seaborn as sn
import torch
import wandb
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LoggerCollection, WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score

from src.utils import plot_confusion_matrix_from_data


def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    """Safely get Weights&Biases logger from Trainer."""
    """
    if trainer.fast_dev_run:
        raise Exception(
            "Cannot use wandb callbacks since pytorch lightning disables loggers in `fast_dev_run=true` mode."
        )
    """
    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    raise Exception(
        "You are using wandb related callback, but WandbLogger was not found for some reason..."
    )


class WatchModel(Callback):
    """Make wandb watch model at the beginning of the run."""

    def __init__(self, log: str = "all", log_freq: int = 1):
        """
        log: gradients, parameters, all, None
        log_freq: log every N batches
        """

        self.log = log
        self.log_freq = log_freq

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        logger.watch(
            model=trainer.model, log=self.log, log_freq=self.log_freq, log_graph=False
        )


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
            logger = get_wandb_logger(trainer)
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
            # experiment.log( { f"{trainer.test_site_prefix}confusion_matrix/{trainer.current_epoch}": wandb.Image( plt) }, commit=False,)
            # experiment.log({f"{trainer.test_site_prefix}confusion_matrix/{experiment.name}": wandb.Image(plt)}, commit=False)
            experiment.log(
                {
                    f"{trainer.test_site_prefix}confusion_matrix/{trainer.current_epoch}": wandb.Image(
                        plot_confusion_matrix_from_data(
                            y_test=targets,
                            predictions=preds,
                            columns=list(range(max(targets) + 1)),
                            cmap=sn.cubehelix_palette(as_cmap=True),
                            fz=12,
                        ),
                    ),
                }
            )

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
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment

            preds = torch.cat(self.preds).cpu().numpy().argmax(axis=1)
            targets = torch.cat(self.targets).cpu().numpy()
            f1 = f1_score(targets, preds, average=None, zero_division=0)
            r = recall_score(targets, preds, average=None, zero_division=0)
            p = precision_score(targets, preds, average=None, zero_division=0)
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
            experiment.log(
                {
                    f"{trainer.test_site_prefix}f1_p_r_heatmap/{trainer.current_epoch}": wandb.Image(
                        plt
                    )
                },
                commit=False,
            )
            # experiment.log({f"{trainer.test_site_prefix}f1_p_r_heatmap/{experiment.name}": wandb.Image(plt)}, commit=False)

            # reset plot
            plt.clf()

            self.preds.clear()
            self.targets.clear()
