from typing import List
import matplotlib.pyplot as plt
import seaborn as sn
import torch
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LoggerCollection, TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score

#tensorboard logger를 사용하는 경우 tensorboard logger를 내보내는 함수
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

#추적할 parameter들에 대한 histogram을 만들어 업로드한다.
class WatchModel(Callback):
    """Make Tensorboard watch model at each epoch."""
    #학습이 시작될 경우 parameter들에 대한 histogram을 그려서 추가한다.
    def on_train_start(self, trainer, pl_module):
        logger = get_tensorboard_logger(trainer=trainer)
        experiment = logger.experiment
        for name, param in trainer.model.named_parameters():
            experiment.add_histogram(f"{name}", param.clone().cpu().detach().numpy(), 0)

    #각 epoch가 끝날 때마다 parameter들에 대한 histogram을 그려서 추가한다.
    def on_train_epoch_end(self, trainer, pl_module):
        logger = get_tensorboard_logger(trainer=trainer)
        experiment = logger.experiment
        for name, param in trainer.model.named_parameters():
            experiment.add_histogram(f"{name}", param.clone().cpu().detach().numpy(), trainer.current_epoch+1)
        
# confusion matrix를 기록하는 함수
class LogConfusionMatrix(Callback):
    """Generate confusion matrix every epoch and send it to wandb.
    Expects validation step to return predictions and targets.
    """
    #초기값 설정
    def __init__(self):
        self.preds = []
        self.targets = []
        self.ready = True

    #건전성 확인 과정이 시작되면 입력을 작동하지 않는다.
    def on_sanity_check_start(self, trainer, pl_module) -> None:
        self.ready = False

    #건전성 확인 과정이 끝나면 입력을 작동한다.
    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    #validation 과정이 끝나면 데이터를 입력한다.
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Gather data from single batch."""
        #건전성 확인 작업 중이 아닐 경우
        if self.ready:
            #target과 예측한 값을 저장한다.
            self.preds.append(outputs["preds"])
            self.targets.append(outputs["targets"])

    #입력된 데이터로 confusion matrix를 만든다.
    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate confusion matrix."""
        #건전성 확인 작업 중이 아닐 경우
        if self.ready:
            #logger를 불러옴
            logger = get_tensorboard_logger(trainer)
            experiment = logger.experiment
            #batch내에서 예측한 값들을 합치고 cpu로 불러와 numpy array로 바꾸고 저장한다. 예측값은 확률로 나오기 때문에 argmax를 사용해서 높은 확률이 나온 것을 고른다.
            preds = torch.cat(self.preds).cpu().numpy().argmax(axis=1)
            targets = torch.cat(self.targets).cpu().numpy()

            #confusion_matrix를 만든다.
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

            #저장한 데이터들을 삭제한다.
            self.preds.clear()
            self.targets.clear()

#heatmap(precision, recall, f1 score를 표시)을 그려주는 함수            
class LogF1PrecRecHeatmap(Callback):
    """Generate f1, precision, recall heatmap every epoch and send it to wandb.
    Expects validation step to return predictions and targets.
    """
    #초기값 설정
    def __init__(self, class_names: List[str] = None):
        self.preds = []
        self.targets = []
        self.ready = True

    #건전성 확인 과정이 시작되면 입력을 작동하지 않는다.
    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    #건전성 확인 과정이 끝나면 입력을 작동한다.
    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    #validation 과정이 끝나면 데이터를 입력한다.
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Gather data from single batch."""
        #건전성 확인 작업 중이 아닐 경우
        if self.ready:
            #target과 예측한 값을 저장한다.
            self.preds.append(outputs["preds"])
            self.targets.append(outputs["targets"])

    #입력된 데이터로 heatmap을 만든다.
    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate f1, precision and recall heatmap."""
        #건전성 확인 작업 중이 아닐 경우
        if self.ready:
            #logger를 불러옴
            logger = get_tensorboard_logger(trainer=trainer)
            experiment = logger.experiment
            #batch내에서 예측한 값들을 합치고 cpu로 불러와 numpy array로 바꾸고 저장한다. 예측값은 확률로 나오기 때문에 argmax를 사용해서 높은 확률이 나온 것을 고른다.
            preds = torch.cat(self.preds).cpu().numpy().argmax(axis=1)
            targets = torch.cat(self.targets).cpu().numpy()

            #f1 score, recall, precision의 값을 계산하고 저장한다.
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
            experiment.add_figure(f'f1_p_r_heatmap/{trainer.current_epoch}', plt.gcf(), trainer.current_epoch)

            # reset plot
            plt.clf()

            #저장한 데이터들을 삭제한다.
            self.preds.clear()
            self.targets.clear()