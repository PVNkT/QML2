from dataclasses import dataclass
from typing import Optional, Dict, Any

from src.tasks import ClassificationTask
from src.data import ROIDataset

from torch.utils.data import Dataset

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import LightningModule, LightningDataModule, Trainer


@dataclass
class Base_Runner:
    log: Dict
    optimizer: Dict
    loader: Dict
    network: Dict
    data: Dict

    def get_network(
        self, Task: LightningModule = ClassificationTask, *args: Any, **kwargs: Any
    ) -> LightningModule:
        model = Task(self.optimizer, self.network, *args, **kwargs)
        return model

    def get_datamodule(
        self,
        dataset: Dataset,
        datamodule: LightningDataModule,
        *args: Any,
        **kwargs: Any
    ) -> LightningDataModule:
        datamodule = datamodule(self.data, self.loader, dataset, *args, **kwargs)
        return datamodule

    def get_callbacks(self):
        """
        add callbacks here
        """
        callbacks = dict(
            filter(lambda item: item[0].endswith("callback"), vars().items())
        ).values()
        callbacks = list(callbacks)
        return callbacks if len(callbacks) > 0 else None

    def run(self, profiler: Optional[str] = False):
        dm = self.get_datamodule(self, ROIDataset)
        model = self.get_network()

        trainer = Trainer(
            logger=TensorBoardLogger(
                save_dir=self.log.log_path,
                name=self.log.model_name,
                default_hp_metric=True,
                # log_graph=False, # inavailable due to bug
            ),
            # ! use all gpu
            # gpus=-1,
            # auto_select_gpus=True,
            # ! use 2 gpu
            # devices=2,
            # accelerator="auto",
            # strategy="ddp",
            # ! use gpu 0
            # devices=[0],
            # accelerator="gpu",
            devices=self.log.device.gpu,
            accelerator="gpu",
            check_val_every_n_epoch=self.log.val_log_freq_epoch,
            log_every_n_steps=1,
            num_sanity_val_steps=0,
            max_epochs=self.log.epoch,
            profiler=profiler,
            
            callbacks=self.get_callbacks(),
            precision=self.log.precision,
        )

        trainer.fit(model, datamodule=dm)
        trainer.test(model, datamodule=dm)
        return trainer.callback_metrics
