from dataclasses import dataclass
from typing import Optional, Dict, Any

from src.tasks import ClassificationTask
from src.data import ROIDataset

from torch.utils.data import Dataset

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import LightningModule, LightningDataModule, Trainer

#기본적인 runner를 정의하는 class로 다른 runner들이 이 class를 상속받아 사용함
@dataclass
class Base_Runner:
    log: Dict
    optimizer: Dict
    loader: Dict
    network: Dict
    data: Dict

    #기본적인 모델을 만들어 반환하는 함수
    def get_network(
        self, Task: LightningModule = ClassificationTask, *args: Any, **kwargs: Any
    ) -> LightningModule:
        model = Task(self.optimizer, self.network, *args, **kwargs)
        return model

    #데이터 모듈에 데이터를 넣어서 반환해주는 함수
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

    #데이터 모듈과 모델을 통해서 실제 훈련을 진행하는 부분
    def run(self, profiler: Optional[str] = False):
        dm = self.get_datamodule(self, ROIDataset)
        model = self.get_network()
        #pytorch lightning의 trainer함수를 통해서 훈련에 사용될 설정들을 입력함
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
        #model을 학습시킴
        trainer.fit(model, datamodule=dm)
        #model을 테스트함
        trainer.test(model, datamodule=dm)
        return trainer.callback_metrics
