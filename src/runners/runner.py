import os
import pickle
import pandas as pd
from glob import glob
from typing import Optional
from copy import deepcopy

from src.runners import Base_Runner
from src.data import ROIDataset, SITES_DICT, MNISTDataset, LOSODataset
from src.datamodules import DataModule, MNISTDataModule, LOSODataModule
from src.tasks import ClassificationTask
from src.utils import plot_paper
from src.callbacks import wandb_callback as wbc
from src.callbacks import tensorboard_callback as tbc

import torch
from torch import nn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from pytorch_lightning import seed_everything

#같은 실험 환경 유지를 위해서 seed를 고정
seed_everything(41)

#신경망에서 사용되는 weight값을 초기화해주는 함수 kaiming_normal_과 xavier_normal_함수를 사용함
def initialize_weights(m):
    if isinstance(m, nn.Conv1d):
        # nn.init.xavier_normal_(m.weight.data)
        nn.init.kaiming_normal_(m.weight.data)
    elif isinstance(m, nn.Linear):
        # nn.init.xavier_normal_(m.weight.data)
        nn.init.kaiming_normal_(m.weight.data)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            mul = param.shape[0] // 4
            for idx in range(4):
                if "bias" in name:
                    nn.init.constant_(param, 0.00)
                elif "weight_ih" in name:
                    nn.init.xavier_normal_(param.data[idx*mul:(idx+1)*mul])
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param.data[idx*mul:(idx+1)*mul])

#각 사이트의 데이터를 훈련 데이터와 테스트 데이터로 나누어 학습을 시키는 경우
class S_Runner(Base_Runner):
    def get_callbacks(self, site: str):
        """
        Write only callbacks that logger is not necessary
        """
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(
                self.log.checkpoint_path,
                self.log.project_name,
                f"version{self.version:03d}",
                site,
            ),
            filename=os.path.join(f"model"),
            monitor=f"{site.upper()}/Accuracy/val",
            mode="max",
            verbose=False,
            save_top_k=1,
        )

        callbacks = dict(
            filter(lambda item: item[0].endswith("callback"), vars().items())
        ).values()
        callbacks = list(callbacks)
        return callbacks if len(callbacks) > 0 else None
    #실제 작동하는 부분
    def run(self, profiler: Optional[str] = None):
        #checkpoint를 저장할 장소를 만듬
        os.makedirs(
            os.path.join(self.log.checkpoint_path, self.log.project_name), exist_ok=True
        )
        #만들어진 경로들의 수로 version을 결정
        self.version = len(
            os.listdir(os.path.join(self.log.checkpoint_path, self.log.project_name))
        )

        #사용할 ROI를 고르는 과정, QML에서는 전체를 사용하는 경우만 사용
        # TODO: extract to function
        if self.data.get("roi", None) is None:
            self.data.roi = list(range(116))
        else:
            if "roi_rank" in self.log.project_name:
                with open("Data/nitrc_niak/roi_rank.pkl", "rb") as f:
                    self.data.roi = pickle.load(f)[: int(self.data.roi)]

            else:
                self.data.roi = [int(self.data.roi)]
        self.network.roi_rank = self.data.roi
        print("ROI = {}".format(self.data.roi))

        # # nyu, kki, peking, ohsu, ni
        # path = glob(os.path.join(self.data.path, "*.pickle"))
        # # nyu, peking, ohsu, kki, ni
        # path[1:4] = path[2], path[3], path[1]
        # SITES = ["Peking", "KKI", "NI", "NYU", "OHSU"]

        final_results = list()
        #각 사이트들의 index
        site_index = [5, 1, 6, 3, 4]
        #각 사이트들에 대해서 훈련을 진행
        for i in site_index:
            train_site = deepcopy(list(SITES_DICT.keys()))
            print(train_site, type(train_site))
            #site_index와 train_site의 관계를 표현
            site_dict = {5:3, 1:0, 6:4, 3:1, 4:2}
            #train하는 사이트의 이름을 저장
            self.data.train_site = train_site[site_dict[i]]
            site_str = SITES_DICT[i]
            #데이터를 불러옴
            dm = self.get_datamodule(dataset=ROIDataset, datamodule=DataModule)
            model = self.get_network(Task=ClassificationTask)
            model.apply(initialize_weights)
            model.prefix = site_str.upper() + "/"
            
            #pytorch lightning의 trainer함수를 통해서 훈련에 사용될 설정들을 입력함
            trainer = Trainer(
                #훈련 log를 저장하는 부분, TensorBoard와 wandb를 사용함
                logger=[
                    TensorBoardLogger(
                        save_dir=self.log.log_path,
                        name=os.path.join(
                            self.log.project_name,
                            f"version{self.version:03d}",
                            site_str,
                        ),
                        default_hp_metric=False,
                        version=None,
                        # log_graph=True, # inavailable due to bug
                    ),
                    WandbLogger(
                        project=self.log.project_name,
                        save_dir=self.log.log_path,
                    ),
                ],
                # ! use all gpu
                # gpus=-1,
                # auto_select_gpus=True,
                # ! use 2 gpu
                # devices=2,
                # accelerator="auto",
                # strategy="ddp",
                # ! use gpu 0
                devices=[0],
                accelerator="gpu",
                #devices=[self.log.device.gpu],
                #accelerator="cpu",
                check_val_every_n_epoch=self.log.val_log_freq_epoch,
                log_every_n_steps=1,
                num_sanity_val_steps=0,
                max_epochs=self.log.epoch,
                profiler=profiler,
                fast_dev_run=self.log.dry_run,
                callbacks=[
                    *self.get_callbacks(site=site_str),
                    wbc.WatchModel(),
                    wbc.LogConfusionMatrix(),
                    wbc.LogF1PrecRecHeatmap(),
                    # tbc.WatchModel(),
                    # tbc.LogConfusionMatrix(),
                    # tbc.LogF1PrecRecHeatmap(),
                ],
                precision=self.log.precision,
                # gradient_clip_val=0.5,
            )
            trainer.test_site_prefix = model.prefix
            #model을 학습시킴
            trainer.fit(model, datamodule=dm)
            #model을 테스트 함
            trainer.test(model, datamodule=dm, ckpt_path="best")
            final_results.append(
                trainer.callback_metrics[f"{model.prefix}Accuracy/test"]
            )
        return

#한 사이트를 테스트 데이터로 사용하고 나머지 사이트의 데이터를 훈련 데이터로 사용하는 경우 
class LOSO_Runner(Base_Runner):
    def get_callbacks(self, site: str):
        """
        Write only callbacks that logger is not necessary
        """
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(
                self.log.checkpoint_path,
                self.log.project_name,
                f"version{self.version:03d}",
                site,
            ),
            filename=os.path.join(f"model"),
            monitor=f"{site.upper()}/Accuracy/val",
            mode="max",
            verbose=False,
            save_top_k=1,
        )

        callbacks = dict(
            filter(lambda item: item[0].endswith("callback"), vars().items())
        ).values()
        callbacks = list(callbacks)
        return callbacks if len(callbacks) > 0 else None
    #실제 작동하는 부분
    def run(self, profiler: Optional[str] = None):
        #checkpoint를 저장할 장소를 만듬
        os.makedirs(
            os.path.join(self.log.checkpoint_path, self.log.project_name), exist_ok=True
        )
        #만들어진 경로들의 수로 version을 결정
        self.version = len(
            os.listdir(os.path.join(self.log.checkpoint_path, self.log.project_name))
        )
        #사용할 ROI를 고르는 과정, QML에서는 전체를 사용하는 경우만 사용
        # TODO: extract to function
        if self.data.get("roi", None) is None:
            self.data.roi = list(range(116))
        else:
            if "roi_rank" in self.log.project_name:
                with open("Data/nitrc_niak/roi_rank.pkl", "rb") as f:
                    self.data.roi = pickle.load(f)[: int(self.data.roi)]

            else:
                self.data.roi = [int(self.data.roi)]
        self.network.roi_rank = self.data.roi
        print("ROI = {}".format(self.data.roi))

        # # nyu, kki, peking, ohsu, ni
        # path = glob(os.path.join(self.data.path, "*.pickle"))
        # # nyu, peking, ohsu, kki, ni
        # path[1:4] = path[2], path[3], path[1]
        # SITES = ["Peking", "KKI", "NI", "NYU", "OHSU"]

        final_results = list()
        site_list = list()
        new_SITES_DICT = dict([(value, key) for key, value in SITES_DICT.items()])
        for site in self.data.site:
            site_list.append(new_SITES_DICT[site])
        
        for i in site_list:
        #각 사이트 들에 대해서 훈련을 진행
            #사이트들을 불러옴
            train_site = deepcopy(list(SITES_DICT.keys()))
            #해당되는 사이트를 test 사이트로 분리함
            test_site = train_site.pop(train_site.index(i))
            #test 사이트와 train 사이트를 저장함
            self.data.train_site = train_site
            self.data.test_site = test_site
            site_str = SITES_DICT[i]

            #데이터를 불러옴
            dm = self.get_datamodule(dataset=LOSODataset, datamodule=LOSODataModule)
            #모델을 불러옴
            model = self.get_network(Task=ClassificationTask)
            #weight초기화
            model.apply(initialize_weights)
            model.prefix = site_str.upper() + "/"
            #pytorch lightning의 trainer함수를 통해서 훈련에 사용될 설정들을 입력함
            trainer = Trainer(
                #훈련 log를 저장하는 부분, TensorBoard와 wandb를 사용함
                logger=[
                    TensorBoardLogger(
                        save_dir=self.log.log_path,
                        name=os.path.join(
                            self.log.project_name,
                            f"version{self.version:03d}",
                            site_str,
                        ),
                        default_hp_metric=False,
                        version=None,
                        # log_graph=True, # inavailable due to bug
                    ),
                    WandbLogger(
                        project=self.log.project_name,
                        save_dir=self.log.log_path,
                    ),
                ],
                # ! use all gpu
                # gpus=-1,
                # auto_select_gpus=True,
                # ! use 2 gpu
                # devices=2,
                # accelerator="auto",
                # strategy="ddp",
                # ! use gpu 0
                devices=[0],
                accelerator="gpu",
                #devices=[self.log.device.gpu],
                #accelerator="cpu",
                check_val_every_n_epoch=self.log.val_log_freq_epoch,
                log_every_n_steps=1,
                num_sanity_val_steps=0,
                max_epochs=self.log.epoch,
                profiler=profiler,
                fast_dev_run=self.log.dry_run,
                callbacks=[
                    *self.get_callbacks(site=site_str),
                    wbc.WatchModel(),
                    wbc.LogConfusionMatrix(),
                    wbc.LogF1PrecRecHeatmap(),
                    # tbc.WatchModel(),
                    # tbc.LogConfusionMatrix(),
                    # tbc.LogF1PrecRecHeatmap(),
                ],
                precision=self.log.precision,
                # gradient_clip_val=0.5,
            )
            trainer.test_site_prefix = model.prefix
            #model을 학습시킴
            trainer.fit(model, datamodule=dm)
            #model을 테스트함
            trainer.test(model, datamodule=dm, ckpt_path="best")
            final_results.append(
                trainer.callback_metrics[f"{model.prefix}Accuracy/test"]
            )

        #사이트들에 대한 훈련이 끝나고 전체 사이트에 대한 이미지를 업로드
        try:
            import wandb

            wb_logger = wbc.get_wandb_logger(trainer)
            wb_logger.experiment.log(
                {"overall_accuracy": torch.Tensor(final_results).mean().item()}
            )
            wb_logger.experiment.log(
                {
                    "overall_accuracy_image": wandb.Image(
                        plot_paper(
                            results=final_results,
                            path=os.path.join(
                                self.log.log_path,
                                self.log.project_name,
                                f"version{self.version:03d}",
                                "Accuracy.png",
                            ),
                        )
                    )
                }
            )
        except Exception as e:
            print(e)

        return

class MNIST_Runner(Base_Runner):
    def get_callbacks(self):
        """
        Write only callbacks that logger is not necessary
        """
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(
                self.log.checkpoint_path,
                self.log.project_name,
                f"version{self.version:03d}",
            ),
            filename=os.path.join(f"model"),
            monitor=f"Accuracy/val",
            mode="max",
            verbose=False,
            save_top_k=1,
        )

        callbacks = dict(
            filter(lambda item: item[0].endswith("callback"), vars().items())
        ).values()
        callbacks = list(callbacks)
        return callbacks if len(callbacks) > 0 else None

    def run(self, profiler: Optional[str] = None):
        os.makedirs(
            os.path.join(self.log.checkpoint_path, self.log.project_name), exist_ok=True
        )
        self.version = len(
            os.listdir(os.path.join(self.log.checkpoint_path, self.log.project_name))
        )


        final_results = list()



        dm = self.get_datamodule(dataset=MNISTDataset, datamodule=MNISTDataModule)
        model = self.get_network(Task=ClassificationTask)
        model.apply(initialize_weights)

        trainer = Trainer(
            logger=[
                TensorBoardLogger(
                    save_dir=self.log.log_path,
                    name=os.path.join(
                        self.log.project_name,
                        f"version{self.version:03d}"
                    ),
                    default_hp_metric=False,
                    version=None,
                    # log_graph=True, # inavailable due to bug
                ),
                WandbLogger(
                    project=self.log.project_name,
                    save_dir=self.log.log_path,
                ),
            ],
            # ! use all gpu
            # gpus=-1,
            # auto_select_gpus=True,
            # ! use 2 gpu
            # devices=2,
            # accelerator="auto",
            # strategy="ddp",
            # ! use gpu 0
            devices=[0],
            accelerator="gpu",
            #devices=[self.log.device.gpu],
            #accelerator="cpu",
            check_val_every_n_epoch=self.log.val_log_freq_epoch,
            log_every_n_steps=1,
            num_sanity_val_steps=0,
            max_epochs=self.log.epoch,
            profiler=profiler,
            callbacks=[
                *self.get_callbacks(),
                wbc.WatchModel(),
                wbc.LogConfusionMatrix(),
                wbc.LogF1PrecRecHeatmap(),
                # tbc.WatchModel(),
                # tbc.LogConfusionMatrix(),
                # tbc.LogF1PrecRecHeatmap(),
            ],
            precision=self.log.precision,
            # gradient_clip_val=0.5,
        )
        trainer.test_site_prefix = model.prefix
        trainer.fit(model, datamodule=dm)
        trainer.test(model, datamodule=dm, ckpt_path="best")
        final_results.append(
            trainer.callback_metrics[f"Accuracy/test"]
        )

    try:
        import wandb

        wb_logger = wbc.get_wandb_logger(trainer=Trainer)
        
    except Exception as e:
        print(e)
    
