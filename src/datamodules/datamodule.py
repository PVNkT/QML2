from copy import deepcopy
import numpy as np
from typing import Any, Optional, List, Dict
from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from pytorch_lightning import LightningDataModule
from src.data import collate_fn, SamplerFactory


@dataclass
class DataModule(LightningDataModule):
    def __init__(self,data: Dict, loader: Dict, dataset: Dict):
        super().__init__()
        self.prepare_data_per_node = True
        self.data = data
        self.loader = loader
        self.dataset = dataset

    def setup(self, stage: Optional[str] = None):
        if stage in ("fit", None):
            self.train_dataset = self.dataset(is_train = True, site = self.data.train_site)
            self.val_dataset = self.dataset(is_train = False, site = self.data.train_site)

        if stage in ("test", None):
            self.test_dataset = self.dataset(is_train = False, site = self.data.train_site)

    def train_dataloader(self):
        conf = deepcopy(self.loader.train)
        batch_size = conf.pop("batch_size")
        conf.shuffle = False
        # return DataLoader(self.train_dataset, **conf, collate_fn=collate_fn)

        return DataLoader(
            self.train_dataset,
            **conf,
            collate_fn=collate_fn,
            batch_sampler=SamplerFactory().get(
                class_idxs=[
                    np.where(self.train_dataset.labels == i)[0].tolist()
                    for i in range(2)
                ],
                batch_size=batch_size,
                n_batches=len(self.train_dataset)//batch_size + 5,
                alpha=1.0,
                kind="fixed",
            ),
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.loader.eval, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.loader.eval, collate_fn=collate_fn)

@dataclass
class LOSODataModule(LightningDataModule):
    def __init__(self,data: Dict, loader: Dict, dataset: Dict):
        super().__init__()
        self.prepare_data_per_node = True
        self.data = data
        self.loader = loader
        self.dataset = dataset

    def setup(self, stage: Optional[str] = None):
        if stage in ("fit", None):
            self.train_dataset = self.dataset(self.data.train_site, False)
            self.val_dataset = self.dataset(self.data.test_site, self.data.same_size)

        if stage in ("test", None):
            self.test_dataset = self.dataset(self.data.test_site, self.data.same_size)

    def train_dataloader(self):
        conf = deepcopy(self.loader.train)
        batch_size = conf.pop("batch_size")
        conf.shuffle = False
        # return DataLoader(self.train_dataset, **conf, collate_fn=collate_fn)

        return DataLoader(
            self.train_dataset,
            **conf,
            collate_fn=collate_fn,
            batch_sampler=SamplerFactory().get(
                class_idxs=[
                    np.where(self.train_dataset.labels == i)[0].tolist()
                    for i in range(2)
                ],
                batch_size=batch_size,
                n_batches=len(self.train_dataset)//batch_size + 5,
                alpha=1.0,
                kind="fixed",
            ),
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.loader.eval, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.loader.eval, collate_fn=collate_fn)


@dataclass
class MNISTDataModule(LightningDataModule):
    
    def __init__(self,data: Dict,loader: Dict,dataset: Dict):
        super().__init__()
        self.prepare_data_per_node = True
        self.data = data
        self.loader = loader
        self.dataset = dataset
    def setup(self, stage: Optional[str] = None):
        if stage in ("fit", None):
            self.train_dataset = self.dataset(n_samples = self.loader.samples.train_sample, is_train=True)
            self.val_dataset = self.dataset(n_samples = self.loader.samples.eval_sample, is_train=False)

        if stage in ("test", None):
            self.test_dataset = self.dataset(n_samples = self.loader.samples.eval_sample, is_train=False)
    
    def train_dataloader(self):
        conf = deepcopy(self.loader.train)
        batch_size = conf.pop("batch_size")
        conf.shuffle = False
        # return DataLoader(self.train_dataset, **conf, collate_fn=collate_fn)

        return DataLoader(
            self.train_dataset,
            **conf,
            collate_fn=collate_fn,
            batch_sampler=SamplerFactory().get(
                class_idxs=[
                    np.where(self.train_dataset.labels == i)[0].tolist()
                    for i in range(10)
                ],
                batch_size=batch_size,
                n_batches=len(self.train_dataset)//batch_size + 5,
                alpha=1.0,
                kind="random",
            ),
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.loader.eval, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.loader.eval, collate_fn=collate_fn)