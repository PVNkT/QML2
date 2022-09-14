from xmlrpc.client import Boolean
import numpy as np
from typing import Union, List

import torch
from torch.utils.data import Dataset

from src.data import Load, SITES_DICT, Load_MNIST, LOSOLoad

#Pytorch에서 사용하는 형식에 dataset 설정, 한 사이트에서 train과 test를 나누어서 데이터를 load해옴  
class ROIDataset(Dataset):
    def __init__( self, is_train: bool, site: Union[List, str]) -> None:
        load = Load()
        self.data, self.labels = load.loadSiteData(is_train, site)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int):
        """
        data shape: (116, time series)
        label: not one hot. 0 or 1.
        """
        data = self.data[index]
        label = self.labels[index]
        return data, label

#Pytorch에서 사용하는 형식에 dataset 설정, 한 사이트를 테스트 데이터로 사용하고 나머지를 train 데이터로 사용함
class LOSODataset(Dataset):
    def __init__( self, site: Union[List, str], same_size:Boolean) -> None:
        load = LOSOLoad(same_size = same_size)
        self.data, self.labels = load.loadSiteData(site)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int):
        """
        data shape: (116, time series)
        label: not one hot. 0 or 1.
        """
        data = self.data[index]
        label = self.labels[index]
        return data, label

class MNISTDataset(Dataset):
    def __init__(self, n_samples: int, is_train: Boolean) -> None:
        Load = Load_MNIST(is_train)
        if n_samples >0:
            self.data = Load.get_samples(n_samples=n_samples).data
            self.labels = Load.get_samples(n_samples=n_samples).targets  
        else:  
            self.data = Load.dataset.data
            self.labels = Load.dataset.targets

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index: int):
        data = self.data[index]
        label = self.labels[index]
        return data, label

def pad_tensor(vec: torch.Tensor, pad: int, dim: int) -> torch.Tensor:
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    vec = torch.Tensor(vec)
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)


def collate_fn(batch):
    dim = 1
    xs, ys = list(zip(*batch))
    # find longest sequence
    max_len = max(map(lambda x: x.shape[dim], xs))
    xs = torch.stack(
        list(map(lambda x: pad_tensor(x, pad=max_len, dim=dim), xs)), dim=0
    )
    ys = torch.LongTensor(ys)
    return (xs, ys)




if __name__ == "__main__":
    from time import sleep
    from copy import deepcopy

    from torch.utils.data import DataLoader
    from sampling import SamplerFactory

    for i, (key, value) in enumerate(SITES_DICT.items()):
        train_site = deepcopy(list(SITES_DICT.keys()))
        test_site = train_site.pop(i)

        train_dataset = ROIDataset(train_site, roi_rank=[3, 11, 91, 1])
        test_dataset = ROIDataset(test_site)
        print(train_site, test_site)
        print(len(train_dataset), len(test_dataset))

        batch_size = 32
        num_workers = 8
        # for x, y in DataLoader(
        #     train_dataset,
        #     =SamplerFactory().get(
        #         class_idxs=[
        #             np.where(train_dataset.labels == i)[0].tolist() for i in range(2)
        #         ],
        #         batch_size=batch_size,
        #         n_batches=50,
        #         alpha=0.0,
        #         kind="fixed",
        #     ),
        #     num_workers=num_workers,
        #     collate_fn=collate_fn,
        #     pin_memory=True,
        # ):
        #     print(
        #         "{}, {}, {}, {}".format(
        #             x.size(), x.is_pinned(), y.size(), y.is_pinned()
        #         )
        #     )
        #     for i, bins in enumerate(np.bincount(y)):
        #         print(f"{i}: {bins:2d} ", end="")
        #     print()

        for x, y in DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=32,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        ):
            print(
                "{}, {}, {}, {}".format(
                    x.size(), x.is_pinned(), y.size(), y.is_pinned()
                )
            )
            for i, bins in enumerate(np.bincount(y)):
                print(f"{i}: {bins:2d} ", end="")
            print()
