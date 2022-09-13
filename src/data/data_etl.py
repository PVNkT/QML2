from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional, Union
from xmlrpc.client import Boolean
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.utils import Bunch
from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker
from torchvision import transforms
from torchvision import datasets as dataset
from itertools import repeat
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

#데이터의 위치와 기본 정보
DATA_DIR = f"Data/nitrc_niak"
SITE_INDEX = [1, 3, 4, 5, 6]
SITES = ["Peking", "KKI", "NI", "NYU", "OHSU"]
SITES_DICT = {idx: site for idx, site in zip(SITE_INDEX, SITES)}

#경로로부터 주어진 파일을 불러온다.
class Extract:
    def __init__(self, data_dir: str = DATA_DIR):
        data_dir = Path(data_dir)
        table = list(data_dir.glob("*.tsv"))[0]

        self.df = pd.read_csv(table, sep="\t")[lambda x: x["Site"].isin(SITE_INDEX)]
        self.nii_path = list(data_dir.glob("**/*fmri*run1.nii.gz"))

    def recordPathwithID(self, save=False):
        ID_dict = {int(p.name.split("_")[2]): str(p) for p in self.nii_path}
        self.df["filePath"] = self.df["ScanDir ID"].apply(
            lambda x: ID_dict.get(x, np.NaN)
        )
        self.df = self.df[~self.df["filePath"].isna()]
        if save:
            self.df["DX"] = np.where(self.df["DX"] == "0", 0, 1)
            #self.df["filePath"] = self.df["filePath"].str.replace(".nii.gz", ".npy")
            self.df.to_csv(Path(DATA_DIR, "master_df.csv"), index=False)
        return self.df


class Transform:
    def __init__(
        self,
        data_dir: str = DATA_DIR,
        site_index: List[int] = SITE_INDEX,
        atlas: Bunch = datasets.fetch_atlas_aal(version="SPM12"),
        standardize: Union[str, bool] = False,
    ):
        self.path = Path(data_dir)
        self.masker = NiftiLabelsMasker(
            labels_img=atlas.maps,
            standardize=standardize,
            memory="nilearn_cache",
            verbose=0,
        )

    def extractTimeSeries(
        self, filePath: Union[List, str], save_npy=False, pid=None, wid=None
    ):
        if isinstance(filePath, str):
            filePath = [filePath]

        desc = "Extracting Time Series"
        if pid is not None:
            desc = f"pid: {pid}, wid: {wid} | " + desc

        timeSeriesData = list()
        for p in tqdm(filePath, desc=desc):
            if Path(p).exists():
                # print(f"{p} already extracted")
                continue

            # (nROI=116, timeSteps)
            timeSeries = self.masker.fit_transform(p.replace(".npy", ".nii.gz")).T
            timeSeriesData.append(timeSeries)
            if save_npy:
                np.save(p, timeSeries)
        return timeSeriesData

    def do_thread_func(self, paths: List, save_npy: bool, max_workers: int, pid: int):
        thread_list = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for wid, p in enumerate(paths):
                thread_list.append(
                    executor.submit(
                        self.extractTimeSeries,
                        filePath=p,
                        save_npy=save_npy,
                        pid=pid,
                        wid=wid,
                    )
                )
            for execution in concurrent.futures.as_completed(thread_list):
                execution.result()

    def do_process_with_thread_func(
        self, paths: List, save_npy: bool, max_workers: int, pid: int
    ):
        self.do_thread_func(chunks(paths, max_workers), save_npy, max_workers, pid)

    def multiproc_multithread_extractTimeSeries(
        self, paths: List, save_npy: bool, nproc: int = 8, max_workers: int = 1
    ):
        with Pool(processes=nproc) as pool:
            pool.starmap(
                self.do_process_with_thread_func,
                zip(
                    chunks(paths, nproc),
                    repeat(save_npy),
                    repeat(max_workers),
                    range(nproc),
                ),
            )

#master_df 파일에서 각 사이트의 데이터의 저장 위치를 불러와서 데이터와 label을 내보낸다.
@dataclass
class Load:
    df: pd.DataFrame = pd.read_csv(Path(DATA_DIR) / "master_df.csv")

    def loadSiteData(self, is_train: bool, site: Union[List[int], int] = SITE_INDEX):
        if isinstance(site, int):
            site = [site]
        df = self.df[self.df['Site'].isin(site)]
        data = [np.load(p) for p in df["filePath"].values]
        labels = df["DX"].values
        print(site)
        #각 사이트 별로 훈련 데이터와 테스트 데이터를 어느 위치에서 자를 것인지를 나타내는 dictionary
        slice_dict = {5:216, 1: 194, 3:83, 4:48,6:79}
        slice = slice_dict[site[0]]
        #훈련 데이터로 사용되는 경우 앞부분의 데이터를 사용한다.
        if is_train:
            data = data[:slice]
            labels = labels[:slice]
        #test나 validation data로 사용될 경우 뒤 부분의 데이터를 사용한다.
        else:
            data = data[slice:]
            labels = labels[slice:]
        #print("확인 :" ,len(data),len(labels))
        #print("확인 :" ,np.array(data).shape,len(labels))
        return data, labels

#master_df 파일에서 각 사이트의 데이터의 저장 위치를 불러와서 데이터와 label을 내보낸다.
@dataclass
class LOSOLoad:

    def __init__(self, same_size: Boolean = False):
        self.same_size = same_size
        self.df: pd.DataFrame = pd.read_csv(Path(DATA_DIR) / "master_df.csv")
    
    #주어진 사이트의 데이터를 불러옴
    def loadSiteData(self, site: Union[List[int], int] = SITE_INDEX):
        #하나의 사이트가 숫자로 주어졌을 경우 
        if isinstance(site, int):
            site = [site]
        else: pass
        #여러 개의 사이트가 주어진 경우 (train data를 불러오는 경우)    
        if len(site) > 1:
            self.same_size = False
        else: pass
        #해당 사이트에 해당되는 데이터만을 골라 데이터 프레임을 만듬
        df = self.df[self.df['Site'].isin(site)]
        #데이터 프레임에서 각 데이터가 저장된 경로를 불러와서 이를 numpy array로 저장하고 데이터의 리스트를 만든다.
        data = [np.load(p) for p in df["filePath"].values]
        #데이터의 해당되는 label(ADHD환자인지 아닌지)를 불러온다.
        labels = df["DX"].values
        #테스트 데이터에서 각 label을 가지는 데이터의 수가 동일하게 하고 싶은 경우
        if self.same_size:
            #각 label을 확인하고 0인 것과 1인 것을 구별하여 데이터의 index를 저장한다. 
            label_0 = []
            label_1 = []
            for i, label in enumerate(labels):
                if label == 0:
                    label_0.append(i)
                else:
                    label_1.append(i)
            #두 label중 어떤 데이터가 많은지를 비교하고 더 적은 쪽에 맞추어 데이터를 자른다.
            if len(label_0) > len(label_1):
                label_0 = label_0[:len(label_1)]
            else:
                label_1 = label_1[:len(label_0)]
            #잘린 데이터와 label을 저장할 list
            sliced_label = []
            sliced_data = []
            #데이터의 index들을 합치고 순서대로 정렬한다.
            label = label_0 + label_1
            label = sorted(label)
            #잘린 데이터의 index로 원하는 데이터만을 저장한다.
            for index in label:
                sliced_label.append(labels[index])
                sliced_data.append(data[index])
            data = sliced_data
            labels = np.array(sliced_label)
        
        return data, labels

@dataclass
class Load_MNIST:
    def __init__(self, is_train:Boolean):
        if is_train:
            self.dataset = dataset.MNIST(root='mnist_dataset/raw', train=True, download=True,
                         transform=transforms.Compose([transforms.ToTensor()]))
        else:
            self.dataset = dataset.MNIST(root='mnist_dataset/raw', train=False, download=True,
                        transform=transforms.Compose([transforms.ToTensor()]))
                        
    def get_samples(self, n_samples:int):
        idx = []
        for i in range(10):
            idx.append(np.where(self.dataset.targets == i)[0][:n_samples])#각 target에 해당되는 번호를 저장한다. 
        idx = np.array(idx).reshape(-1)
        self.dataset.data = self.dataset.data[idx]
        self.dataset.targets = self.dataset.targets[idx]
        return self.dataset


def chunks(lst, n):
    div = int(np.ceil(len(lst) / n))
    return [lst[i : i + div] for i in range(0, len(lst), div)]

