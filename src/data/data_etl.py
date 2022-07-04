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

DATA_DIR = f"Data/nitrc_niak"
SITE_INDEX = [1, 3, 4, 5, 6]
SITES = ["Peking", "KKI", "NI", "NYU", "OHSU"]
SITES_DICT = {idx: site for idx, site in zip(SITE_INDEX, SITES)}


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


@dataclass
class Load:
    df: pd.DataFrame = pd.read_csv(Path(DATA_DIR) / "master_df.csv")

    def loadSiteData(self, site: Union[List[int], int] = SITE_INDEX):
        if isinstance(site, int):
            site = [site]

        df = self.df[self.df['Site'].isin(site)]
        data = [np.load(p) for p in df["filePath"].values]
        labels = df["DX"].values

        return data, labels


def chunks(lst, n):
    div = int(np.ceil(len(lst) / n))
    return [lst[i : i + div] for i in range(0, len(lst), div)]





