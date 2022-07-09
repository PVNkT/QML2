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


class QuantumCircuit:
    """
    This class provides a simple interface for interaction
    with the quantum circuit
    """

    def __init__(self, n_qubits, backend, shots):
        # --- Circuit definition ---
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        self.n_qubit = n_qubits
        self.all_qubits = [i for i in range(n_qubits)]
        # self.theta = qiskit.circuit.Parameter("theta")
        # --- multi qubit ---#
        self.theta = [qiskit.circuit.Parameter(f"theta_{i}") for i in self.all_qubits]

        deleted = self.all_qubits.copy()
        del deleted[1]

        self._circuit.h(deleted)
        # self._circuit.h(0)
        
        
        self._circuit.cx(0, 1)
        # self._circuit.ry(self.theta, all_qubits)
        self._circuit.barrier()
        # --- multi qubit ---#
        for theta, qubit in zip(self.theta, self.all_qubits):
            self._circuit.ry(theta, qubit)

        for i in [2,3,4,5]:
            self._circuit.cx(0,i)

        self._circuit.measure_all()
        # ---------------------------

        self.backend = backend
        self.shots = shots

    def run(self, thetas):
        t_qc = transpile(self._circuit, self.backend)
        # qobj = assemble(
        #     t_qc,
        #     shots=self.shots,
        #     parameter_binds=[{self.theta: theta} for theta in thetas],
        # )
        # --- multi qubit -- #
        qobj = assemble(
            t_qc,
            shots=self.shots,
            parameter_binds=[{self.theta[i]: thetas[i] for i in range(len(thetas))}],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        counts = np.array(list(result.values()))
        # states = np.array(list(result.keys())).astype(float)
        # --- multi qubit -- #
        possible_states = []
        for s in range(2**(self.n_qubit)):
            possible_states.append(format(s, "b").zfill(self.n_qubit))
       
        states = []
        for i in possible_states:
            try:
                states.append(result[i])
            except:
                states.append(0)   
        states = np.array(states, dtype=np.float64)
         
        return states/self.shots#기댓값을 출력

"""
class HybridFunction(Function):
    #Hybrid quantum - classical function definition

    @staticmethod
    def forward(ctx, input, quantum_circuit, shift):
        #Forward pass computation
        ctx.shift = shift
        ctx.quantum_circuit = quantum_circuit

        # expectation_z = ctx.quantum_circuit.run(input[0].tolist())
        # result = torch.tensor([expectation_z])
        # -- multi qubit -- #
        expectation_z = [
            torch.Tensor(ctx.quantum_circuit.run(input[i].tolist()))
            for i in range(input.size(0))
        ]
        result = torch.stack(expectation_z, axis=0)
        ctx.save_for_backward(input, result)
        
        return result

    @staticmethod
    def backward(ctx, grad_output):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #Backward pass computation
        input, expectation_z = ctx.saved_tensors
        input_list = np.array(input.tolist())

        shift_right = input_list + np.ones(input_list.shape) * ctx.shift
        shift_left = input_list - np.ones(input_list.shape) * ctx.shift

        gradients = []
        for i in range(len(input_list)):
            expectation_right = ctx.quantum_circuit.run(shift_right[i])
            expectation_left = ctx.quantum_circuit.run(shift_left[i])

            gradient = torch.tensor(np.array([expectation_right])) - torch.tensor(
                np.array([expectation_left])
            )
            gradients.append(gradient)
        gradients = torch.concat(gradients, axis=0)

        possible_states = []
        for s in range(2**(len(input[0]))):
            possible_states.append(np.array(list(format(s, "b").zfill(len(input[0]))),np.float64))
        possible_states=np.array(possible_states)
        
        grad = gradients * grad_output
        grad = torch.matmul(grad.type(torch.FloatTensor), torch.FloatTensor(possible_states))
        
        return grad.to(device), None, None


class Hybrid(nn.Module):
    #Hybrid quantum - classical layer definition

    def __init__(
        self, n_qubits=2, backend="aer_simulator", shots=100, shift=0.6
    ):
        super(Hybrid, self).__init__()
        self.quantum_circuit = QuantumCircuit(
            n_qubits, qiskit.Aer.get_backend(backend), shots
        )
        self.shift = shift

    def forward(self, input):
        return HybridFunction.apply(input, self.quantum_circuit, self.shift)

"""

