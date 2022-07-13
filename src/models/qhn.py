import qiskit
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Dict


from src.layers import Hybrid, Aer_Hybrid


class Simple_QHN(nn.Module):
    def __init__(self, params: Optional[Dict] = None, *args, **kwargs) -> None:
        super(Simple_QHN, self).__init__()
        self.params = params.Simple_QHN
        self.no_quantum = self.params.no_quantum
        self.model = self.params.model
        self.simulation = self.params.backend.simulation
        self.lstm_hidden = self.params.lstm_hidden
        self.conv1 = nn.Conv1d(116, 6, kernel_size=5)
        self.conv2 = nn.Conv1d(6, 16, kernel_size=5)
        self.dropout = nn.Dropout2d()
        self.lstm = nn.LSTM(
            input_size=16,
            hidden_size=self.lstm_hidden,
            num_layers=1,
            bidirectional=True,
        )
        self.fc1 = nn.Linear(256, self.params.linear_out)
        self.fc2 = nn.Linear(self.params.linear_out, self.params.n_qubits)
        if self.params.backend.backend[0:3] == "aer":
            self.hybrid = Aer_Hybrid(
            model = self.model,
            n_qubits = self.params.n_qubits,
            backend = self.params.backend,
            shots = self.params.shots,
            shift = self.params.shift, 
            )
        else:
            self.hybrid = Hybrid
        self.fc3 = nn.Linear(2**self.params.n_qubits, 2)
        self.fc4 = nn.Linear(self.params.n_qubits, 2)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        x = torch.flatten(x, 2)
        x, (h, c) = self.lstm(x.permute(0, 2, 1))
        x = torch.concat(
            [x[:, -1, : self.lstm_hidden], x[:, 0, self.lstm_hidden :]], axis=1
        )
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        if self.no_quantum:
            x = self.fc4(x)
        else:
            x = torch.tanh(x) * torch.ones_like(x) * torch.tensor(np.pi / 2)
            if self.params.backend.backend[0:3] == "aer":
                x = self.hybrid(x).to(self.device)
                
            else: 
                x = self.hybrid(input = x,
                    model = self.params.model, 
                    n_qubits = self.params.n_qubits,
                    backend = self.params.backend,
                    shots = self.params.shots,
                    shift = self.params.shift,).forward().to(self.device)
            x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x


class MNIST_QHN(nn.Module):
    def __init__(self, params: Optional[Dict] = None, *args, **kwargs) -> None:
        super(MNIST_QHN, self).__init__()
        self.params = params.MNIST_QHN
        self.no_quantum = self.params.no_quantum
        self.model = self.params.model
        self.simulation = self.params.backend.simulation
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(256, self.params.linear_out)
        self.fc2 = nn.Linear(self.params.linear_out, self.params.n_qubits)
        if self.params.backend.backend[0:3] == "aer":
            self.hybrid = Aer_Hybrid(
            model = self.model,
            n_qubit = self.params.n_qubits,
            backend = self.params.backend,
            shots = self.params.shots,
            shift = self.params.shift, 
            )
        else:
            self.hybrid = Hybrid
        self.fc3 = nn.Linear(2**self.params.n_qubits, 10)
        self.fc4 = nn.Linear(self.params.n_qubits, 2)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        x = x.view(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.tanh(x) * torch.ones_like(x) * torch.tensor(np.pi / 2)
        if self.no_quantum:
            x = self.fc4(x)
        else:
            x = torch.tanh(x) * torch.ones_like(x) * torch.tensor(np.pi / 2)
            if self.simulation:
                x = self.hybrid(x).to(self.device)
   
            else: 
                x = self.hybrid(input = x,
                    model = self.params.model, 
                    n_qubits = self.params.n_qubits,
                    backend = self.params.backend,
                    shots = self.params.shots,
                    shift = self.params.shift,).forward().to(self.device)
            x = self.fc3(x)
        x = F.softmax(self.fc3(x), dim=1)
        return x

if __name__ == "__main__":
    backend = "aer_simulator"
    print(backend[0:2])
