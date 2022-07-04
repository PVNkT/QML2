import qiskit
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Dict


from src.layers import Hybrid


class Simple_QHN(nn.Module):
    def __init__(self, params: Optional[Dict] = None, *args, **kwargs) -> None:
        super(Simple_QHN, self).__init__()
        params = params.Simple_QHN
        self.lstm_hidden = params.lstm_hidden
        self.conv1 = nn.Conv1d(116, 6, kernel_size=5)
        self.conv2 = nn.Conv1d(6, 16, kernel_size=5)
        self.dropout = nn.Dropout2d()
        self.lstm = nn.LSTM(
            input_size=16,
            hidden_size=self.lstm_hidden,
            num_layers=1,
            bidirectional=True,
        )
        self.fc1 = nn.Linear(256, params.linear_out)
        self.fc2 = nn.Linear(params.linear_out, params.n_qubits)
        self.hybrid = Hybrid(
            params.n_qubits,
            params.backend,
            100,
            shift=params.shift,
            
        )
        self.fc3 = nn.Linear(2**params.n_qubits, 2)
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
        x = torch.tanh(x) * torch.ones_like(x) * torch.tensor(np.pi / 2)
        x = self.hybrid(x).to(self.device)
        x = F.softmax(self.fc3(x), dim=1)
        return x

