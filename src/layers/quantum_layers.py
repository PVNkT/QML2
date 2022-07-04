import qiskit
from qiskit import transpile, assemble
import torch
from torch import nn
from torch.autograd import Function
import numpy as np


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

        for i in range(2, self.n_qubit):
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


class HybridFunction(Function):
    """Hybrid quantum - classical function definition"""

    @staticmethod
    def forward(ctx, input, quantum_circuit, shift):
        """Forward pass computation"""
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
        """Backward pass computation"""
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
    """Hybrid quantum - classical layer definition"""

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
