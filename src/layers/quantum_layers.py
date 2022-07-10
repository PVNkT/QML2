import qiskit
from qiskit import transpile, assemble
import torch
from torch import nn
from torch.autograd import Function
import numpy as np
from qiskit import IBMQ

from src.circuits import real_circuits, aer_circuits

class QuantumCircuit:
    """
    This class provides a simple interface for interaction
    with the quantum circuit
    """

    def __init__(self, model, n_qubits, backend, shots, thetas):
        self.backend = backend
        self.shots = shots
        self.thetas = thetas
        self.n_qubit = n_qubits
        self.circuits = getattr(real_circuits, model)(n_qubits, thetas).get_circuit()
        
    def run(self):
        
        #t_qc = transpile(self._circuit, self.backend)
        # qobj = assemble(
        #     t_qc,
        #     shots=self.shots,
        #     parameter_binds=[{self.theta: theta} for theta in thetas],
        # )
        # --- multi qubit -- #
        circuit_list = self.circuits
        output_name = [f"i" for i in range(len(self.thetas))]
        
        t_qc = transpile(circuit_list, self.backend,output_name=output_name)
        qobj = assemble(t_qc, shots=self.shots, backend = self.backend)
        job = self.backend.run(qobj)
        results = job.result().get_counts()

        #counts = np.array(list(result.values()))
        # states = np.array(list(result.keys())).astype(float)
        # --- multi qubit -- #
        possible_states = []
        for s in range(2**(self.n_qubit)):
            possible_states.append(format(s, "b").zfill(self.n_qubit))
        states = []
        for result in results:
            state = []
            for i in possible_states:
                try:
                    state.append(result[i])
                except:
                    state.append(0)   
            state = np.array(state, dtype=np.float64)
            states.append(state)
        states = np.array(states) 
        return states/self.shots#기댓값을 출력


class HybridFunction(Function):
    """Hybrid quantum - classical function definition"""

    @staticmethod
    def forward(ctx, n_qubits, backend, shots, input, quantum_circuit, shift, model):
        """Forward pass computation"""
        ctx.shift = shift
        ctx.n_qubits = n_qubits
        ctx.backend = backend
        ctx.shots = shots
        ctx.model = model
        ctx.quantum_circuit = quantum_circuit

        # expectation_z = ctx.quantum_circuit.run(input[0].tolist())
        # result = torch.tensor([expectation_z])
        # -- multi qubit -- #
        result = torch.Tensor(ctx.quantum_circuit(model, n_qubits, backend, shots, thetas = input).run())
        
        #result = torch.stack(expectation_z, axis=0)
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
        
        expectation_right = ctx.quantum_circuit(ctx.model, ctx.n_qubits, ctx.backend, ctx.shots, thetas =shift_right).run()
        expectation_left = ctx.quantum_circuit(ctx.model, ctx.n_qubits, ctx.backend, ctx.shots, thetas =shift_left).run()

        gradient = torch.tensor(np.array([expectation_right])) - torch.tensor(np.array([expectation_left]))


        possible_states = []
        for s in range(2**(len(input[0]))):
            possible_states.append(np.array(list(format(s, "b").zfill(len(input[0]))),np.float64))
        possible_states=np.array(possible_states)
        
        grad = gradient * grad_output
        grad = torch.matmul(grad.type(torch.FloatTensor), torch.FloatTensor(possible_states)).squeeze()
        return None, None, None, grad.to(device), None, None, None


class Hybrid(nn.Module):
    """Hybrid quantum - classical layer definition"""

    def __init__(
        self, input, model, n_qubits=4, backend="aer_simulator", shots=100, shift=0.6, 
    ):
        super(Hybrid, self).__init__()
        provider = IBMQ.get_provider(hub='ibm-q-skku', group='hanyang-uni', project='hu-students')
        self.backend = provider.get_backend(backend)
        self.quantum_circuit = QuantumCircuit
        self.shift = shift
        self.input = input
        self.n_qubits = n_qubits
        self.shots = shots
        self.model = model
    def forward(self):
        return HybridFunction.apply(
            self.n_qubits, 
            self.backend, 
            self.shots, 
            self.input, 
            self.quantum_circuit, 
            self.shift, 
            self.model)



class Aer_QuantumCircuit:
    
    #This class provides a simple interface for interaction with the quantum circuit
    

    def __init__(self, model, n_qubits, backend, shots):
        # --- Circuit definition ---
        self.backend = backend
        self.shots = shots
        self.n_qubit = n_qubits
        self.all_qubits = [i for i in range(n_qubits)]
        self.theta = [qiskit.circuit.Parameter(f"theta_{i}") for i in self.all_qubits]
        self._circuit = getattr(aer_circuits, model)(n_qubits, self.theta).get_circuit()

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


class Aer_HybridFunction(Function):
    #Hybrid quantum - classical function definition

    @staticmethod
    def forward(ctx, input, model, Aer_quantum_circuit, shift):
        #Forward pass computation
        ctx.shift = shift
        ctx.quantum_circuit = Aer_quantum_circuit
        ctx.model = model
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
        
        return grad.to(device), None, None, None


class Aer_Hybrid(nn.Module):
    #Hybrid quantum - classical layer definition

    def __init__(
        self, model, n_qubits=2, backend="aer_simulator", shots=100, shift=0.6
    ):
        super(Aer_Hybrid, self).__init__()
        self.model = model
        self.quantum_circuit = Aer_QuantumCircuit(
            self.model,n_qubits, qiskit.Aer.get_backend(backend), shots
        )
        self.shift = shift

    def forward(self, input):
        return Aer_HybridFunction.apply(input, self.model, self.quantum_circuit, self.shift)

if __name__ == "__main__":
    print(Hybrid(input = torch.Tensor([[1,2,3,4],[5,6,7,8]]).cuda(),backend="ibmq_lima").forward())


