import qiskit
from qiskit import transpile, assemble
import torch
from torch import nn
from torch.autograd import Function
import numpy as np
from qiskit import IBMQ
from qiskit.providers.aer import AerSimulator
from qiskit.providers.ibmq import least_busy
from qiskit.tools import job_monitor
from src.circuits import real_circuits, aer_circuits


class QuantumCircuit:
    """
    양자 회로를 설정하고 입력받은 변수를 바탕으로 양자 회로를 시행하고 결과를 반환한다.
    aer 시뮬레이터가 아닌 실제 양자 컴퓨터에 접속하거나 실제 양자 컴퓨터에 대한 시뮬레이션으로 작업을 진행할 때 사용되는 class이다.
    """

    def __init__(self, model, n_qubits, backend, shots, thetas):
        #IBMQ에서 원하는 provider를 불러옴
        self.provider = IBMQ.get_provider(hub='ibm-q-skku', group='hanyang-uni', project='hu-students')
        #시뮬레이션 여부 저장
        self.simulation = backend.simulation
        #대기열이 가장 적은 backend로 시행하는 코드
        if backend.backend[0:5] == "least":
            print("finding least busy \r")
            #가능한 backend를 모두 불러옴
            available_backends = self.provider.backends(n_qubits > n_qubits, operational=True, simulator=False)
            pending = []
            fast_backend = False
            #모든 backend에 대해서 상태를 확인
            for a_backend in available_backends:
                try:
                    #예약중이거나 수리중인 backend를 제외
                    if a_backend.status().status_msg != "active":
                        pending.append(1000)
                    else:
                        #작업이 없는 backend가 있을 경우 loop를 멈추고 그 backend로 실행
                        if a_backend.status().pending_jobs < 1:
                            self.backend = a_backend
                            fast_backend = True
                            break
                        else:
                            #리스트에 대기열의 작업수를 입력
                            pending.append(a_backend.status().pending_jobs)
                except:
                    #backend의 상태를 불러올 수 없는 경우 대기열을 1000으로 입력해서 사용하지 않음 
                    print(f"{a_backend} state load error")
                    pending.append(1000)
            #사용할 backend 출력
            if fast_backend:
                print("run in fast backend: ",f"{self.backend}")
            else:
                pending = np.array(pending)

                self.backend = available_backends[np.argmin(pending)]
                print("least busy backend is ", self.backend)
            """
            #IBMQ에서 제공하는 leastbusy함수, 오류를 줄이기 위해서 다시 작성함
            self.backend =least_busy(provider.backends(n_qubits > n_qubits, operational=True, simulator=False))
            """
        else:
            #직접 backend를 설정한 경우 그 backend를 사용한다.
            self.backend = self.provider.get_backend(backend.backend)
        #시뮬레이션을 사용한 경우에 그 backend에 맞는 시뮬레이션을 만든다. (GPU 시뮬레이션 사용)
        if self.simulation:
            self.backend = AerSimulator(device="GPU").from_backend(self.backend)
        else:    
            pass
        #입력된 값들을 저장한다.
        self.shots = shots
        self.thetas = thetas
        self.n_qubit = n_qubits
        #주어진 양자 회로를 불러온다.
        self.circuits = getattr(real_circuits, model)(n_qubits, thetas).get_circuit()
        
    def run(self):
        """
        #aer simulator를 사용하는 경우에는 다음과 같이 parameter bind를 사용해서 변수를 입력할 수 있었는데 
        #실제 양자 컴퓨터에 입력할 때 이러한 방식이 불가능해서 각 회로에 직접 입력값을 주는 방식으로 바꾸었다.
        t_qc = transpile(self._circuit, self.backend)
        qobj = assemble(
            t_qc,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        """
        #batch 크기에 맞는 양자 회로들의 list를 받는다.
        circuit_list = self.circuits
        #결과들의 이름 (크게 중요하지 않음)
        output_name = [f"{i}" for i in range(len(self.thetas))]
        #양자 회로를 각 backend에 맞게 transpile, assemble한다.
        t_qc = transpile(circuit_list, self.backend,output_name=output_name)
        qobj = assemble(t_qc, shots=self.shots, backend = self.backend)
        #오류가 났을 경우 자동으로 다시 시작하게 하는 코드
        run = True
        while run:
            try:
                job = self.backend.run(qobj)
                if self.simulation:
                    run = False
                else:
                    job_monitor(job)
                    job.wait_for_final_state()
                    if str(self.provider.backend.jobs()[0].status())[10:] == "DONE":
                        run = False
                    else:
                        run = True
            except:
                print("error")
                run = True
        #결과를 저장
        results = job.result().get_counts()

        #counts = np.array(list(result.values()))
        #states = np.array(list(result.keys())).astype(float)
        
        #결과로 얻은 값들을 상태 벡터의 형태로 바꾸기 위한 코드
        possible_states = []
        #가능한 상태들을 모두 string의 형태로 순서대로 저장한다.
        for s in range(2**(self.n_qubit)):
            possible_states.append(format(s, "b").zfill(self.n_qubit))
        
        states = []
        #각 실험의 결과를 상태의 순서대로 저장한다. 
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

#양자 회로와 고전적인 신경망을 이어주기 위한 class
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

        """
        #aer simulator의 경우 다음의 코드를 사용
        expectation_z = ctx.quantum_circuit.run(input[0].tolist())
        result = torch.tensor([expectation_z])
        """
        #결과를 pytorch의 텐서로 저장한다.
        result = torch.Tensor(ctx.quantum_circuit(model, n_qubits, backend, shots, thetas = input).run())
        #backpropagation을 위해서 입력값과 결과를 저장한다.
        ctx.save_for_backward(input, result)
        
        return result

    @staticmethod
    def backward(ctx, grad_output):
        #GPU 사용 설정
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        """Backward pass computation"""
        #저장된 값을 불러옴
        input, expectation_z = ctx.saved_tensors
        input_list = np.array(input.tolist())
        #gradient를 계산하기 위해 입력값을 shift만큼 더하고 뺀 값을 계산(양자 회로를 시행)한다.
        shift_right = input_list + np.ones(input_list.shape) * ctx.shift
        expectation_right = ctx.quantum_circuit(ctx.model, ctx.n_qubits, ctx.backend, ctx.shots, thetas =shift_right).run()
                
        shift_left = input_list - np.ones(input_list.shape) * ctx.shift.shift      
        expectation_left = ctx.quantum_circuit(ctx.model, ctx.n_qubits, ctx.backend, ctx.shots, thetas =shift_left).run()

        #계산된 두 값의 차이를 통해서 gradient를 계산한다.
        gradient = torch.tensor(expectation_right) - torch.tensor(expectation_left)
        
        #가능한 상태들을 행렬의 형태가 되도록 저장한다.
        possible_states = []
        for s in range(2**(len(input[0]))):
            possible_states.append(np.array(list(format(s, "b").zfill(len(input[0]))),np.float64))
        possible_states=np.array(possible_states)
        #결과 값에 gradient를 곱한다.
        grad = gradient * grad_output
        #위에서 만든 행렬을 곱해 입력값의 크기와 같게 맞추고 반환한다.
        grad = torch.matmul(grad.type(torch.FloatTensor), torch.FloatTensor(possible_states)).squeeze()
        return None, None, None, grad.to(device), None, None, None

#하이브리드 함수를 적용해주는 class
class Hybrid(nn.Module):
    """Hybrid quantum - classical layer definition"""

    def __init__(
        self, input, model, n_qubits, backend, shots, shift):
        super(Hybrid, self).__init__()
        self.backend = backend
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


#aer 시뮬레이터를 사용하는 경우
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
        #양자 회로를 transpile, assemble하는 코드
        t_qc = transpile(self._circuit, self.backend)
        qobj = assemble(
            t_qc,
            shots=self.shots,
            parameter_binds=[{self.theta[i]: thetas[i] for i in range(len(thetas))}],
        )
        #시뮬레이션을 시행
        job = self.backend.run(qobj)
        #결과를 받는다.
        result = job.result().get_counts()

        # counts = np.array(list(result.values()))
        # states = np.array(list(result.keys())).astype(float)
        
        #가능한 상태들을 모두 string의 형태로 순서대로 저장한다.
        possible_states = []
        for s in range(2**(self.n_qubit)):
            possible_states.append(format(s, "b").zfill(self.n_qubit))

        #상태를 순서대로 정렬하여 상태 벡터의 형태로 만든다.
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

        #batch수만큼 양자 회로를 돌리고 결과를 tensor의 형태로 저장한다.
        expectation_z = [
            torch.Tensor(ctx.quantum_circuit.run(input[i].tolist()))
            for i in range(input.size(0))
        ]
        #결과를 적절한 형태로 변형한다.
        result = torch.stack(expectation_z, axis=0)
        #backpropagation을 위해서 저장
        ctx.save_for_backward(input, result)
        
        return result

    @staticmethod
    def backward(ctx, grad_output):
        #gpu사용 설정
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #Backward pass computation
        input, expectation_z = ctx.saved_tensors
        input_list = np.array(input.tolist())
        #gradient를 계산하기 위해서 shift만큼 더하고 빼진 입력값을 계산한다.
        shift_right = input_list + np.ones(input_list.shape) * ctx.shift
        shift_left = input_list - np.ones(input_list.shape) * ctx.shift
        gradients = []
        #batch크기만큼 시행을 반복
        for i in range(len(input_list)):
            #shift만큼 벗어난 두 값에 대한 시뮬레이션을 시행하고 결과를 빼서 gradient를 계산한다.
            expectation_right = ctx.quantum_circuit.run(shift_right[i])
            expectation_left = ctx.quantum_circuit.run(shift_left[i])

            gradient = torch.tensor(np.array([expectation_right])) - torch.tensor(
                np.array([expectation_left])
            )
            gradients.append(gradient)
        #얻어진 gradient들을 batch형태에 맞게 정렬한다.
        gradients = torch.concat(gradients, axis=0)
        #가능한 상태들로 이루어진 행렬을 만든다.
        possible_states = []
        for s in range(2**(len(input[0]))):
            possible_states.append(np.array(list(format(s, "b").zfill(len(input[0]))),np.float64))
        possible_states=np.array(possible_states)
        #얻어진 결과에 gradient를 곱한다.
        grad = gradients * grad_output
        #행렬 곱을 통해서 입력값에 맞는 gradient값을 계산한다.
        grad = torch.matmul(grad.type(torch.FloatTensor), torch.FloatTensor(possible_states))
        
        return grad.to(device), None, None, None

#하이브리드 함수를 적용해주는 class
class Aer_Hybrid(nn.Module):
    #Hybrid quantum - classical layer definition

    def __init__(
        self, model, n_qubits, backend, shots, shift):
        super(Aer_Hybrid, self).__init__()
        self.model = model
        self.backend = backend.backend
        self.backend = AerSimulator(device='GPU')
        self.quantum_circuit = Aer_QuantumCircuit(
            self.model,n_qubits, self.backend, shots
        )
        self.shift = shift

    def forward(self, input):
        return Aer_HybridFunction.apply(input, self.model, self.quantum_circuit, self.shift)

if __name__ == "__main__":
    print(Hybrid(input = torch.Tensor([[1,2,3,4],[5,6,7,8]]).cuda(),backend={"backend":"ibmq_lima", "simulation": True},model="model_7", n_qubits = 4, shots=200, shift=0.2).forward())


