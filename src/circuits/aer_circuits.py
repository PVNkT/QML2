import qiskit

#aer simulator에 사용되는 여러 종류의 양자 회로를 저장해놓음
#모두 rotation y gate를 사용해서 입력값을 회로에 적용한다.

class model_1:
    """얽힘이 없는 회로"""
    def __init__(self, n_qubits, theta):
        self.n_qubit = n_qubits
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        self.all_qubits = [i for i in range(n_qubits)]
        self.theta = theta
        self._circuit.h(self.all_qubits)
        
        self._circuit.barrier()
        
        for theta, qubit in zip(self.theta, self.all_qubits):
            self._circuit.ry(theta, qubit)

        self._circuit.barrier()

        self._circuit.measure_all()
    def get_circuit(self):
        return self._circuit

class model_2:
    """qubit 1, 2가 bell state를 이루는 회로"""
    def __init__(self, n_qubits, theta):
        self.n_qubit = n_qubits
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        self.all_qubits = [i for i in range(n_qubits)]
        self.theta = theta

        deleted = self.all_qubits.copy()
        del deleted[1]

        self._circuit.h(deleted)
        self._circuit.cx(2, 1)

        self._circuit.barrier()

        for theta, qubit in zip(self.theta, self.all_qubits):
            self._circuit.ry(theta, qubit)

        self._circuit.barrier()

        self._circuit.measure_all()
    def get_circuit(self):
        return self._circuit

class model_3:
    """얽힘이 적용되고  RY gate가 적용된 뒤 다른 qubit들과 CNOT gate로 연결된 회로"""
    def __init__(self, n_qubits, theta):
        self.n_qubit = n_qubits
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        self.all_qubits = [i for i in range(n_qubits)]
        self.theta = theta

        deleted = self.all_qubits.copy()
        del deleted[1]

        self._circuit.h(deleted)      
        self._circuit.cx(2, 1)

        self._circuit.barrier()

        for theta, qubit in zip(self.theta, self.all_qubits):
            self._circuit.ry(theta, qubit)

        self._circuit.barrier()

        self._circuit.cx(1, 0)
        self._circuit.cx(1, 3)

        self._circuit.measure_all()
    def get_circuit(self):
        return self._circuit

class model_4:
    """2개의 Bell state를 만들고 RY gate를 적용한 뒤 두 얽힘 상태를 cnot gate로 연결함"""
    def __init__(self, n_qubits, theta):
        self.n_qubit = n_qubits
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        self.all_qubits = [i for i in range(n_qubits)]
        self.theta = theta

        deleted = self.all_qubits.copy()
        del deleted[0]
        del deleted[3]

        self._circuit.h(deleted)
        
        self._circuit.cx(1, 0)
        self._circuit.cx(2, 3)

        self._circuit.barrier()

        for theta, qubit in zip(self.theta, self.all_qubits):
            self._circuit.ry(theta, qubit)

        self._circuit.cx(1, 0)
        self._circuit.cx(1, 3)

        self._circuit.measure_all()
    def get_circuit(self):
        return self._circuit

class model_5:
    """Bell state를 만들고 RY gate 이전에 다른 qubit들과 연결함"""
    def __init__(self, n_qubits, theta):
        self.n_qubit = n_qubits
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        self.all_qubits = [i for i in range(n_qubits)]
        self.theta = theta

        deleted = self.all_qubits.copy()
        del deleted[2]

        self._circuit.h(deleted)
        # self._circuit.h(0)
        
        
        self._circuit.cx(1, 2)
        self._circuit.cx(0, 1)
        self._circuit.cx(2, 3)

        self._circuit.barrier()

        for theta, qubit in zip(self.theta, self.all_qubits):
            self._circuit.ry(theta, qubit)

        self._circuit.measure_all()
    def get_circuit(self):
        return self._circuit

class model_6:
    """Bell state를 만들고 RY gate 이전에 다른 qubit들과 연결함"""
    def __init__(self, n_qubits, theta):
        self.n_qubit = n_qubits
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        self.all_qubits = [i for i in range(n_qubits)]
        self.theta = theta

        deleted = self.all_qubits.copy()
        del deleted[0]

        self._circuit.h(deleted)
        
        self._circuit.cx(1, 0)
        self._circuit.cx(0, 2)
        self._circuit.cx(0, 3)

        self._circuit.barrier()

        for theta, qubit in zip(self.theta, self.all_qubits):
            self._circuit.ry(theta, qubit)

        self._circuit.measure_all()
    def get_circuit(self):
        return self._circuit

class model_7:
    """model_5의 뒤에 cnot gate를 추가로 사용함"""
    def __init__(self, n_qubits, theta):
        self.n_qubit = n_qubits
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        self.all_qubits = [i for i in range(n_qubits)]
        self.theta = theta

        deleted = self.all_qubits.copy()
        del deleted[2]

        self._circuit.h(deleted)
        self._circuit.cx(1, 2)
        self._circuit.cx(0, 1)
        self._circuit.cx(2, 3)

        self._circuit.barrier()

        for theta, qubit in zip(self.theta, self.all_qubits):
            self._circuit.ry(theta, qubit)

        self._circuit.barrier()

        self._circuit.cx(1, 0)
        self._circuit.cx(1, 3)

        self._circuit.measure_all()
    def get_circuit(self):
        return self._circuit

class model_8:
    """model_6의 뒤에 cnot gate를 추가로 연결함"""
    def __init__(self, n_qubits, theta):
        self.n_qubit = n_qubits
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        self.all_qubits = [i for i in range(n_qubits)]
        self.theta = theta

        deleted = self.all_qubits.copy()
        del deleted[0]

        self._circuit.h(deleted)
        
        self._circuit.cx(1, 0)
        self._circuit.cx(0, 2)
        self._circuit.cx(0, 3)

        self._circuit.barrier()

        for theta, qubit in zip(self.theta, self.all_qubits):
            self._circuit.ry(theta, qubit)

        self._circuit.barrier()

        self._circuit.cx(1, 0)
        self._circuit.cx(1, 3)
        self._circuit.measure_all()
    def get_circuit(self):
        return self._circuit


