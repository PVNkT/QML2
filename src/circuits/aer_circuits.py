import qiskit

class model_1:
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


