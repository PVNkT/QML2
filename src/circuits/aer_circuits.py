import qiskit

class model_1:
    def __init__(self, n_qubits, theta):
        self.n_qubit = n_qubits
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        self.all_qubits = [i for i in range(n_qubits)]
        self.theta = theta

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

        for i in range(2,self.n_qubit):
            self._circuit.cx(0,i)

        self._circuit.measure_all()
    def get_circuit(self):
        return self._circuit



