import qiskit


class model_1:
    def __init__(self, n_qubits, thetas):
        self.all_qubits = [i for i in range(n_qubits)]
        deleted = self.all_qubits.copy()
        self.circuits = []
        del deleted[2]
        self.thetas = thetas

        for theta in thetas:
        # --- Circuit definition ---
            circuit = qiskit.QuantumCircuit(n_qubits)
            circuit.h(deleted)
                   
            circuit.cx(1, 2)
            circuit.cx(0, 1)
            circuit.cx(2, 3)
            circuit.barrier()
            # --- multi qubit ---#
            for j in range(len(theta)):
                circuit.ry(float(theta[j]), j)

            for i in [0,3]:
                circuit.cx(1,i)

            circuit.measure_all()
            self.circuits.append(circuit)

    def get_circuit(self):
        return self.circuits

    