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
            circuit.h(self.all_qubits)

            circuit.barrier()

            for j in range(len(theta)):
                circuit.ry(float(theta[j]), j)

            circuit.barrier()
            circuit.measure_all()
            self.circuits.append(circuit)

    def get_circuit(self):
        return self.circuits

class model_2:
    def __init__(self, n_qubits, thetas):
        self.all_qubits = [i for i in range(n_qubits)]
        deleted = self.all_qubits.copy()
        self.circuits = []
        del deleted[1]
        self.thetas = thetas

        for theta in thetas:
        # --- Circuit definition ---
            circuit = qiskit.QuantumCircuit(n_qubits)
            circuit.h(deleted)
            circuit.cx(2, 1)

            circuit.barrier()

            for j in range(len(theta)):
                circuit.ry(float(theta[j]), j)

            circuit.barrier()

            circuit.measure_all()
            self.circuits.append(circuit)

    def get_circuit(self):
        return self.circuits

class model_3:
    def __init__(self, n_qubits, thetas):
        self.all_qubits = [i for i in range(n_qubits)]
        deleted = self.all_qubits.copy()
        self.circuits = []
        del deleted[1]
        self.thetas = thetas

        for theta in thetas:
        # --- Circuit definition ---
            circuit = qiskit.QuantumCircuit(n_qubits)
            circuit.h(deleted)
            circuit.cx(2, 1)

            circuit.barrier()

            for j in range(len(theta)):
                circuit.ry(float(theta[j]), j)

            circuit.barrier()

            circuit.cx(1, 0)
            circuit.cx(1, 3)

            circuit.barrier()

            circuit.measure_all()
            self.circuits.append(circuit)

    def get_circuit(self):
        return self.circuits

class model_4:
    def __init__(self, n_qubits, thetas):
        self.all_qubits = [i for i in range(n_qubits)]
        deleted = self.all_qubits.copy()
        self.circuits = []
        del deleted[0]
        del deleted[3]
        self.thetas = thetas

        for theta in thetas:
        # --- Circuit definition ---
            circuit = qiskit.QuantumCircuit(n_qubits)
            circuit.h(deleted)
            circuit.cx(1, 0)
            circuit.cx(2, 3)

            circuit.barrier()

            for j in range(len(theta)):
                circuit.ry(float(theta[j]), j)

            circuit.barrier()

            for i in [0,3]:
                circuit.cx(1,i)

            circuit.barrier()

            circuit.measure_all()
            self.circuits.append(circuit)

    def get_circuit(self):
        return self.circuits

class model_5:
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

            for j in range(len(theta)):
                circuit.ry(float(theta[j]), j)

            circuit.barrier()

            circuit.measure_all()
            self.circuits.append(circuit)

    def get_circuit(self):
        return self.circuits

class model_6:
    def __init__(self, n_qubits, thetas):
        self.all_qubits = [i for i in range(n_qubits)]
        deleted = self.all_qubits.copy()
        self.circuits = []
        del deleted[0]
        self.thetas = thetas

        for theta in thetas:
        # --- Circuit definition ---
            circuit = qiskit.QuantumCircuit(n_qubits)
            circuit.h(deleted)
            circuit.cx(1, 0)
            circuit.cx(0, 2)
            circuit.cx(2, 3)

            circuit.barrier()

            for j in range(len(theta)):
                circuit.ry(float(theta[j]), j)

            circuit.barrier()

            circuit.measure_all()
            self.circuits.append(circuit)

    def get_circuit(self):
        return self.circuits

class model_7:
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

            for j in range(len(theta)):
                circuit.ry(float(theta[j]), j)
            
            circuit.barrier()

            for i in [0,3]:
                circuit.cx(1,i)

            circuit.barrier()

            circuit.measure_all()
            self.circuits.append(circuit)

    def get_circuit(self):
        return self.circuits

class model_8:
    def __init__(self, n_qubits, thetas):
        self.all_qubits = [i for i in range(n_qubits)]
        deleted = self.all_qubits.copy()
        self.circuits = []
        del deleted[0]
        self.thetas = thetas

        for theta in thetas:
        # --- Circuit definition ---
            circuit = qiskit.QuantumCircuit(n_qubits)
            circuit.h(deleted)
            circuit.cx(1, 0)
            circuit.cx(0, 2)
            circuit.cx(0, 3)
            
            circuit.barrier()

            for j in range(len(theta)):
                circuit.ry(float(theta[j]), j)
            
            circuit.barrier()

            for i in [0,3]:
                circuit.cx(1,i)
            
            circuit.barrier()

            circuit.measure_all()
            self.circuits.append(circuit)

    def get_circuit(self):
        return self.circuits


    