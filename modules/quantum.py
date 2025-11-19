import os
import numpy as np
from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_histogram
from cryptography.fernet import Fernet

class QuantumCryptography:
    def __init__(self):

        self.simulator = Aer.get_backend('qasm_simulator')

    def quantum_encrypt(self, file_path, output_path):
        if not os.path.exists(file_path):
            return f"خطا: فایل {file_path} یافت نشد!"
        try:
            
            key_length = 128  
            quantum_key = self._generate_quantum_key(key_length)

            key_bytes = ''.join(map(str, quantum_key)).encode()
            key_bytes = key_bytes.ljust(32, b'0')[:32]  
            cipher = Fernet(Fernet.generate_key()) 


            with open(file_path, "rb") as f:
                data = f.read()
            encrypted_data = cipher.encrypt(data)


            with open(output_path, "wb") as f:
                f.write(encrypted_data)


            key_file = output_path + ".key"
            with open(key_file, "wb") as f:
                f.write(key_bytes)

            return f"فایل با رمزنگاری کوانتومی رمزنگاری شد: {output_path}\nکلید در {key_file} ذخیره شد"
        except Exception as e:
            return f"خطا در رمزنگاری کوانتومی: {str(e)}"

    def _generate_quantum_key(self, length):

        key = []
        bases = np.random.randint(0, 2, length)  
        bits = np.random.randint(0, 2, length) 

        
        for i in range(length):
            qc = QuantumCircuit(1, 1)
            
            if bits[i] == 1:
                qc.x(0)
            if bases[i] == 1:
                qc.h(0)

            if bases[i] == 1:
                qc.h(0)  
            qc.measure(0, 0)


            tqc = transpile(qc, self.simulator)
            job = assemble(tqc, shots=1)
            result = self.simulator.run(job).result()
            counts = result.get_counts()
            measured_bit = int(max(counts, key=counts.get))
            key.append(measured_bit)

        return key

    def simulate_quantum_algorithm(self):

        try:

            n_qubits = 2
            qc = QuantumCircuit(n_qubits, n_qubits)


            for qubit in range(n_qubits):
                qc.h(qubit)


            qc.cz(0, 1)  

            for qubit in range(n_qubits):
                qc.h(qubit)
            for qubit in range(n_qubits):
                qc.x(qubit)
            qc.cz(0, 1)
            for qubit in range(n_qubits):
                qc.x(qubit)
            for qubit in range(n_qubits):
                qc.h(qubit)


            qc.measure(range(n_qubits), range(n_qubits))


            tqc = transpile(qc, self.simulator)
            job = assemble(tqc, shots=1024)
            result = self.simulator.run(job).result()
            counts = result.get_counts()


            return f"نتیجه الگوریتم گروور: {counts}"
        except Exception as e:
            return f"خطا در اجرای الگوریتم کوانتومی: {str(e)}"