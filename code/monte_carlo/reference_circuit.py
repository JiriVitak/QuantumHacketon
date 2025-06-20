#!/usr/bin/env python
# coding: utf-8

from IBM_runtime_setup import *

import qiskit
import numpy as np
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator


# Evolution matrix creation

# In[6]:


def householder_unitary(psi: np.ndarray) -> np.ndarray:
    """
    Construct a real unitary matrix U such that U @ |0> = psi.

    Parameters:
        psi (np.ndarray): Real normalized state vector.

    Returns:
        U (np.ndarray): Real unitary matrix.
    """
    psi = psi.astype(np.float64)
    psi = psi / np.linalg.norm(psi)

    dim = len(psi)
    e1 = np.zeros(dim)
    e1[0] = 1.0

    v = psi - e1
    v = v / np.linalg.norm(v)

    U = np.eye(dim) - 2.0 * np.outer(v, v)
    return U

def print_matrix(U: np.ndarray, precision: int = 4):
    """
    Pretty-print a matrix with fixed precision.

    Parameters:
        U (np.ndarray): Matrix to print.
        precision (int): Decimal digits.
    """
    np.set_printoptions(precision=precision, suppress=True)
    print(U)

# Example probability distribution (must sum to 1)
p = np.array([1, 0, 0, 0, 0, 0, 0, 1]) / np.sqrt(2)
p /= np.sum(p)

# Target quantum state
psi = np.sqrt(p)

# Generate unitary matrix
U = householder_unitary(psi)

# Print the result
#print_matrix(U)


# Reference circuit creation
#
# - create circuit using qiskit function: UnitaryGate - not optimal, but good enough as a starting point

# In[4]:


def create_reference_circuit(matrix):
    # get the number of qubits from the matrix size
    num_qubits = int(np.log2(len(matrix)))

    # put matrix into a UnitaryGate
    gate = UnitaryGate(matrix)

    # create circuit
    qc = QuantumCircuit(num_qubits)
    qc.append(gate, qc.qubits)

    compiled_qc = pm.run(qc)
    circuit_depth = compiled_qc.depth()
    print(f"Circuit depth: {circuit_depth}")

    return compiled_qc

def reference_circuit_on_simulator(matrix):
    num_qubits = int(np.log2(len(matrix)))

    # put matrix into a UnitaryGate
    gate = UnitaryGate(matrix)

    # create circuit
    qc = QuantumCircuit(num_qubits)
    qc.append(gate, qc.qubits)
    qc.measure_all()

    print(qc)
    shots = 1000
    simulator = AerSimulator(method='statevector')
    job = simulator.run(qc, shots=shots)
    result = job.result()
    counts = result.get_counts()
    print(f"Counts: {counts}")
    return counts


# Test

# In[5]:


#GHZ = np.array([1, 0, 0, 0, 0, 0, 0, 1]) / np.sqrt(2)
#GHZ_evolve_matrix = householder_unitary(GHZ)
#qc_ghz = create_reference_circuit(GHZ_evolve_matrix)
#print(qc_ghz)
#qc_ghz.decompose().draw('mpl', scale=0.1)

#qc_ghz_sim = reference_circuit_on_simulator(GHZ_evolve_matrix)


# In[ ]:




