import numpy as np
from qiskit import qpy
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerOptions
from qiskit_ibm_runtime.options import TwirlingOptions, DynamicalDecouplingOptions, EnvironmentOptions
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2
from qiskit_ibm_runtime import EstimatorV2

runtime_service_paid = QiskitRuntimeService(
    channel='ibm_quantum',
    instance='cvut/general/simulphys',
    token='63c4bc4d05bc301d88a79333e99f7595c30542815998fbc4c0b3e2b12f3d251bf80f29fa43b35fe7fbf9d49816eed489b9f1e36aed117185400098f6b19b93b9'
)

#backend = runtime_service_paid.backend('ibm_fez')
#pm = generate_preset_pass_manager(optimization_level=3, backend=backend)

"""
job_no_mit = runtime_service_paid.job('d1ahw5rn2txg008e4430')
job_result_no_mit = job_no_mit.result()

job_twirling = runtime_service_paid.job('d1ahw6r5z6q0008pedd0')
job_result_twirling = job_twirling.result()

job_dd_XX = runtime_service_paid.job('d1ahw7rn2txg008e4440')    
job_result_dd_XX = job_dd_XX.result()

job_dd_XY4 = runtime_service_paid.job('d1ahw8hmya70008n98m0')
job_result_dd_XY4 = job_dd_XY4.result()
"""

job_no_mit = runtime_service_paid.job('d1ajjfs3grvg008msapg')
job_result_no_mit = job_no_mit.result()

job_twirling = runtime_service_paid.job('d1ajjgtn2txg008e4geg')
job_result_twirling = job_twirling.result()

#job_dd_XX = runtime_service_paid.job('d1ahnfd5z6q0008pe950')    
#job_result_dd_XX = job_dd_XX.result()

#job_dd_XY4 = runtime_service_paid.job('d1ahnh6v3z50008a95yg')
#job_result_dd_XY4 = job_dd_XY4.result()

binary_order = ['000', '001', '010', '011', '100', '101', '110', '111']
#binary_order = ['0000', '0001', '0010', '0011', '0100', '0101', '0110', '0111', '1000', '1001', '1010', '1011', '1100', '1101', '1110', '1111']
#binary_order = ['0', '1']

def compute_probabilities(job_result, binary_order):
    probability_arrays = []
    
    for i in range(len(job_result)):
        counts = job_result[i].data.c0.get_counts()
        #counts = job_result[i].data.meas.get_counts()
        print(f"Counts for job {i}: {counts}")
        sorted_counts = [counts.get(key, 0) for key in binary_order]
        total_counts = 1000
        probabilities = [count / total_counts for count in sorted_counts]
        probability_arrays.append(probabilities)

    return np.array(probability_arrays)

prob_no_mit = compute_probabilities(job_result_no_mit, binary_order)
print("Probabilities without mitigation:", prob_no_mit)

prob_twirling = compute_probabilities(job_result_twirling, binary_order)
print("Probabilities with twirling:", prob_twirling)

#prob_dd_XX = compute_probabilities(job_result_dd_XX, binary_order)
#print("Probabilities with DD XX:", prob_dd_XX)

#prob_dd_XY4 = compute_probabilities(job_result_dd_XY4, binary_order)
#print("Probabilities with DD XY4:", prob_dd_XY4)

NUM_QUBITS = 3  # Adjust based on your circuit
x_vals = (np.arange(2**NUM_QUBITS) + 0.5) / 2**NUM_QUBITS
f_vals = np.sin(np.pi * x_vals)**2

from scipy.integrate import cumulative_trapezoid as cumtrapz
import matplotlib.pyplot as plt

estimated_cumulative = np.cumsum(prob_twirling)
true_cumulative = cumtrapz(f_vals, x_vals, initial=0)
# Plot cumulative integrals
plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
plt.plot(x_vals, estimated_cumulative, label="Quantum Estimated Cumulative")
plt.plot(x_vals, true_cumulative, '--', label="True Cumulative Integral")
plt.xlabel("x")
plt.ylabel("Cumulative Integral")
plt.title("Cumulative Integral: Quantum vs True")
plt.legend()
plt.grid(True)
# Plot absolute error
plt.subplot(2, 1, 2)
abs_error = np.abs(estimated_cumulative - true_cumulative)
plt.plot(x_vals, abs_error, color='red', label="Absolute Error")
plt.xlabel("x")
plt.ylabel("Error")
plt.title("Cumulative Integral Absolute Error")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

job_twirling_ancilla = runtime_service_paid.job('d1ahw6r5z6q0008pedd0')
job_result_twirling_ancilla = job_twirling_ancilla.result()
binary_order_2 = ['1', '0']
prob_twirling_ancilla = compute_probabilities(job_result_twirling_ancilla, binary_order_2)
print("Probabilities with twirling:", prob_twirling_ancilla)