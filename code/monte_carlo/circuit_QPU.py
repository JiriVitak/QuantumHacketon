import numpy as np
from qiskit import qpy
from qiskit.circuit import Parameter, QuantumCircuit, ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerOptions
from qiskit_ibm_runtime.options import TwirlingOptions, DynamicalDecouplingOptions, EnvironmentOptions
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2
from qiskit_ibm_runtime import EstimatorV2

twirling_options = TwirlingOptions(
    enable_gates=True,
    enable_measure=False,
    num_randomizations="auto",
    shots_per_randomization="auto",
    strategy="active-circuit"
)

dd_options_XX = DynamicalDecouplingOptions(
    enable=True,
    extra_slack_distribution = "middle",
    scheduling_method = "alap",
    sequence_type = "XX",
    skip_reset_qubits = False
)

dd_options_XY4 = DynamicalDecouplingOptions(
    enable=True,
    extra_slack_distribution = "middle",
    scheduling_method = "alap",
    sequence_type = "XY4",
    skip_reset_qubits = False
)

with open("sin^2.qpy", "rb") as handle:
    qc = qpy.load(handle)
"""
# Make sure there is at least 1 classical bit:
if qc[0].num_clbits < 1:
    cr = ClassicalRegister(1)
    qc[0].add_register(cr)

# Now it's safe to measure qubit 3 to classical bit 0
qc[0].measure(3, 0)
"""

# Make sure there is at least 3 classical bits:
if qc[0].num_clbits < 3:
    # Create a new classical register with 3 bits
    cr = ClassicalRegister(3)
    qc[0].add_register(cr)

# Measure first 3 qubits into the classical bits 0,1,2
for qubit_idx in range(3):
    qc[0].measure(qubit_idx, qubit_idx)

#print(qc[0])


runtime_service_paid = QiskitRuntimeService(
    channel='ibm_quantum',
    instance='cvut/general/simulphys',
    token='63c4bc4d05bc301d88a79333e99f7595c30542815998fbc4c0b3e2b12f3d251bf80f29fa43b35fe7fbf9d49816eed489b9f1e36aed117185400098f6b19b93b9'
)

backend = runtime_service_paid.backend('ibm_fez')
pm = generate_preset_pass_manager(optimization_level=3, backend=backend)

no_mit_env_options = EnvironmentOptions(job_tags=["H_no_mit_3"])
twirling_env_options = EnvironmentOptions(job_tags=["H_twirling_3"])
dd_XX_env_options = EnvironmentOptions(job_tags=["H_dd_XX_3"])
dd_XY4_env_options = EnvironmentOptions(job_tags=["H_dd_XY4_3"])

sampler_no_mit = SamplerV2(mode=backend, options=SamplerOptions(environment=no_mit_env_options))
sampler_twirling = SamplerV2(mode=backend, options=SamplerOptions(environment=twirling_env_options, twirling=twirling_options))
sampler_dd_XX = SamplerV2(mode=backend, options=SamplerOptions(environment=dd_XX_env_options, dynamical_decoupling=dd_options_XX))
sampler_dd_XY4 = SamplerV2(mode=backend, options=SamplerOptions(environment=dd_XY4_env_options, dynamical_decoupling=dd_options_XY4))

shots = 1000

isa_circuit = pm.run(qc[0])
#print(isa_circuit)

job_no_mit = sampler_no_mit.run([isa_circuit], shots=shots)
job_twirling = sampler_twirling.run([isa_circuit], shots=shots)
#job_dd_XX = sampler_dd_XX.run([isa_circuit], shots=shots)
#job_dd_XY4 = sampler_dd_XY4.run([isa_circuit], shots=shots)

