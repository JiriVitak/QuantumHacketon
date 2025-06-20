#!/usr/bin/env python
# coding: utf-8

# Libraries

# In[3]:


from qiskit_ibm_runtime import SamplerV2, EstimatorV2, QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


# IBM service log in

# In[4]:


runtime_service_Jiri = QiskitRuntimeService(
    channel='ibm_quantum',
    instance='cvut/general/simulphys',
    token='63c4bc4d05bc301d88a79333e99f7595c30542815998fbc4c0b3e2b12f3d251bf80f29fa43b35fe7fbf9d49816eed489b9f1e36aed117185400098f6b19b93b9'
)

backend = runtime_service_Jiri.backend('ibm_aachen')
pm = generate_preset_pass_manager(optimization_level=3, backend=backend)


# In[ ]:




