import cudaq
from cudaq import spin
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

cudaq.set_qpu("custatevec_f32")

n_qubits = 10
n_samples = 50
h = spin.z(0)

n_parameters = n_qubits * 3
parameters = np.random.default_rng(13).uniform(
    low=0, high=1, size=(n_samples, n_parameters)
)
np.random.seed(1)

kernel, params = cudaq.make_kernel(list)

qubits = kernel.qalloc(n_qubits)
qubits_list = list(range(n_qubits))

for i in range(n_qubits):
    kernel.rx(params[i], qubits[i])

for i in range(n_qubits):
    kernel.ry(params[i + n_qubits], qubits[i])

for i in range(n_qubits):
    kernel.rz(params[i + n_qubits * 2], qubits[i])

for q1, q2 in zip(qubits_list[0::2], qubits_list[1::2]):
    kernel.cz(qubits[q1], qubits[q2])

exp_vals = [
    cudaq.observe(kernel, h, parameters[i]).expectation_z()
    for i in range(parameters.shape[0])
]


print(exp_vals)
