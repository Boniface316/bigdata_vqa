import cudaq
import numpy as np


def get_optimizer(max_iterations, circuit_depth, qubits):
    parameter_count = 4 * circuit_depth * (qubits - 1)

    optimizer = cudaq.optimizers.COBYLA()
    optimizer.initial_parameters = np.random.uniform(
        -np.pi / 8.0, np.pi / 8.0, parameter_count
    )
    optimizer.max_iterations = max_iterations
    return optimizer, parameter_count
