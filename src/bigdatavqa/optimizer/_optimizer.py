from typing import Tuple

import cudaq
import numpy as np


def get_optimizer_for_VQE(
    optimizer: cudaq.optimizers.optimizer, max_iterations, **kwargs
) -> Tuple[cudaq.optimizers.optimizer, int]:
    """Returns the optimizer with the given parameters

    Args:
        optimizer (cudaq.optimizers.optimizer): Optimizer
        max_iterations (int): Maximum number of iterations
        **kwargs: Additional arguments

    Returns:
        tuple(cudaq.optimizers.optimizer, int): Optimizer and parameter count
    """
    parameter_count = 4 * kwargs["circuit_depth"] * kwargs["qubits"]
    initial_params = np.random.uniform(-np.pi / 8.0, np.pi / 8.0, parameter_count)
    optimizer.initial_parameters = initial_params

    optimizer.max_iterations = max_iterations
    return optimizer, parameter_count


def get_optimizer_for_QAOA(
    optimizer: cudaq.optimizers.optimizer, max_iterations, **kwargs
) -> Tuple[cudaq.optimizers.optimizer, int]:
    """
    Returns the optimizer with the given parameters

    Args:
        optimizer (cudaq.optimizers.optimizer): Optimizer
        max_iterations (int): Maximum number of iterations
        **kwargs: Additional arguments

    Returns:
        tuple(cudaq.optimizers.optimizer, int): Optimizer and parameter count
    """

    parameter_count = 2 * kwargs["circuit_depth"]
    optimizer.initial_parameters = np.random.uniform(-np.pi / 8.0, np.pi / 8.0, parameter_count)
    optimizer.max_iterations = max_iterations
    return optimizer, parameter_count
