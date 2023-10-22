from typing import Optional

import cudaq
import networkx as nx
import numpy as np
import pandas as pd
from cudaq import spin

from ..coreset._coreset import coreset_to_graph, get_cv_cw
from ._helpers import create_pauli_operators


def kernel_two_local(number_of_qubits, circuit_depth) -> cudaq.Kernel:
    """QAOA ansatz for maxcut"""
    kernel, thetas = cudaq.make_kernel(list)
    qreg = kernel.qalloc(number_of_qubits)

    # Loop over the layers
    theta_position = 0

    for i in range(circuit_depth):
        for j in range(1, number_of_qubits):
            kernel.rz(thetas[theta_position], qreg[j % number_of_qubits])
            kernel.rx(thetas[theta_position + 1], qreg[j % number_of_qubits])
            kernel.cx(qreg[j], qreg[(j + 1) % number_of_qubits])
            kernel.rz(thetas[theta_position + 2], qreg[j % number_of_qubits])
            kernel.rx(thetas[theta_position + 3], qreg[j % number_of_qubits])
            theta_position += 4

    return kernel
