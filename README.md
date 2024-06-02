# Big Data Small Quantum Computer

## Abstract

Current quantum hardware prohibits any direct use of large classical datasets. Coresets allow for a succinct description of these large datasets and their solution in a computational task is competitive with the solution on the original dataset. The method of combining coresets with small quantum computers to solve a given task that requires a large number of data points was first introduced by Harrow [arXiv:2004.00026]. In this paper, we apply the coreset method in three different well-studied classical machine learning problems, namely Divisive Clustering, 3-means Clustering, and Gaussian Mixture Model Clustering. We provide a Hamiltonian formulation of the aforementioned problems for which the number of qubits scales linearly with the size of the coreset. Then, we evaluate how the variational quantum eigensolver (VQE) performs on these problems and demonstrate the practical efficiency of coresets when used along with a small quantum computer. We perform noiseless simulations on instances of sizes up to 25 qubits on CUDA Quantum and show that our approach provides comparable performance to classical solvers.

Authors: [Boniface Yogendran](https://github.com/Boniface316), [Daniel Charlton](https://github.com/DanLeeC), [Miriam Beddig](https://github.com/12mB7693), [Ioannis Kolotouros](https://github.com/greater) and [Dr. Petros Wallden](http://www.pwallden.gr/)

This repository contains the code and data required to reproduce the results presented in the paper titled "Big Data Small Quantum Computer." The paper is available on arXiv: [arXiv:2402.01529](https://arxiv.org/abs/2402.01529).


There are three algorithms presented in the paper, each with its own notebook in the `Notebooks` directory. The notebooks are as follows:

1. Coreset
2. 3-means Clustering
3. Divisive Clustering
4. Gaussian Mixture Model Clustering

## Coreset

Coreset construction is a method to reduce the size of a dataset while maintaining the same properties as the original dataset. This notebook demonstrates the construction of a coreset for a given dataset and the comparison of the performance of the coreset with the original dataset. For this particular experiment we used Algorithm 2 from the papers [Practical Coreset Constructions for Machine Learning](https://arxiv.org/abs/1703.06476) and [New Frameworks for Offline and Streaming Coreset Constructions](https://arxiv.org/abs/1612.00889). The notebook contains the script to generate the coreset using the algorithm mentioned above.

There are more algorithms available for coreset construction, and the choice of algorithm depends on the dataset and the problem at hand. If you want to use a different algorithm, you can include that in the `Coreset` class in `src/bigdatavqa/_coreset.py`. In order to maintain the same syntax, the new algorithm must have `get_coresets_using_` as the prefix for the method that generates the coreset. This method needs to mentioned in the `get_best_coresets` method in the `Coreset` class. Below is an example of how to add a new algorithm to the `Coreset` class.

```python
class Coreset:
    def __init__():
        pass
    def get_coresets_using_new_algorithm(self):
        # Add the code to generate the coreset using the new algorithm
        return coreset

    def get_best_coresets(self):
        # code from the original implementation
        
        if self._coreset_method == "older_algorithms":
            coreset_vectors, coreset_weights =  self.get_coresets_using_algorithm_1()

        elif self._coreset_method == "new_algorithm":
            coreset_vectors, coreset_weights = self.get_coresets_using_new_algorithm()
        else:
            raise ValueError("Invalid coreset method")

        # code from the original implementation

```
Future work is planned to modularize the coreset construction process and make it easier to implement new algorithms.

## Divisive Clustering

The divisive clustering algorithm is a method to divide a dataset into clusters. The algorithm starts with the entire dataset as one cluster and then divides the dataset into smaller clusters. The algorithm continues to divide the clusters until the desired number of clusters is reached. `Divisive_clustering_toy_problem.ipynb` demonstrates the divisive clustering algorithm on a toy dataset. The divisive clustering in the notebook was implemented using variational quantum algorithm (VQA), KMeans, random and MaxCut.


### VQA Divisive Clustering

VQA is a quantum algorithm that uses quantum-classical loop to solve optimization problems. The algorithm uses a quantum circuit to encode the problem and a classical optimizer to find the optimal solution. The main parts of the VQA algorithm are the ansatz, problem Hamiltonian and classical optimizer. 

#### Ansatz
The ansatz confiuration used for this experiment is located in `src/bigdatavqa/ansatz/_ansatz.py`. A new ansatz configuration can be added by creating a new function in the `_ansatz.py` file. The new function must follow the naming convention `get_[NAME]_circuit` where `[NAME]` is the name of the new ansatz. The new function must return a `cudaq.Kernel` object.

#### Problem Hamiltonian
The problem Hamiltonian is the Hamiltonian that encodes the problem. The Hamiltonian configuration used for this experiment is located in `src/bigdatavqa/Hamiltonians/_hamiltonian.py`. A new Hamiltonian configuration can be added by creating a new function in the `_hamiltonian.py` file. The new function must follow the naming convention `get_[NAME]_Hamiltonian` where `[NAME]` is the name of the new Hamiltonian. The new function must return a `cudaq.SpinOperator` object.

#### Classical Optimizer
The classical optimizer is the optimizer that finds the optimal solution. The classical optimizer configuration used for this experiment is located in `src/bigdatavqa/optimizer/_optimizer.py`. A new classical optimizer configuration can be added by creating a new function in the `_optimizer.py` file. The new function will have the prefix `get_optimizer_for` followed by the name of the approach. The new function must return a `cudaq.optimizers.optimizer` and  `int`. The `int` is the number of parameters for the optimizer.

### Example of combining the ansatz, Hamiltonian and optimizer

Below is an example of how to combine the ansatz, Hamiltonian and optimizer to create a VQA algorithm for the divisive clustering problem. These functions can be directly used in the `DivisiveClusteringVQA`. 

```python
from bigdatavqa.divisiveclustering import DivisiveClusteringVQA

def get_VQE_circuit():
    # Add the code to generate the VQE circuit
    return circuit

def get_K2_Hamiltonian():
    # Add the code to generate the Hamiltonian
    return Hamiltonian

def get_optimizer_for_VQE():
    # Add the code to generate the optimizer
    return optimizer, num_params

optimizer = cudaq.optimizers.COBYLA()

VQE_divisive_clustering = DivisiveClusteringVQA(
    coreset_df=coreset_df,
    vector_columns=vector_columns,
    weights_column=weights_column,
    circuit_depth=circuit_depth,
    max_iterations=max_iterations,
    max_shots=max_shots,
    threshold_for_max_cut=0.75,
    create_Hamiltonian=get_K2_Hamiltonian,
    optimizer=optimizer,
    optimizer_function=get_optimizer_for_VQE,
    create_circuit=get_VQE_circuit,
    normalize_vectors=True,
    sort_by_descending=True,
    coreset_to_graph_metric="dist",
)
   
```



# Running the experiements

## Setting up the environment

1. Clone the repository
2. run `docker build .` to build the docker image.
3. run `docker run -it --runtime=nvidia -p 8888:8888 --gpus all <image_id>` to start the jupyter notebook server.

## Running the experiments in the notebook

1. Open the jupyter notebook server in your browser by navigating to `http://localhost:8888/`
2. Open any notebook in the `notebooks` directory and run the cells to reproduce the results.

# Support

This repo is intended to be used as a reference for the paper. We are in the process of making the code more user-friendly and easier to use. This may break the current code. If you do encounter any issues, please open an issue and assign it to `@Boniface316`.

# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
