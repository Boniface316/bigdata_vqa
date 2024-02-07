# Big Data Small Quantum Computer

Authors: [Boniface Yogendran](https://github.com/Boniface316), [Daniel Charlton](https://github.com/DanLeeC), [Miriam Beddig](https://github.com/12mB7693), [Ioannis Kolotouros](https://github.com/greater) and [Dr. Petros Wallden](http://www.pwallden.gr/)

This repository contains the code and data required to reproduce the results presented in the paper titled "Big Data Small Quantum Computer." The paper is available on arXiv: [arXiv:2402.01529](https://arxiv.org/abs/2305.13594).

There are three algorithms presented in the paper, each with its own notebook in the `notebooks` directory. The notebooks are as follows:

1. 3-means Clustering
2. Divisive Clustering
3. Gaussian Mixture Model Clustering

## Abstract

Current quantum hardware prohibits any direct use of large classical datasets. Coresets allow for a succinct description of these large datasets and their solution in a computational task is competitive with the solution on the original dataset. The method of combining coresets with small quantum computers to solve a given task that requires a large number of data points was first introduced by Harrow [arXiv:2004.00026]. In this paper, we apply the coreset method in three different well-studied classical machine learning problems, namely Divisive Clustering, 3-means Clustering, and Gaussian Mixture Model Clustering. We provide a Hamiltonian formulation of the aforementioned problems for which the number of qubits scales linearly with the size of the coreset. Then, we evaluate how the variational quantum eigensolver (VQE) performs on these problems and demonstrate the practical efficiency of coresets when used along with a small quantum computer. We perform noiseless simulations on instances of sizes up to 25 qubits on CUDA Quantum and show that our approach provides comparable performance to classical solvers.

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
