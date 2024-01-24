# NysKoop

Kernel estimators for the Koopman operator with the Nystrom approximation.

This repository contains code to go along the following paper:
[G. Meanti*, A. Chatalic*,  V. R. Kostic, P. Novelli, M. Pontil, L. Rosasco, Estimating Koopman operators with sketching to
provably learn large scale dynamical systems, Conference on Neural Information Processing Systems, 2023.](https://arxiv.org/abs/2306.04520) 
Implementing all the algorithms and estimators (the Nystrom versions of KRR, PCR and RRR) 
described in the paper. 


## Installing

The code depends on [falkon](https://github.com/falkonml/falkon) and [keops](https://github.com/getkeops/keops) 
for kernel functions and fast kernel-vector products. 
[PyTorch](https://pytorch.org) and [numpy](https://numpy.org) are also used for their tensor implementations and linear algebra routines.

To install this package, first update pip and setuptools
```
pip install -U pip setuptools
```
then run pip install with the `--no-build-isolation` flag (otherwise pip will create a mess with pytorch versions):
```
pip install --no-build-isolation git+https://github.com/Giodiro/NystromKoopman.git
```

## API

The API follows scikit-learn conventions. There are 3 main estimators implemented:
 - Nystrom KRR
 - Nystrom PCR, with an additional randomized version (faster when the number of Nystrom centers is large)
 - Nystrom RRR, with an additional randomized version.

In addition to the canonical `fit` and `predict` methods, the estimators implement some additional methods: 
 - `eigenfunctions` which provides with eigenvalues of the operator, as well as left and right eigenfunctions
 - `modes` for the Koopman modes
 - `modes_forecast` for making predictions with the Koopman modes

## GPU Support

A subset of the estimators can run fully on the GPU, if PyTorch and Falkon are installed properly.
In particular the `RandomizedKoopmanNystromPcr` estimator will run wholly on the GPU. 
Other estimators, like `RandomizedKoopmanNystromRrr` should run on the GPU for the most part,
but may need to synchronize with the CPU for certain steps.

## Examples

The [notebooks](/notebooks) should provide some thorough examples, which were used to run the experiments contained in the paper.
In particular, check out the [BigProteins.ipynb](/notebooks/BigProteins.ipynb) notebook for an example of using the randomized versions of PCR and RRR 
to scale to large datasets.

## Citation

If you find this work useful, please cite the following paper:
```
@inproceedings{nyskoop23,
    title = {Estimating {Koopman} operators with sketching to provably learn large scale dynamical systems},
    author = {Giacomo Meanti and Antoine Chatalic and Vladimir R. Kostic and Pietro Novelli and Massimiliano Pontil and Lorenzo Rosasco},
    year = {2023},
    booktitle = {NeurIPS 2023}
}
```
