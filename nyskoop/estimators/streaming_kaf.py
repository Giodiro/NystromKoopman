"""
Python translation of the matlab streaming KAF code found here:
https://github.com/rward314/StreamingKAF

Implements the Scalable Kernel Analog Forecasting (Scalable-KAF) algorithm described in

D. Giannakis, A. Henriksen, J. Tropp, R. Ward, Learning to Forecast Dynamical Systems from
Streaming Data, SIAM Journal on Applied Dynamical Systems, 2023.
"""
import math
from typing import Tuple

import numpy as np
import scipy.sparse.linalg
import torch

from nyskoop.estimators.estimator_utils import pseudo_inverse_evals

__all__ = [
    "ScalableKAF",
]


class ScalableKAF:
    def __init__(self,
                 k_bandwidth: float,
                 d: int,
                 num_features: int,
                 approx_rank: int,
                 block_size: int = 1000):
        self.num_features = num_features
        self.approx_rank = approx_rank
        self.k_bandwidth = k_bandwidth
        self.feat_dir = (
            math.sqrt(2 * self.k_bandwidth) * np.random.randn(self.num_features, d)
        )  # F, D
        self.feat_shift = (
            2 * math.pi * np.random.rand(self.num_features, 1)
        )  # F, 1
        self.alpha_ = None
        self.fit_X_, self.fit_Y_ = None, None
        self.block_size = block_size
        self.passes = 1

    def fit(self, X: torch.Tensor, Y: torch.Tensor):
        # To keep the translation from matlab as literal as possible we change the matrix order
        # from samples (N), features (D) -- which is used as input format -- to features, samples
        # which is used internally by the algorithm.
        self.fit_X_ = X.T.numpy()  # D, N
        self.fit_Y_ = Y.T.numpy()  # Dy, N

        subspace, evals = self.feat_nystrom(self.fit_X_)
        evals = np.diag(evals)
        evals = evals + np.max(evals) * 1e-6

        C = self.rmult_t(self.fit_X_, self.fit_Y_)  # Dy, F
        self.alpha_ = ((C @ subspace) @ scipy.linalg.pinv(evals)) @ subspace.T
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        X = X.T.numpy()
        return torch.from_numpy(self.rmult(X, self.alpha_).T)  # N, ?

    def feat_nystrom(self, X):
        """Computes a truncated eigenvalue decomposition of
        featurized data by streaming over the columns

        Parameters
        ----------
         X : np.ndarray of shape num_dims x num_points
        """
        rank = min(self.num_features, self.approx_rank)
        subspace_dim = min(2 * self.approx_rank, self.num_features)  # oversample 2x

        Z = np.random.randn(self.num_features, subspace_dim)  # F x K
        Q = None
        for i in range(self.passes):
            Q, _ = scipy.linalg.qr(Z, mode="economic", check_finite=False)  # F x K
            Z = self.mult_cov(X, Q)  # F, K

        C = Q.T @ Z  # K, K
        C = (C + C.T) / 2  # symmterize
        evals, evec = scipy.linalg.eigh(C)
        evals = np.sqrt(np.maximum(evals, 0))
        inv_evals = pseudo_inverse_evals(evals, 1e-12)[0]

        nystrom_sqrt = (Z @ evec) @ np.diag(inv_evals)

        Q, sigma, _ = scipy.linalg.svd(nystrom_sqrt, full_matrices=False, check_finite=False)
        Q = Q[:, :rank]
        evals = sigma[:rank] ** 2
        return Q, evals

    def mult_cov(self, X, M):
        """Forms phi(X) @ phi(X).T @ M in one pass over rows of X.

        X: D, N
        M: F, ?
        out: F, ?
        """
        assert M.shape[0] == self.num_features
        CM = np.zeros((self.num_features, M.shape[1]))

        num_blocks, block_size = self._get_block_info(X.shape[1])
        for j in range(num_blocks):
            phix = self.featurize(X[:, j * block_size: (j + 1) * block_size])  # F, block
            CM = CM + phix @ (phix.T @ M)

        return CM

    def featurize(self, X):
        """
        Computes features over columns of X

        :param X: D, N
        :return:  F, N
        """
        feat_shift = self.feat_shift @ np.ones((1, X.shape[1]))  # F, N
        phi_x = self.feat_dir @ X  # F, N
        phi_x = phi_x + feat_shift
        phi_x = math.sqrt(2 / self.num_features) * np.cos(phi_x)
        return phi_x

    def rmult_t(self, X, M):
        """
        M @ phi(X).T

        :param X: D, N
        :param M: ?, N
        :return: ?, F
        """
        num_rows = X.shape[1]
        assert M.shape[1] == num_rows

        mphi_t = np.zeros((M.shape[0], self.num_features))
        num_blocks, block_size = self._get_block_info(num_rows)
        for j in range(num_blocks):
            phix = self.featurize(X[:, j * block_size: (j + 1) * block_size])  # F, block
            mphi_t = mphi_t + M[:, j * block_size: (j + 1) * block_size] @ phix.T
        return mphi_t

    def rmult(self, X, M):
        """
        M @ phi(X)

        :param X:  D, N
        :param M:  ?, F
        :return:   ?, N
        """
        assert M.shape[1] == self.num_features

        mphi = np.zeros((M.shape[0], X.shape[1]))  # ?, N
        num_blocks, block_size = self._get_block_info(X.shape[1])
        for j in range(num_blocks):
            phix = self.featurize(X[:, j * block_size: (j + 1) * block_size])  # F, block
            mphi[:, j * block_size: (j + 1) * block_size] = M @ phix
        return mphi  # ?, N

    def _get_block_info(self, num_points) -> Tuple[int, int]:
        if self.block_size is None:
            return 1, num_points
        return math.ceil(num_points / self.block_size), self.block_size
