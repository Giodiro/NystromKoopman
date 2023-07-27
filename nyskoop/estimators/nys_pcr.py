from typing import Tuple

import falkon
import scipy
import torch
from scipy.sparse.linalg import LinearOperator
from sklearn.utils.validation import check_is_fitted

from nyskoop.estimators.base_estimator import BaseNystromKoopmanEstimator
from nyskoop.estimators.estimator_utils import (
    pseudo_inverse, fetch_topr_eigensystem, EIGENVALUE_EPS,
    Eigendecomposition, id_regularize, svd_flip,
)


class ExactKoopmanNystromPcr(BaseNystromKoopmanEstimator):
    """Nystroem PCR estimator for Koopman operator regression.

    Parameters
    ----------
    kernel : falkon.kernels.Kernel
        The kernel used for embedding states of the dynamical system
    center_selection : falkon.center_selection.CenterSelector
        Strategy for selecting the Nystroem centers. For example you can use
        :class:`falkon.center_selection.UniformSelector` to choose the centers uniformly at
        random.
    num_components : int
        Number of principal components for the estimator. The resulting Koopman approximation
        will have at most rank `num_components`.
    svd_solver : {'full', 'arnoldi'}, default='full'
        The algorithm used to compute a top-r SVD of an m*m matrix. The number of SVD components
        is controlled by `num_components` and the matrix size is controlled by the number of
        centers from the `center_selection` strategy.
        The Arnoldi solver becomes more efficient as the ratio of SVD components and centers
        becomes smaller.

    Notes
    -----
    Memory requirements for the :meth:`fit` method are a full n*m matrix (n is the number of points,
    m the number of centers), and two full m*m matrices.
    """
    def __init__(self,
                 kernel: falkon.kernels.Kernel,
                 center_selection: falkon.center_selection.CenterSelector,
                 num_components: int,
                 svd_solver: str = 'full'):
        super().__init__(kernel, center_selection)
        self.num_components = num_components
        self.svd_solver = svd_solver
        self.alpha_ = None

    def fit(self, X, Y):
        """Learns the parameters of the principal component regression problem.

        Parameters
        ----------
        X : torch.Tensor
            Covariates. These should be points in a time-series
        Y : torch.Tensor
            Targets. The corresponding future data in the time-series
        """
        ny_pts_x, ny_pts_y = self.center_selection.select(X, Y)
        self.ny_pts_x_, self.ny_pts_y_ = ny_pts_x, ny_pts_y

        self.U_, self.V_, self.fitted_components_ = self._compute_U_V_r(X, Y, ny_pts_x, ny_pts_y)
        self.alpha_ = self.V_ @ self.U_.T @ ny_pts_y
        return self

    def predict(self, X):
        check_is_fitted(self, ["ny_pts_x_", "alpha_"])
        return self.kernel.mmv(X, self.ny_pts_x_, self.alpha_)

    def _compute_U_V_r(self, X, Y, ny_pts_x, ny_pts_y) -> Tuple[torch.Tensor, torch.Tensor, int]:
        kx_mm = self.kernel(ny_pts_x, ny_pts_x)
        ky_mm = self.kernel(ny_pts_y, ny_pts_y)
        eps = self.kernel.params.cg_epsilon(X.dtype) * ny_pts_x.shape[0]

        # Generalized eigenvalue problem A @ v_i = w_i * B @ v_i
        # where A = kx_nm_sq is real symmetric and B = kx_mm is positive definite.
        if self.svd_solver == 'full':
            kx_nm = self.kernel(X, ny_pts_x)
            kx_nm_sq = kx_nm.T @ kx_nm
            w, v = scipy.linalg.eigh(kx_nm_sq.numpy(), id_regularize(kx_mm, eps))  # normalization leads to needing to invert evals
        elif self.svd_solver == 'arnoldi':
            # linop = LinearOperator(
            #     (ny_pts_x.shape[0], ny_pts_x.shape[0]),
            #     matvec=lambda x: self.kernel.dmmv(X, ny_pts_x, torch.from_numpy(x), None).numpy(),
            #     dtype=X.numpy().dtype
            # )
            # w, v = scipy.sparse.linalg.eigsh(
            #     linop, M=id_regularize(kx_mm, eps).numpy(),
            #     k=self.num_components + 3, which='LM',
            #     tol=1e-5, maxiter=10,
            # )
            kx_nm = self.kernel(X, ny_pts_x)
            kx_nm_sq = kx_nm.T @ kx_nm
            w, v = scipy.sparse.linalg.eigsh(
                kx_nm_sq.numpy(), M=id_regularize(kx_mm, eps).numpy(),
                k=self.num_components + 3, which='LM',
            )
        else:
            raise NotImplementedError(self.svd_solver)
        edec = Eigendecomposition(torch.from_numpy(w), torch.from_numpy(v))
        edec_topr = fetch_topr_eigensystem(edec, r=self.num_components, threshold=1e-12, what='real')
        edec_topr_inv = pseudo_inverse(edec_topr, EIGENVALUE_EPS[X.dtype])

        V = edec_topr_inv.eigenvectors * edec_topr_inv.eigenvalues

        # U = kx_nm @ edec_topr_inv.eigenvectors
        U = self.kernel.mmv(X, ny_pts_x, edec_topr_inv.eigenvectors)
        U = self.kernel.mmv(ny_pts_y, Y, U)
        U = torch.linalg.lstsq(id_regularize(ky_mm, eps), U).solution.contiguous()

        num_nonzero = edec_topr_inv.num_nonzero or self.num_components
        return U, V, num_nonzero


class RandomizedKoopmanNystromPcr(ExactKoopmanNystromPcr):
    """Nystroem PCR estimator for Koopman operator regression.

    This estimator uses randomized SVD for added computational efficiency compared to
    :class:`ExactKoopmanNystromPcr` at the expense of a higher variance.

    Parameters
    ----------
    kernel : falkon.kernels.Kernel
        The kernel used for embedding states of the dynamical system
    center_selection : falkon.center_selection.CenterSelector
        Strategy for selecting the Nystroem centers
    num_components : int
        Number of principal components for the estimator. The resulting Koopman approximation
        will have at most rank `num_components`.
    oversampling : int
        Controls the number of singular values/vectors computed using randomized SVD. The default
        (`oversampling=2`) means that `2 * num_components` singular vectors will be computed,
        and then truncated to `num_components`. A higher value leads to more precise results at
        the expense of speed.
    n_iter : int
        Number of power iterations in the randomized SVD algorithm. A higher value leads to
        better precision at the expense of speed.
    power_iteration_normalizer : {'auto', 'QR', 'LU', 'none'}, default='auto'
        How to normalize each power iteration: 'QR' factorization (the slowest but most accurate),
        'none' (fastest but becomes numerically unstable when `n_iter` is large, e.g. typically 5
        or larger), or 'LU' factorization (numerically stable but can lose slightly in accuracy).
        The 'auto' mode applies no normalization if `n_iter` <= 2 and switches to LU otherwise.

    Notes
    -----
    The memory requirements are smaller than :class:`ExactKoopmanNystromPcr`, as the n*m matrix
    is never formed. Only m*m matrices are needed (n the number of points, m the number of centers).
    """
    def __init__(self,
                 kernel: falkon.kernels.Kernel,
                 center_selection: falkon.center_selection.CenterSelector,
                 num_components: int,
                 oversampling: int = 2,
                 n_iter: int = 2,
                 power_iteration_normalizer: str = "auto"):
        super().__init__(kernel, center_selection, num_components=num_components)
        self.oversampling = oversampling
        self.n_iter = n_iter
        if power_iteration_normalizer == "auto":
            if self.n_iter <= 2:
                power_iteration_normalizer = "none"
            else:
                power_iteration_normalizer = "LU"
        self.power_iteration_normalizer = power_iteration_normalizer
        self.svd_cuda_driver = 'gesvda'  # gesvd, gesvdj or gesvda

    def _compute_U_V_r(self, X, Y, ny_pts_x, ny_pts_y) -> Tuple[torch.Tensor, torch.Tensor, int]:
        kx_mm = self.kernel(ny_pts_x, ny_pts_x)
        ky_mm = self.kernel(ny_pts_y, ny_pts_y)
        eps = self.kernel.params.cg_epsilon(X.dtype) * ny_pts_x.shape[0]
        r = int(self.num_components * self.oversampling)
        assert r <= ny_pts_x.shape[0]  # needed to get full rank F0

        L, info = torch.linalg.cholesky_ex(id_regularize(kx_mm, eps))  # L L^T = Kmm

        # Randomized SVD rangefinder
        Q = torch.randn((ny_pts_x.shape[0], r), dtype=X.dtype, device=X.device)  # M, r
        for i in range(self.n_iter):
            # A @ Q
            AQ = self.kernel.mmv(X, ny_pts_x, torch.linalg.solve_triangular(L.T, Q, upper=True).contiguous())
            if self.power_iteration_normalizer == "none":
                Q = AQ
            elif self.power_iteration_normalizer == "qr":
                Q = torch.linalg.qr(AQ, mode="reduced").Q.contiguous()
            elif self.power_iteration_normalizer == "lu":
                P, Lu, _ = torch.linalg.lu(AQ)
                Q = P @ Lu  # apply the permutation
            # A.T @ Q
            AhQ = torch.linalg.solve_triangular(L, self.kernel.mmv(ny_pts_x, X, Q), upper=False).contiguous()
            if self.power_iteration_normalizer == "none":
                Q = AhQ
            elif self.power_iteration_normalizer == "qr":
                Q = torch.linalg.qr(AhQ, mode="reduced").Q.contiguous()
            elif self.power_iteration_normalizer == "lu":
                P, Lu, _ = torch.linalg.lu(AhQ)
                Q = P @ Lu  # apply the permutation

        AQ = self.kernel.mmv(X, ny_pts_x, torch.linalg.solve_triangular(L.T, Q, upper=True).contiguous())
        Q = torch.linalg.qr(AQ, mode="reduced").Q.contiguous()  # n x r

        # B = Q.T @ A = Q.T @ kx_nm @ L^{-T} : r x m
        B = torch.linalg.solve_triangular(L, self.kernel.mmv(ny_pts_x, X, Q), upper=False).T

        # compute the SVD on the short, wide matrix `B`
        if B.is_cuda:
            Uhat, _, Vh = torch.linalg.svd(B, full_matrices=False, driver=self.svd_cuda_driver)
        else:
            Uhat, _, Vh = torch.linalg.svd(B, full_matrices=False)

        Uhat, _ = svd_flip(Uhat, Vh)
        evecs = torch.matmul(Q, Uhat)  # (n, min(r, d))
        evecs = evecs[:, :self.num_components].contiguous()

        # normalizer = L^{-1} @ kx_mn @ evecs^T
        normalizer = torch.linalg.solve_triangular(L, self.kernel.mmv(ny_pts_x, X, evecs), upper=False).contiguous()
        normalizer = torch.sqrt(
            torch.clamp_min(torch.real(torch.sum(normalizer ** 2, dim=0)), 1e-8)
        )
        evecs = evecs / normalizer
        V = torch.cholesky_solve(self.kernel.mmv(ny_pts_x, X, evecs), L, upper=False)

        U = self.kernel.mmv(ny_pts_y, Y, evecs)
        U = torch.linalg.lstsq(id_regularize(ky_mm, eps), U).solution.contiguous()

        return U, V, self.num_components
