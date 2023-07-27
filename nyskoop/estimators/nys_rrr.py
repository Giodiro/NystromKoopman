import warnings
from typing import Tuple

import falkon
import scipy
import torch
from falkon.utils import TicToc
from sklearn.utils.validation import check_is_fitted

from nyskoop.estimators.base_estimator import BaseNystromKoopmanEstimator
from nyskoop.estimators.estimator_utils import (
    fetch_topr_eigensystem, id_regularize, weighted_norm, Eigendecomposition
)
from .conjgrad import ConjugateGradient


class ExactKoopmanNystromRrr(BaseNystromKoopmanEstimator):
    def __init__(self,
                 kernel: falkon.kernels.Kernel,
                 penalty: float,
                 center_selection: falkon.center_selection.CenterSelector,
                 num_components: int):
        super().__init__(kernel, center_selection)
        self.penalty = penalty
        self.num_components = num_components
        self.alpha_ = None

    def fit(self, X, Y):
        ny_pts_x, ny_pts_y = self.center_selection.select(X, Y)
        self.ny_pts_x_, self.ny_pts_y_ = ny_pts_x, ny_pts_y

        self.U_, self.V_, self.fitted_components_ = self._compute_U_V_r(X, Y, ny_pts_x, ny_pts_y)
        self.alpha_ = self.V_ @ self.U_.T @ ny_pts_y
        return self

    def predict(self, X):
        check_is_fitted(self, ["ny_pts_x_", "alpha_"])
        return self.kernel.mmv(X, self.ny_pts_x_, self.alpha_)

    def _compute_U_V_r(self, X, Y, ny_pts_x, ny_pts_y) -> Tuple[torch.Tensor, torch.Tensor, int]:
        penalty = self.penalty * X.shape[0]
        debug = False
        with TicToc("UV", debug=debug):
            with TicToc("kernels", debug=debug):
                kx_nm = self.kernel(X, ny_pts_x)
                kx_mm = self.kernel(ny_pts_x, ny_pts_x)

                ky_nm = self.kernel(Y, ny_pts_y)
                ky_mm = self.kernel(ny_pts_y, ny_pts_y)
                eps = self.kernel.params.cg_epsilon(X.dtype) * ny_pts_x.shape[0]

                # K_mn^x k_nm^y inv(k_mm^y) k_mn^y k_nm^x
                k_ymn_xnm = ky_nm.T @ kx_nm

            with TicToc("lstsq", debug=debug):
                k_ymminv_kymn_xnm = torch.linalg.lstsq(id_regularize(ky_mm, eps), k_ymn_xnm).solution
                kxyx = k_ymn_xnm.T @ k_ymminv_kymn_xnm

            # k_mn^x k_nm^x + n \lambda k_mm^x
            gkx = kx_nm.T @ kx_nm + penalty * kx_mm

            with TicToc("lstsq", debug=debug):
                solvebb = torch.linalg.lstsq(gkx, kxyx).solution

            with TicToc("eig", debug=debug):
                edecbb = torch.linalg.eig(solvebb)
                edecbb_topr = fetch_topr_eigensystem(edecbb, r=self.num_components, threshold=1e-12, what='real')
                if len(edecbb_topr.eigenvalues) < self.num_components:
                    warnings.warn(f"Discarding {self.num_components - len(edecbb_topr.eigenvalues)} components")

            # Check no imaginary data in the projector.
            # TODO Actually, the projector might be ok and complex if a global phase is present. Fix this.
            evec_w = edecbb_topr.eigenvectors
            if torch.is_complex(evec_w):
                if torch.amax(torch.abs(evec_w.imag)) > 1e-8:
                    warnings.warn(
                        "Computed projector is not real. The Kernel matrix is either severely ill "
                        "conditioned or non-symmetric, discarding imaginary parts.")
                evec_w = evec_w.real

            # Normalize
            evec_w = evec_w / weighted_norm(evec_w, kxyx)

            U = k_ymminv_kymn_xnm @ evec_w
            V = solvebb @ evec_w

        return U, V, len(edecbb_topr.eigenvalues)


class RandomizedKoopmanNystromRrr(ExactKoopmanNystromRrr):
    def __init__(self,
                 kernel: falkon.kernels.Kernel,
                 penalty: float,
                 center_selection: falkon.center_selection.CenterSelector,
                 num_components: int,
                 oversampling: float = 2.0,
                 n_iter: int = 2):
        super().__init__(kernel, penalty, center_selection, num_components)
        self.oversampling = oversampling
        self.n_iter = n_iter
        self.linsys_solver = "lu"

    def _explicit_glambda(self, X, ny_pts_x, penalty: float, reg_kmm: float):
        kx_nm = self.kernel(X, ny_pts_x)
        kx_mm = id_regularize(self.kernel(ny_pts_x, ny_pts_x), reg_kmm)  # TODO: Reg here is fundamental, but really high?
        gkx = kx_nm.T @ kx_nm + penalty * kx_mm
        return gkx

    def _implicit_glambda(self, X, ny_pts_x, penalty, reg_kmm):
        kx_mm = id_regularize(self.kernel(ny_pts_x, ny_pts_x), reg_kmm)  # TODO: Reg here is fundamental, but really high?

        def inner(vec):
            out = (self.kernel.dmmv(X, ny_pts_x, vec, None) + penalty * kx_mm @ vec)
            return out

        return inner

    def _solve(self, X, ny_pts_x, rhs, penalty, **saved_data) -> Tuple[torch.Tensor, dict]:
        if self.linsys_solver == 'lu':
            if 'gl_ludec' not in saved_data:
                gkx = self._explicit_glambda(X, ny_pts_x, penalty, 1e-3)
                LU, pivots = torch.linalg.lu_factor(gkx)
                saved_data['gl_ludec'] = (LU, pivots)
            LU, pivots = saved_data['gl_ludec']
            return torch.linalg.lu_solve(LU, pivots, rhs).contiguous(), saved_data
        elif self.linsys_solver == 'cholesky':
            if 'gl_cho' not in saved_data:
                gkx = self._explicit_glambda(X, ny_pts_x, penalty, 1e-3)
                L, info = torch.linalg.cholesky_ex(gkx)
                if info > 0:
                    raise RuntimeError(f"Failed cholesky decomposition at leading minor of order {info.item()}")
                saved_data['gl_cho'] = L
            L = saved_data['gl_cho']
            return torch.cholesky_solve(rhs, L).contiguous(), saved_data
        elif self.linsys_solver == 'cg':
            cg = ConjugateGradient()
            out = cg.solve(None, rhs, self._implicit_glambda(X, ny_pts_x, penalty, 1e-5), 100)
            return out, saved_data
        else:
            raise ValueError(self.linsys_solver)

    def _kxyx_matmul(self, X, ny_pts_x, Y, ny_pts_y, rhs, **saved_data) -> Tuple[torch.Tensor, dict]:
        ker = self.kernel
        rhs = ker.mmv(X, ny_pts_x, rhs)
        rhs = ker.mmv(ny_pts_y, Y, rhs)
        if 'kymm_ludec' not in saved_data:
            ky_mm = id_regularize(
                ker(ny_pts_y, ny_pts_y),
                ker.params.cg_epsilon(Y.dtype) * ny_pts_y.shape[0]
            )
            saved_data['kymm_ludec'] = torch.linalg.lu_factor(ky_mm)
        LU, pivots = saved_data['kymm_ludec']
        rhs = torch.linalg.lu_solve(LU, pivots, rhs).contiguous()
        rhs = ker.mmv(Y, ny_pts_y, rhs)
        rhs = ker.mmv(ny_pts_x, X, rhs)
        return rhs, saved_data

    def _compute_U_V_r(self, X, Y, ny_pts_x, ny_pts_y) -> Tuple[torch.Tensor, torch.Tensor, int]:
        penalty = self.penalty * X.shape[0]
        ker = self.kernel
        dt, dev = X.dtype, X.device

        r = int(self.num_components * self.oversampling)

        rand_vecs = torch.randn((X.shape[0], r), dtype=dt, device=dev)
        sp_bar = ker.mmv(ny_pts_x, X, rand_vecs)
        saved_data = {}
        for i in range(self.n_iter):
            sp_bar, saved_data = self._solve(X, ny_pts_x, sp_bar, penalty, **saved_data)
            sp_bar, saved_data = self._kxyx_matmul(X, ny_pts_x, Y, ny_pts_y, sp_bar, **saved_data)
        sp, saved_data = self._solve(X, ny_pts_x, sp_bar, penalty, **saved_data)
        F0 = sp_bar.T @ sp
        sp1_bar, saved_data = self._kxyx_matmul(X, ny_pts_x, Y, ny_pts_y, sp, **saved_data)
        F1 = sp.T @ sp1_bar

        evals_np, evecs_np = scipy.linalg.eigh(F1.to("cpu").numpy(), F0.to("cpu").numpy())
        edec = Eigendecomposition(torch.from_numpy(evals_np).to(dev), torch.from_numpy(evecs_np).to(dev))
        edec = fetch_topr_eigensystem(edec, r=self.num_components, threshold=1e-12, what='real')
        if torch.is_complex(edec.eigenvectors):
            if torch.amax(torch.abs(edec.eigenvectors.imag)) > 1e-8:
                warnings.warn(
                    "Computed projector is not real. The Kernel matrix is either severely ill "
                    "conditioned or non-symmetric, discarding imaginary parts.")
            edec.eigenvectors = edec.eigenvectors.real

        # Normalize
        evecs = edec.eigenvectors / weighted_norm(edec.eigenvectors, F0)

        # Compute V, U for output
        V = sp @ evecs
        U = ker.mmv(X, ny_pts_x, V)
        U = ker.mmv(ny_pts_y, Y, U)
        LU, pivots = saved_data['kymm_ludec']
        U = torch.linalg.lu_solve(LU, pivots, U).contiguous()

        return U, V, len(edec.eigenvalues)
