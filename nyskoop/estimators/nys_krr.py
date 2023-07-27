import falkon
import torch
from sklearn.utils.validation import check_is_fitted

from nyskoop.estimators.base_estimator import BaseNystromKoopmanEstimator
from nyskoop.estimators.estimator_utils import (
    fetch_topr_eigensystem, pseudo_inverse, EIGENVALUE_EPS, id_regularize
)


class KoopmanNystromKrr(BaseNystromKoopmanEstimator):
    """Nystroem KRR estimator for Koopman operator regression.

    Parameters
    ----------
    kernel : falkon.kernels.Kernel
        The kernel used for embedding states of the dynamical system
    penalty : float
        The Tikhonov penalty.
    center_selection : falkon.center_selection.CenterSelector
        Strategy for selecting the Nystroem centers. For example you can use
        :class:`falkon.center_selection.UniformSelector` to choose the centers uniformly at
        random.
    """
    def __init__(self,
                 kernel: falkon.kernels.Kernel,
                 penalty: float,
                 center_selection: falkon.center_selection.CenterSelector):
        super(KoopmanNystromKrr, self).__init__(kernel, center_selection)
        self.penalty = penalty
        self.alpha_ = None
        self.fitX_ = None
        self.fitY_ = None

    def fit(self, X, Y):
        r"""
        Solve the Koopman Nystroem problem for KRR:

        .. math:: (K^x_{mn} K^x_{nm} + n\lambda K^x_{mm})^{-1} K^x_{mn} K^y_{nm} K^y_{mm}^{-1} f(Y)

        where for now :math:`f` is the identity function

        Parameters
        ----------
        X : torch.Tensor
            Covariates. These should be points in a time-series
        Y : torch.Tensor
            Targets. The corresponding future data in the time-series
        """
        ny_pts_x, ny_pts_y = self.center_selection.select(X, Y)
        self.fitX_, self.fitY_ = X, Y
        self.ny_pts_x_, self.ny_pts_y_ = ny_pts_x, ny_pts_y

        penalty = self.penalty * X.shape[0]

        kx_nm = self.kernel(X, ny_pts_x)
        kx_mm = self.kernel(ny_pts_x, ny_pts_x)

        ky_nm = self.kernel(Y, ny_pts_y)
        ky_mm = self.kernel(ny_pts_y, ny_pts_y)

        solve1 = torch.linalg.lstsq(ky_mm, ny_pts_y, driver='gels').solution

        rhs = kx_nm.T @ ky_nm @ solve1
        lhs = kx_nm.T @ kx_nm + penalty * kx_mm
        solve2 = torch.linalg.solve(lhs, rhs)
        self.alpha_ = solve2
        return self

    def predict(self, X):
        check_is_fitted(self, ["ny_pts_x_", "alpha_"])
        return self.kernel(X, self.ny_pts_x_) @ self.alpha_

    def eigenfunctions(self, what: str = 'eig'):
        check_is_fitted(self, ["fitX_", "fitY_", "ny_pts_x_", "ny_pts_y_"])
        self.U_, self.V_, self.fitted_components_ = self._compute_U_V_r(
            self.fitX_, self.fitY_, self.ny_pts_x_, self.ny_pts_y_)
        return super(KoopmanNystromKrr, self).eigenfunctions(what)

    def _compute_U_V_r(self, X, Y, ny_pts_x, ny_pts_y):
        penalty = self.penalty * X.shape[0]

        kx_nm = self.kernel(X, ny_pts_x)
        kx_mm = self.kernel(ny_pts_x, ny_pts_x)

        ky_nm = self.kernel(Y, ny_pts_y)
        ky_mm = self.kernel(ny_pts_y, ny_pts_y)

        toinv = kx_nm.T @ kx_nm + penalty * kx_mm
        edec1 = torch.linalg.eigh(toinv)
        r = kx_mm.shape[0]
        edec1_topr = fetch_topr_eigensystem(edec1, r=r)  # This is for sorting
        inv_edec1 = pseudo_inverse(edec1_topr, EIGENVALUE_EPS[edec1_topr.eigenvectors.dtype])
        V = inv_edec1.eigenvectors * inv_edec1.eigenvalues
        U = ky_nm.T @ kx_nm @ inv_edec1.eigenvectors
        U = torch.linalg.solve(id_regularize(ky_mm, 1e-8), U)
        return U, V, inv_edec1.num_nonzero or r
