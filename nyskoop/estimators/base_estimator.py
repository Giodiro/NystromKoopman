import falkon.kernels
import torch
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted

from nyskoop.estimators.estimator_utils import (
    fetch_topr_eigensystem, to_complex, weighted_norm,
    Eigendecomposition
)


class BaseNystromKoopmanEstimator(BaseEstimator, RegressorMixin):
    def __init__(self,
                 kernel: falkon.kernels.Kernel,
                 center_selection: falkon.center_selection.CenterSelector,):
        super().__init__()
        self.kernel = kernel
        self.center_selection = center_selection

        self.ny_pts_x_ = None
        self.ny_pts_y_ = None
        self.U_ = None
        self.V_ = None
        self.fitted_components_ = None

    def eigenfunctions(self, what='eig'):
        check_is_fitted(self, ['ny_pts_x_', 'ny_pts_y_', 'U_', 'V_', 'fitted_components_'])
        ny_pts_x, ny_pts_y = self.ny_pts_x_, self.ny_pts_y_
        U, V, r = self.U_, self.V_, self.fitted_components_
        C = V.T @ self.kernel.mmv(ny_pts_x, ny_pts_y, U)

        W_X = V.T @ self.kernel.mmv(ny_pts_x, ny_pts_x, V)
        W_Y = U.T @ self.kernel.mmv(ny_pts_y, ny_pts_y, U)

        # Right eigenfunctions
        edec_right = torch.linalg.eig(C)
        edec_left = torch.linalg.eig(C.H)
        edec_left = Eigendecomposition(edec_left.eigenvalues, edec_left.eigenvectors.conj())

        edec_right_topr = fetch_topr_eigensystem(edec_right, r, what='real')
        edec_left_topr = fetch_topr_eigensystem(edec_left, r, what='real')
        evals = edec_right_topr.eigenvalues.conj()

        norm_left = weighted_norm(edec_left_topr.eigenvectors, to_complex(W_X))
        norm_right = weighted_norm(edec_right_topr.eigenvectors, to_complex(W_Y))

        evec_right = edec_right_topr.eigenvectors / norm_right
        evec_left = edec_left_topr.eigenvectors / norm_left

        def efun_right(new_X):
            k = to_complex(self.kernel.mmv(new_X, ny_pts_x, U))
            return k @ evec_right

        def efun_left(new_X):
            k = to_complex(self.kernel.mmv(new_X, ny_pts_y, V))
            return k @ evec_left / evals

        # Normalization required for Koopman modes
        if what == 'modes':
            left_right_dot = torch.sum(evec_left.conj() * evec_right, 0)
            left_right_norms = 1 / left_right_dot
            return evals, evec_right, left_right_norms, efun_left

        return evals, efun_right, efun_left

    def modes(self):
        check_is_fitted(self, ['ny_pts_y_', 'U_'])
        ny_pts_y = self.ny_pts_y_
        U = self.U_

        _, evec_left, left_right_norms, _ = self.eigenfunctions(what='modes')
        # modes = to_complex(ny_pts_y.T @ V) @ evec_left.conj() * left_right_norms.view(1, -1)
        modes = (evec_left.conj() * left_right_norms.view(1, -1)).T @ to_complex(U.T @ ny_pts_y)

        return modes

    def modes_forecast(self, X, steps=1):
        modes = self.modes()
        evals, evec_left, _, efun_right = self.eigenfunctions(what='modes')

        efun_right_x = efun_right(X)
        if isinstance(steps, int):
            steps = torch.arange(1, 1 + steps)
        evals_t = torch.pow(evals.view(-1, 1), steps[None, :].repeat(evals.shape[0], 1))
        forecasted = torch.einsum('ro,rt,nr->nto', modes, evals_t, efun_right_x)
        return forecasted.real
