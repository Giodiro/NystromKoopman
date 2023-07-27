import dataclasses
from typing import Protocol, Optional, Union, Tuple

import numpy as np
import torch


EIGENVALUE_EPS = {
    torch.float64: 1e-13,
    torch.float32: 1e-6,
}


class EigendecompositionProt(Protocol):
    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor


@dataclasses.dataclass
class Eigendecomposition(EigendecompositionProt):
    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor
    num_nonzero: Optional[int] = None


def fetch_topr_eigensystem(edec: EigendecompositionProt,
                           r: int,
                           threshold: Optional[float] = None,
                           what: str = 'real') -> Eigendecomposition:
    if what == 'real':
        sorter = torch.argsort(edec.eigenvalues.real, descending=True)[:r]
    elif what == 'modulus':
        sorter = torch.argsort(
            edec.eigenvalues.real ** 2 + edec.eigenvalues.imag ** 2, descending=True)[:r]
    else:
        raise ValueError(what)
    sorted_evals = edec.eigenvalues[sorter]
    sorted_evecs = edec.eigenvectors[:, sorter]
    if threshold is not None:
        valid = torch.argwhere(torch.ge(torch.real(sorted_evals), threshold)).view(-1)
        sorted_evals = sorted_evals[valid]
        sorted_evecs = sorted_evecs[:, valid]
    return Eigendecomposition(
        eigenvalues=sorted_evals, eigenvectors=sorted_evecs.contiguous())


def pseudo_inverse_evals(evals: Union[torch.Tensor, np.ndarray], threshold: Union[float, complex]):
    if isinstance(evals, np.ndarray):
        invalid_evals = np.abs(evals) < threshold
        inverse_evals = np.zeros_like(evals)
    else:
        invalid_evals = torch.lt(torch.abs(evals), threshold)
        inverse_evals = torch.zeros_like(evals)
        
    inverse_evals[~invalid_evals] = 1 / evals[~invalid_evals]

    return inverse_evals, invalid_evals


def pseudo_inverse(edec: EigendecompositionProt, threshold) -> Eigendecomposition:
    evals, evecs = edec.eigenvalues, edec.eigenvectors
    first_invalid_eval = torch.nonzero(torch.lt(evals, threshold))
    inverse_evals = torch.reciprocal(evals)
    if len(first_invalid_eval) > 0:
        first_invalid_eval = first_invalid_eval[0, 0]
        inverse_evals[first_invalid_eval:] = 0.0
        evecs[:, first_invalid_eval:].fill_(0.0)
        print(f"The numerical rank of the projector is smaller than the selected rank "
              f"({len(evals)}). {len(evals) - first_invalid_eval} degrees of freedom will be ignored.")
    elif evals[0] < 0:
        first_invalid_eval = 0
        print("No valid eigenvalue.")
    else:
        first_invalid_eval = None
    return Eigendecomposition(inverse_evals, evecs, num_nonzero=first_invalid_eval)


def to_complex(t: torch.Tensor) -> torch.Tensor:
    return torch.complex(t, torch.zeros_like(t))


def to_real(t: torch.Tensor) -> torch.Tensor:
    return torch.real(t)


def weighted_norm(A: torch.Tensor, M: Optional[torch.Tensor] = None):
    """Weighted norm of the columns of A.

    Args:
        A (ndarray): 1D or 2D array. If 2D, the columns are treated as vectors.
        M (ndarray or LinearOperator, optional): Weigthing matrix. the norm of the vector a is given by a.T@M@a. Defaults to None, corresponding to the Identity matrix. Warning: no checks are performed on M being a PSD operator.

    Returns:
        (ndarray or float): if A.ndim == 2 returns 1D array of floats corresponding to the norms of the columns of A. Else return a float.
    """
    if M is None:
        norm = torch.linalg.norm(A, dim=0)
    else:
        weighted_a = M @ A
        norm = torch.sqrt(
            torch.clamp_min(torch.real(torch.sum(torch.conj(A) * weighted_a, dim=0)), 1e-8)
        )
    return norm


def id_regularize(mat: torch.Tensor, reg_amount: Union[torch.Tensor, float]) -> torch.Tensor:
    n = mat.shape[0]
    return mat + (reg_amount * n * torch.eye(n, dtype=mat.dtype, device=mat.device))


def svd_flip(
    u: torch.Tensor,
    v: torch.Tensor,
    u_based_decision: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sign correction to ensure deterministic output from SVD.

    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.

    Parameters
    ----------
    u : torch.Tensor
        Parameters u and v are the output of `linalg.svd`, with matching inner
        dimensions so one can compute `np.dot(u * s, v)`.
    v : torch.Tensor
        Parameters u and v are the output of `linalg.svd`, with matching inner
        dimensions so one can compute `np.dot(u * s, v)`.
        The input v should really be called vt to be consistent with scipy's output.
    u_based_decision : bool, default=True
        If True, use the columns of u as the basis for sign flipping.
        Otherwise, use the rows of v. The choice of which variable to base the
        decision on is generally algorithm dependent.

    Returns
    -------
    u_adjusted : torch.Tensor
        Array u with adjusted columns and the same dimensions as u.

    v_adjusted : torch.Tensor
        Array v with adjusted rows and the same dimensions as v.
    """
    if u_based_decision:
        # columns of u, rows of v
        max_abs_cols = torch.argmax(torch.abs(u), dim=0)
        signs = torch.sign(u[max_abs_cols, torch.arange(u.shape[1])])
        u *= signs
        v *= signs[:, None]
    else:
        # rows of v, columns of u
        max_abs_rows = torch.argmax(torch.abs(v), dim=1)
        signs = torch.sign(v[torch.arange(v.shape[0]), max_abs_rows])
        u *= signs
        v *= signs[:, None]
    return u, v
