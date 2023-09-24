from typing import Union, Sequence, Tuple, List

import numpy as np
import torch
from falkon import FalkonOptions
from falkon.center_selection import UniformSelector, FixedSelector
from falkon.kernels import GaussianKernel
from kooplearn import PrincipalComponent, ReducedRank
from kooplearn.kernels import RBF
from matplotlib import pyplot as plt

from nyskoop.estimators import (
    KoopmanNystromKrr,
    ExactKoopmanNystromPcr, RandomizedKoopmanNystromPcr,
    ExactKoopmanNystromRrr, RandomizedKoopmanNystromRrr,
    ScalableKAF,
)

__all__ = [
    "train_nystrom_est", "train_kaf_est", "train_full_est", "train_est",
    "set_matplotlib_rc", "gen_lagged", "nrmse", "rmse",
    "calc_lagged_num_steps", "IBM_COLORS", "implied_timescales",
]


IBM_COLORS = [
    "#FE6100",
    "#648FFF",
    "#FFB000",
    "#785EF0",
]


def train_est(X, Y, kind, **kwargs):
    if kind.startswith("full"):
        return train_full_est(X, Y, kind=kind.split("_")[1], **kwargs)
    elif kind.startswith("kaf"):
        return train_kaf_est(X, Y, kind=kind.split("_")[1], **kwargs)
    return train_nystrom_est(X, Y, kind=kind, **kwargs)


def train_full_est(X, Y, kind="pcr", **kind_kwargs):
    if isinstance(X, torch.Tensor):
        X = X.numpy()
    if isinstance(Y, torch.Tensor):
        Y = Y.numpy()

    if isinstance(kind_kwargs["kernel"], GaussianKernel):
        kernel = RBF(
            length_scale=kind_kwargs.pop("kernel").sigma.numpy().astype(X.dtype)
        )
    else:
        raise ValueError("full estimators only accept the Gaussian kernel")
    rank = kind_kwargs.pop("num_components")
    if kind == "pcr":
        est = PrincipalComponent(kernel=kernel, rank=rank, backend='numpy')
    elif kind == "rrr":
        tikhonov_reg = kind_kwargs.pop("penalty")
        est = ReducedRank(kernel=kernel, rank=rank, tikhonov_reg=tikhonov_reg, backend='numpy')
    else:
        raise ValueError(kind)
    est.fit(X, Y)
    return est


def train_nystrom_est(X, Y, M, kernel, kind="krr", **kind_kwargs):
    """Train Nystrom Koopman estimator.
    """
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
    if isinstance(Y, np.ndarray):
        Y = torch.from_numpy(Y)
    cs = UniformSelector(random_gen=torch.randint(0, 1000, (1,)).item(), num_centers=M)
    # Choose centers for X, Y differently
    # cs_x = cs.select(X.contiguous(), None)
    # cs_y = cs.select(Y.contiguous(), None)
    # Choose the same centers for X, Y
    cs_x, cs_y = cs.select(X.contiguous(), Y.contiguous())
    cs_fixed = FixedSelector(cs_x.contiguous(), cs_y.contiguous())
    if kind == "krr":
        est = KoopmanNystromKrr(kernel=kernel, center_selection=cs_fixed, **kind_kwargs)
    elif kind == "pcr":
        est = ExactKoopmanNystromPcr(kernel=kernel, center_selection=cs_fixed, **kind_kwargs)
    elif kind == "fastpcr":
        est = RandomizedKoopmanNystromPcr(kernel=kernel, center_selection=cs_fixed, **kind_kwargs)
    elif kind == "rrr":
        est = ExactKoopmanNystromRrr(kernel=kernel, center_selection=cs_fixed, **kind_kwargs)
    elif kind == "fastrrr":
        est = RandomizedKoopmanNystromRrr(kernel=kernel, center_selection=cs_fixed, **kind_kwargs)
    else:
        raise ValueError(kind)
    est.fit(X, Y)
    return est


def train_kaf_est(X, Y, sigma, kind, **kwargs):
    """Train StreamingKAF (or the exact version) algorithm from Giannakis, Tropp
    """
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
    if isinstance(Y, np.ndarray):
        Y = torch.from_numpy(Y)
    opt = FalkonOptions(use_cpu=True)
    if kind == "scalable":
        est = ScalableKAF(k_bandwidth=1 / (2 * sigma ** 2), d=X.shape[1], **kwargs)
    else:
        raise ValueError(kind)
    est.fit(X, Y)
    return est


def calc_lagged_num_steps(n_train: int, n_test: int, num_test_sets: int, q: int) -> int:
    """Compute the number of samples needed for generating lags
    """
    return n_train + num_test_sets * n_test + q * (num_test_sets + 1)


def gen_lagged(full_data: torch.Tensor,
               n_train: int,
               n_test: int,
               num_test_sets: int,
               q: int,
               dtype=torch.float64,
               ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
    """Generate lagged data with multiple test-sets for confidence interval calculations

    Parameters
    ----------
    full_data : full dataset (one long trajectory)
    n_train : number of training set samples
    n_test : number of samples in each test set
    num_test_sets : number of test sets
    q : lag time
    dtype: data-type of the generated data

    Returns
    -------

    """
    assert len(full_data >= n_train + num_test_sets * n_test + q * (num_test_sets + 1))
    X_train = full_data[:n_train].to(dtype=dtype).contiguous()
    Y_train = full_data[q: n_train + q].to(dtype=dtype).contiguous()
    all_test_sets = []
    for i in range(num_test_sets):
        test_set_start = n_train + q * (i + 1) + i * n_test
        X_test = full_data[test_set_start: test_set_start + n_test].to(dtype=dtype).contiguous()
        Y_test = full_data[test_set_start + q: test_set_start + n_test + q].to(dtype=dtype).contiguous()
        all_test_sets.append((X_test, Y_test))
    return X_train, Y_train, all_test_sets


def nrmse(est,
          all_test_sets: Sequence[Tuple[torch.Tensor, torch.Tensor]],
          variable: Union[str, int] = 'all'
          ) -> Tuple[torch.Tensor, torch.Tensor]:
    nrmses = []
    for i, (x_test, y_test) in enumerate(all_test_sets):
        y_pred = est.predict(x_test)
        if isinstance(variable, int):
            y_test = y_test[:, variable]
            y_pred = y_pred[:, variable]
        if isinstance(y_pred, np.ndarray):
            y_pred = torch.from_numpy(y_pred)

        err = torch.square(y_pred - y_test)
        nrmses.append(
            torch.sqrt(torch.mean(err, 0)) / torch.std(y_test, 0)
        )

    nrmses = torch.stack(nrmses, 0)
    return torch.mean(nrmses, 0), torch.std(nrmses, 0)


def rmse(est,
         all_test_sets: Sequence[Tuple[torch.Tensor, torch.Tensor]],
         variable: Union[str, int] = 'all'
         ) -> Tuple[torch.Tensor, torch.Tensor]:
    rmses = []
    for i, (x_test, y_test) in enumerate(all_test_sets):
        y_pred = est.predict(x_test)
        if isinstance(variable, int):
            y_test = y_test[:, variable]
            y_pred = y_pred[:, variable]
        if isinstance(y_pred, np.ndarray):
            y_pred = torch.from_numpy(y_pred)

        err = torch.square(y_pred - y_test)
        rmses.append(
            torch.sqrt(torch.mean(err, 0))
        )

    rmses = torch.stack(rmses, 0)
    return torch.mean(rmses, 0), torch.std(rmses, 0)


def set_matplotlib_rc(small_size=14, medium_size=16, bigger_size=18):
    plt.rc('font', size=small_size)  # controls default text sizes
    plt.rc('axes', titlesize=small_size)  # fontsize of the axes title
    plt.rc('axes', labelsize=medium_size)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=small_size)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=small_size)  # fontsize of the tick labels
    plt.rc('legend', fontsize=small_size)  # legend fontsize
    plt.rc('figure', titlesize=bigger_size)  # fontsize of the figure title
    plt.rc('text', usetex=True)


def implied_timescales(evals, tau):
    return -tau / torch.log(evals)
