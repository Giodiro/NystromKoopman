import numpy as np
import scipy
from scipy.integrate import quad
from scipy.stats import beta
from scipy.special import binom
from scipy.stats.sampling import NumericalInversePolynomial
from tqdm import tqdm


class CosineDistribution:
    def __init__(self, N):
        self.N = N
        self.C_N = np.pi / scipy.special.beta(N // 2 + 0.5, 0.5)

    def pdf(self, x):
        return self.C_N * ((np.cos(np.pi * x)) ** self.N)


class LogisticMap:
    def __init__(self, N=None):
        self._noisy = False
        self.ndim = 1
        if N is not None:
            # Noisy case
            self._noisy = True
            self.N = N
            self.C_N = np.pi / scipy.special.beta(N // 2 + 0.5, 0.5)
            self._evals, self._PF_largest_evec, self._Koop_evecs = self._transfer_matrix_eig_process()
            self._urng = np.random.default_rng()
            self._rng = NumericalInversePolynomial(self, domain=(0, 1), random_state=self._urng)
            self._noise_dist = CosineDistribution(N)
            self._noise_rng = NumericalInversePolynomial(
                self._noise_dist, domain=(-0.5, 0.5), mode=0, random_state=self._urng)
        else:
            # Noiseless case
            pass

    def pdf(self, x):
        if self._noisy:
            if np.isscalar(x):
                y = 0
            else:
                y = np.zeros(x.shape)
            for i in range(self.N + 1):
                y += self._feature(x, i) * self._PF_largest_evec[i]
            return np.abs(y)
        else:
            return beta(0.5, 0.5).pdf(x)

    def rvs(self, size=1):
        if np.isscalar(size):
            size = (size, 1)
        if self._noisy:
            return self._rng.rvs(size)
        else:
            return beta(0.5, 0.5).rvs(size=size)

    def noise(self, size=1):
        if np.isscalar(size):
            size = (size, 1)
        if self._noisy:
            return self._noise_rng.rvs(size)
        else:
            raise ValueError("This method not needed for noiseless case")

    def sample(self, size=1, iid=True):
        if np.isscalar(size):
            size = (size, 1)
        if iid:
            X = self.rvs(size)
            Y = self.map(X, noisy=self._noisy)
        else:
            _raw = np.zeros((size[0] + 1, size[1]))
            _raw[0] = self.rvs(size[1])
            for i in range(1, size[0] + 1):
                _raw[i] = self.map(_raw[i - 1], noisy=self._noisy)
            X = _raw[:-1]
            Y = _raw[1:]
        return X, Y

    def _transfer_matrix(self):
        if self._noisy:
            N = self.N
            eps = 1e-10
            A = np.zeros((N + 1, N + 1))
            for i in tqdm(range(N + 1), desc='Init: Transfer matrix'):
                for j in range(N + 1):
                    alpha = lambda x: self._feature(self.map(x), i)
                    beta = lambda x: self._feature(x, j)
                    f = lambda x: alpha(x) * beta(x)
                    q = quad(f, 0, 1, epsabs=eps, epsrel=eps)
                    A[i, j] = q[0]
            return A
        else:
            raise ValueError("This method not needed for noiseless case")

    def _transfer_matrix_eig_process(self):
        if self._noisy:
            A = self._transfer_matrix()
            self._A = A
            ev, lv, rv = scipy.linalg.eig(A, left=True, right=True)
            invariant_eig_idx = None
            for idx, v in enumerate(ev):
                if np.isreal(v):
                    if np.abs(v - 1) < 1e-10:
                        invariant_eig_idx = idx
                        break
            if invariant_eig_idx is None:
                raise ValueError("No invariant eigenvalue found")
            PF_largest_evec = rv[:, invariant_eig_idx]
            if not np.all(np.isreal(PF_largest_evec)):
                print(
                    f"Largest eigenvector is not real, largest absolute imaginary part is {np.abs(np.imag(PF_largest_evec)).max()}. Forcing it to be real.")
            return ev, np.real(PF_largest_evec), lv

        else:
            raise ValueError("This method not needed for noiseless case")

    def _feature(self, x, i):
        if self._noisy:
            N = self.N
            C_N = self.C_N
            return ((np.sin(np.pi * x)) ** (N - i)) * ((np.cos(np.pi * x)) ** i) * np.sqrt(binom(N, i) * C_N)
        else:
            raise ValueError("This method not needed for noiseless case")

    def map(self, x, noisy=False):
        if noisy:
            y = 4 * x * (1 - x)
            if np.isscalar(x):
                xi = self.noise(1)[0]
            else:
                xi = self.noise(x.shape)
            return np.mod(y + xi, 1)
        else:
            return 4 * x * (1 - x)
