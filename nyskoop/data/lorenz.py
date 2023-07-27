import numpy as np
import scipy.integrate


class Lorenz63:
    """
    Object represents atmospheric convection using the non-linear,
    deterministic Lorenz 63 model
    https://github.com/kendixon/Lorenz63/blob/master/Lorenz63.py
    """

    def __init__(self, x, y, z, sigma, rho, beta, dt, burnin=5, l63_version='normal'):
        """
        Initialize Lorenz63 model state (x,y,z), timestep (dt), and parameters
        (sigma, rho, and beta).
        """
        self.initial_state = np.array([x, y, z])
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.dt = dt
        self.burnin = burnin
        self.l63_version = l63_version

    def lorenz63_fn(self):
        sigma = self.sigma
        rho = self.rho
        beta = self.beta
        out = np.empty((3,))
        if self.l63_version == 'normal':
            def inner(t, y):
                out[0] = sigma * (y[1] - y[0])
                out[1] = y[0] * (rho - y[2])
                out[2] = y[0] * y[1] - beta * y[2]
                return out
        elif self.l63_version == 'skaf':
            def inner(t, y):
                out[0] = sigma * (y[1] - y[0])
                out[1] = -y[1] - y[0] * y[2]
                out[2] = -beta * y[2] + y[0] * y[1] - beta * rho
                return out
        else:
            raise ValueError(self.l63_version)
        return inner

    def solve(self, num_points):
        end_t = self.burnin + num_points * self.dt
        sol = scipy.integrate.solve_ivp(
            self.lorenz63_fn(),
            t_span=[0, end_t],
            y0=self.initial_state,
            method='RK45',
            t_eval=np.arange(self.burnin, end_t, self.dt),
            rtol=1e-6,
            atol=1e-8,
        )
        return np.vstack(sol.y).T
