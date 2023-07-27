import math

import numpy as np
import sdeint


class MullerBrownPotential:
    def __init__(self, x0, kt):
        self.a = np.array([-1, -1, -6.5, 0.7])
        self.b = np.array([0.0, 0.0, 11.0, 0.6])
        self.c = np.array([-10, -10, -6.5, 0.7])
        self.A = np.array([-200, -100, -170, 15])
        self.X = np.array([1, 0, -0.5, -1])
        self.Y = np.array([0, 0.5, 1.5, 1])
        self.x0 = x0
        self.kt = kt

    def fn(self, x):
        t1 = x[0] - self.X
        t2 = x[1] - self.Y
        tinexp = self.a * (t1 ** 2) + self.b * t1 * t2 + self.c * (t2 ** 2)
        return np.dot(self.A, np.exp(tinexp))

    def det_part(self, x, t):
        """
        A_j * exp[a_j * (x1^2 - 2 * x1 * X_j) + b_j * (x1 * x2 - x1 * Y_j)]

        -> grad inner exp:
            a_j * (2 * x1 - 2 * X_j) + b_j * (x2 - Y_j)
        """
        t1 = x[0] - self.X
        t2 = x[1] - self.Y
        grad_inner_exp_x1 = 2 * self.a * t1 + self.b * t2
        grad_inner_exp_x2 = 2 * self.c * t2 + self.b * t1
        tinexp = np.exp(self.a * (t1 ** 2) + self.b * t1 * t2 + self.c * (t2 ** 2))

        return -np.array([
            np.dot(self.A, tinexp * grad_inner_exp_x1),
            np.dot(self.A, tinexp * grad_inner_exp_x2),
        ]) / self.kt

    def rnd_part(self, x, t):
        return np.diag([math.sqrt(2 * 1e-2), math.sqrt(2 * 1e-2)])

    def solve(self, num_points):
        tspan = np.arange(0, 0.1 * num_points, 0.1)
        result = sdeint.itoint(self.det_part, self.rnd_part, self.x0, tspan)
        return result
