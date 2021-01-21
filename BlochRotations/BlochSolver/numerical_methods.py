import numpy as np


class NumericalMethods:
    hessian = np.zeros(2, 2)
    __variables = ["x", "y"]

    @classmethod
    def __get_derivative(cls, f, x0: float, y0: float, hx: float = 0, hy: float = 0):
        h = hx if hy is 0 else hy
        return (f(x0 + hx, y0 + hy) - f(x0 - hx, y0 + hy)) / (2 * h)

    @classmethod
    def __get_second_derivative(cls, f, x0: float, y0: float, hx: float, hy: float, var):
        dx = hx if var is 'x' else 0
        dy = hy if var is 'y' else 0
        h = hx if var is 'x' else hy
        return (f(x0 + 2 * dx, y0 + 2 * dy) - 2 * f(x0 + dx, y0 + dy) + f(x0, y0)) / (h ** 2)

    @classmethod
    def __get_mixed_derivative(cls, f, x0: float, y0: float, hx: float, hy: float):
        return (f(x0 + hx, y0 + hy) - f(x0 + hx, y0 - hy) - f(x0 - hx, y0 + hy) + f(x0 - hx, y0 - hy)) / (4 * hx * hy)

    @classmethod
    def __get_gradient(cls, f, x0: float, y0: float, hx: float, hy: float):
        data = [[x0, y0, hx, 0], [x0, y0, 0, hy]]
        return np.fromiter((cls.__get_derivative(f, xi, yi, dx, dy) for xi, yi, dx, dy in data), np.float64)

    # TODO: Analytical derivative is calculated by using chain rule.
    @classmethod
    def __get_analytical_derivative(cls, f):
        return

    @classmethod
    def __get_analytical_second_derivative(cls, f):
        return

    @classmethod
    def __get_analytical_mixed_derivative(cls):
        return

    @classmethod
    def __get_analytical_gradient(cls):
        return

    @classmethod
    def __get_hessian_matrix(cls, f, x0: float, y0: float, hx: float, hy: float):
        for i in range(2):
            for j in range(2):
                if i == j:
                    cls.hessian[i, j] = cls.__get_second_derivative(f, x0, y0, hx, hy, cls.__variables[i])
                else:
                    cls.hessian[i, j] = cls.__get_mixed_derivative(f, x0, y0, hx, hy)
        return

    @classmethod
    def __get_diagonal_hessian(cls, matrix: np.array):
        return np.diag(np.diag(matrix))

    @classmethod
    def __get_inverse_matrix(cls, matrix: np.array):
        return np.linalg.inv(matrix)
