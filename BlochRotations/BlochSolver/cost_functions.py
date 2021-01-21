import numpy as np


class CostFunctions:
    pulse_state = None

    @classmethod
    def fidelity_cost_function(cls, pulse_matrix_func: np.array, final_state: np.array):
        cls.pulse_state = np.matmul(pulse_matrix_func, cls.pulse_state)
        return np.abs(np.dot(cls.pulse_state, final_state))**2

    @classmethod
    def matrix_cost_function(cls, pulse_matrix_func, real_matrix_func, **kwargs):
        return np.sum(pulse_matrix_func(**kwargs) - real_matrix_func(**kwargs))
