import numpy as np
from Rotations import rotation_handler


class CostFunctions:
    pulse_state = None
    rh = rotation_handler.RotationHandler

    @classmethod
    def fidelity_cost_function(cls, pulse_matrix_func: np.array, final_state: np.array):
        cls.pulse_state = np.matmul(pulse_matrix_func, cls.pulse_state)
        return np.abs(np.dot(cls.pulse_state, final_state)) ** 2

    @classmethod
    def matrix_cost_function(cls):
        return np.sum(getattr(cls.rh, "get_pulse_rotation_matrix") - getattr(cls.rh, "get_final_matrix")())
