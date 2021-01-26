import numpy as np
from Rotations import rotation_handler


class CostFunctions:
    pulse_state = None
    rh = rotation_handler.RotationHandler

    @classmethod
    def test_cost_function(cls, x, y):
        return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2

    @classmethod
    def get_fidelity(cls, x):
        cls.pulse_state = np.matmul(cls.rh.get_pulse_rotation_matrix(x), cls.pulse_state)
        final_state = np.matmul(cls.rh.get_final_matrix(), cls.pulse_state)
        return np.abs(np.dot(cls.pulse_state, final_state)) ** 2

