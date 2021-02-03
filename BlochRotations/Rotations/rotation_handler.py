import numpy as np
from Utils import settings as s
from BlochSolver import numerical_methods as nm


# TODO: Maybe different instances will be needed for multiprocessing
class RotationHandler:

    @classmethod
    def __get_pulse_detuning(cls, pulse: float):
        return np.sqrt(0.25 * pulse ** 2 + 2 * s.settings["time_tc"]) - 0.5 * pulse

    # TODO: Apply Daniel filter function
    @classmethod
    def __filter(cls, pulse: float):
        pulse_prime = pulse
        return pulse_prime

    @classmethod
    def apply_filter_to_pulse(cls, guess_pulse: np.array):
        return np.fromiter((cls.__filter(pulse) for pulse in guess_pulse), np.float)

    @classmethod
    def __get_evolution_operator(cls, pulse: float):
        arg = (s.settings["time_tc"] / s.settings["h_bar"]) * \
              (s.settings["dg_factor"] * s.settings["bohr_magneton"] * s.settings["magnetic_field"]) * 0.5
        j_arg = (s.settings["time_tc"] / s.settings["h_bar"]) * cls.__get_pulse_detuning(pulse)
        return np.array([[np.cos(arg), -1j * np.sin(arg)],
                         [-1j * np.sin(arg) * np.exp(2j * j_arg), np.cos(arg) * np.exp(2j * j_arg)]])

    @classmethod
    def get_pulse_operators(cls, pulses: np.array):
        return np.fromiter((cls.__get_evolution_operator(pulse) for pulse in pulses), np.complex)

    @classmethod
    def get_evolution(cls, pulse_sequence: np.array):
        return np.linalg.multi_dot(pulse_sequence)

    @classmethod
    def __get_rotation_matrix(cls, alpha: float, axis: str):
        if axis == "x":
            return cls.__get_x_rotation(alpha)
        elif axis == "y":
            return cls.__get_y_rotation(alpha)
        else:
            return cls.__get_z_rotation(alpha)

    @classmethod
    def get_target_operator(cls, angles: np.array, axes: np.array, init_state: np.array):
        angles, axes = angles[::-1], axes[::-1]
        rotations_sequence = cls.__get_rotation_sequence(angles, axes)
        target_rotation = cls.get_evolution(rotations_sequence)
        ideal_state = cls.get_state(target_rotation, init_state)
        return nm.NumericalMethods.get_density_operator(ideal_state)

    @classmethod
    def __get_deviation_angle(cls):
        return

    @classmethod
    def __get_rotation_sequence(cls, angles: np.array, axes: np.array):
        return np.fromiter((cls.__get_rotation_matrix(alpha, axis) for alpha, axis in zip(angles, axes)), np.complex)

    @classmethod
    def get_state(cls, evolution_operator: np.array, init_state: np.array):
        return np.dot(evolution_operator, init_state)

    @staticmethod
    def __get_x_rotation(alpha: float):
        return np.array([[np.cos(alpha / 2), -1j * np.sin(alpha / 2)], [-1j * np.sin(alpha / 2), np.cos(alpha / 2)]])

    @staticmethod
    def __get_y_rotation(alpha: float):
        return np.array([[np.cos(alpha / 2), - np.sin(alpha / 2)], [-np.sin(alpha / 2), np.cos(alpha / 2)]])

    @staticmethod
    def __get_z_rotation(alpha: float):
        return np.array([[np.exp(1j * alpha / 2), 0], [0, np.exp(1j * alpha / 2)]])
