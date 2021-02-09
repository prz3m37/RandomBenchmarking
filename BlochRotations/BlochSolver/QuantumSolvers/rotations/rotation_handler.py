import numpy as np
from BlochSolver.Utils import settings as s
from BlochSolver.QuantumSolvers.numerics import numerical_methods as nm


# TODO: Maybe different instances will be needed for multiprocessing
class RotationHandler:
    idn = np.identity(2)

    @classmethod
    def __get_pulse_detuning(cls, pulse: float):
        return np.sqrt((0.25 * pulse ** 2) + (2 * s.settings["time_tc"]**2)) - 0.5 * pulse

    # TODO: Apply Daniel filter function
    @classmethod
    def __filter(cls, pulse: float):
        pulse_prime = pulse
        return pulse_prime

    @classmethod
    def apply_filter_to_pulse(cls, guess_pulse: np.array):
        return np.fromiter((cls.__filter(pulse) for pulse in guess_pulse), np.float)

    # TODO: Verify rotation operator !
    @classmethod
    def __get_evolution_operator(cls, pulse: float):
        arg = (0.41 * 10**(-9) / s.settings["h_bar"]) * \
              (s.settings["dg_factor"] * s.settings["bohr_magneton"] * s.settings["magnetic_field"])* 0.5
        j_arg = (0.41 * 10**(-9) / s.settings["h_bar"]) * cls.__get_pulse_detuning(pulse)
        return np.array([[np.cos(arg * 0.5) * np.exp(-1j * j_arg * 0.5), -1j * np.sin(arg)],
                         [-1j * np.sin(arg), np.cos(arg * 0.5) * np.exp(1j * j_arg * 0.5)]])

    @classmethod
    def get_pulse_operators(cls, pulses: np.array):
        return np.array([cls.__get_evolution_operator(pulse) for pulse in pulses])

    @classmethod
    def get_evolution(cls, pulse_sequence: np.array):
        if pulse_sequence.shape[0] == 1:
            return pulse_sequence[0]
        else:
            return np.linalg.multi_dot(pulse_sequence)

    @classmethod
    def get_step_density_operator(cls, init_state: np.array, pulse_operators: np.array):
        evolution = cls.get_evolution(pulse_operators)
        step_state = cls.get_state(evolution, init_state)
        return nm.NumericalMethods.get_density_operator(step_state)

    @classmethod
    def __get_rotation_matrix(cls, alpha: float, axis: str):
        if axis == "x":
            return cls.__get_x_rotation(alpha)
        elif axis == "y":
            return cls.__get_y_rotation(alpha)
        else:
            return cls.__get_z_rotation(alpha)

    @classmethod
    def get_target_state(cls, angles: np.array, axes: np.array, init_state: np.array):
        angles, axes = angles[::-1], axes[::-1]
        rotations_sequence = cls.__get_rotation_sequence(angles, axes)
        target_rotation = cls.get_evolution(rotations_sequence)
        ideal_state = cls.get_state(target_rotation, init_state)
        return ideal_state, nm.NumericalMethods.get_density_operator(ideal_state)

    @classmethod
    def __get_rotation_sequence(cls, angles: np.array, axes: np.array):
        return np.array([cls.__get_rotation_matrix(alpha=alpha, axis=axis) for alpha, axis in zip(angles, axes)])

    @classmethod
    def get_state(cls, evolution_operator: np.array, init_state: np.array):
        # evolution_operator = np.kron(evolution_operator, cls.idn)
        return np.dot(evolution_operator, init_state)

    @staticmethod
    def get_state_stat(evolution_operator: np.array, init_state: np.array):
        return np.dot(evolution_operator, init_state)

    @staticmethod
    def __get_x_rotation(alpha: float):
        return np.array([[np.cos(alpha / 2), -1j * np.sin(alpha / 2)], [-1j * np.sin(alpha / 2), np.cos(alpha / 2)]])

    @staticmethod
    def __get_y_rotation(alpha: float):
        return np.array([[np.cos(alpha / 2), - np.sin(alpha / 2)],
                         [np.sin(alpha / 2), np.cos(alpha / 2)]])

    @staticmethod
    def __get_z_rotation(alpha: float):
        return np.array([[np.exp(-1j * alpha / 2), 0],
                         [0, np.exp(1j * alpha / 2)]])
