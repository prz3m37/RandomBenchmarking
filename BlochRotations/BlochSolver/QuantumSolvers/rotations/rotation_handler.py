import numpy as np
from BlochSolver.Utils import settings as s
from BlochSolver.QuantumSolvers.numerics import numerical_methods as nm


class RotationHandler:
    idn = np.identity(2)

    @classmethod
    def get_pulse_detuning(cls, pulse: float):
        return np.sqrt((0.25 * pulse ** 2) + (2 * s.settings["time_tc"] ** 2)) - 0.5 * pulse

    @classmethod
    def get_pulse_detunings(cls, pulses: np.array):
        return np.array([cls.get_pulse_detuning(pulse) for pulse in pulses])

    @classmethod
    def __get_pulse_arg(cls, pulse_detuning: float):
        return (2 * s.settings["time_tc"] ** 2 - pulse_detuning ** 2) / pulse_detuning

    @classmethod
    def get_pulse_args(cls, pulse_detunings: np.array):
        return np.array([cls.__get_pulse_arg(pulse_detuning) for pulse_detuning in pulse_detunings])

    @classmethod
    def __get_pulse_arg(cls, pulse_detuning: float):
        return (2 * s.settings["time_tc"] ** 2 - pulse_detuning ** 2) / pulse_detuning

    # TODO: Apply Daniel filter function
    @classmethod
    def __filter(cls, pulse: float):
        pulse_prime = pulse
        return pulse_prime

    @classmethod
    def apply_filter_to_pulse(cls, guess_pulse: np.array):
        return np.fromiter((cls.__filter(pulse) for pulse in guess_pulse), np.float)

    @classmethod
    def __get_rotation_operator(cls, pulse: float):
        j_f = cls.get_pulse_detuning(pulse)
        f_term = s.settings["dg_factor"] * s.settings["bohr_magneton"] * s.settings["magnetic_field"]
        omega = np.sqrt(j_f ** 2 + f_term ** 2)
        phi = (omega * s.settings["pulse_time"]) / (2 * s.settings["h_bar"])
        alpha = -j_f / omega
        beta = f_term / omega
        return np.array([[np.cos(phi) + 1j * alpha * np.sin(phi), -1j * beta * np.sin(phi)],
                         [-1j * beta * np.sin(phi), np.cos(phi) - 1j * alpha * np.sin(phi)]])

    @classmethod
    def get_control_hamiltonian(cls):
        return cls.__get_pauli_z()

    @classmethod
    def get_rotation_operators(cls, pulses: np.array):
        return np.array([cls.__get_rotation_operator(pulse) for pulse in pulses])

    @classmethod
    def get_evolution(cls, pulse_sequence: np.array):
        # print(pulse_sequence.shape)
        if pulse_sequence.shape[0] == 1:
            return pulse_sequence[0]
        else:
            return np.linalg.multi_dot(pulse_sequence)

    @classmethod
    def get_step_density_operator(cls, pulse_operators: np.array, init_state: np.array = None):
        evolution = cls.get_evolution(pulse_operators)
        if init_state is None:
            return evolution
        else:
            step_state = cls.get_dot_product(evolution, init_state)
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
        ideal_state = cls.get_dot_product(target_rotation, init_state)
        return target_rotation, nm.NumericalMethods.get_density_operator(ideal_state), ideal_state

    @classmethod
    def __get_rotation_sequence(cls, angles: np.array, axes: np.array):
        return np.array([cls.__get_rotation_matrix(alpha=alpha, axis=axis) for alpha, axis in zip(angles, axes)])

    @classmethod
    def get_dot_product(cls, evolution_operator: np.array, init_state: np.array):
        return np.dot(evolution_operator, init_state)

    @staticmethod
    def get_state_stat(evolution_operator: np.array, init_state: np.array):
        return np.dot(evolution_operator, init_state)

    @staticmethod
    def __get_x_rotation(alpha: float):
        return np.array([[np.cos(alpha / 2), -1j * np.sin(alpha / 2)],
                         [-1j * np.sin(alpha / 2), np.cos(alpha / 2)]])

    @staticmethod
    def __get_y_rotation(alpha: float):
        return np.array([[np.cos(alpha / 2), - np.sin(alpha / 2)],
                         [np.sin(alpha / 2), np.cos(alpha / 2)]])

    @staticmethod
    def __get_z_rotation(alpha: float):
        return np.array([[np.exp(-1j * alpha / 2), 0],
                         [0, np.exp(1j * alpha / 2)]])

    @staticmethod
    def __get_pauli_x():
        return np.array([[0, 1], [1, 0]])

    @staticmethod
    def __get_pauli_y():
        return np.array([[0, -1j], [1j, 0]])

    @staticmethod
    def __get_pauli_z():
        return np.array([[1, 0], [0, -1]])
