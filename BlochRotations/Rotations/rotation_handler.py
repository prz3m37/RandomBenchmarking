import numpy as np
from Utils import settings as s


# TODO: Maybe different instances will be needed for multiprocessing
class RotationHandler:
    theta = None
    pulse_state = None

    @classmethod
    def __get_deviation_angle(cls, pulse: float):
        j_function = np.sqrt(0.25 * pulse ** 2 + 2 * s.settings["time_tc"]) - 0.5 * pulse
        cls.theta = np.arctan(s.settings["magnetic_field"] * s.settings["dg_factor"] *
                              s.settings["bohr_magneton"] / j_function)
        cls.theta = np.deg2rad(np.real(cls.theta))
        return

    @classmethod
    def __get_rotation_angle(cls, pulse: float):
        phi = pulse
        return phi

    @classmethod
    def __get_rotation_vectors(cls, pulse: float):
        cls.__get_deviation_angle(pulse)
        rotation_vector = np.array([0, np.cos(cls.theta), np.sin(cls.theta)])
        return rotation_vector

    @classmethod
    def __filter(cls, pulse: float):
        pulse_prime = pulse
        return pulse_prime

    # TODO: Apply filter to epsilon_i so to get epsilon_i prime
    @classmethod
    def __apply_filter_to_pulse(cls, guess_pulse: np.array):
        return np.fromiter((cls.__filter(pulse) for pulse in guess_pulse), np.complex)

    @classmethod
    def __apply_initial_pulse(cls, guess_pulse: np.array):
        filtered_pulses = cls.__apply_filter_to_pulse(guess_pulse)
        pulse_matrices = np.fromiter((cls.get_pulse_rotation_matrix(filtered_pulse) for
                                      filtered_pulse in filtered_pulses), np.complex)
        return pulse_matrices

    @classmethod
    def __get_reference_matrix(cls):
        return

    @classmethod
    def get_pulse_rotation_matrix(cls, pulse: float):
        nx, ny, nz = cls.__get_rotation_vectors(pulse)
        phi = cls.__get_rotation_angle(pulse)
        rot_matrix = np.array([[np.cos(phi / 2) + 1j * nz * np.sin(phi / 2), (1j * nx + ny) * np.sin(phi / 2)],
                               [(1j * nx - ny) * np.sin(phi / 2), np.cos(phi / 2) - 1j * nz * np.sin(phi / 2)]])
        return rot_matrix

    @classmethod
    def get_final_matrix(cls):
        alpha = s.settings["rotation_angle"]
        if s.settings["rotation_axis"] == "x":
            return cls.__get_x_rotation(alpha)
        elif s.settings["rotation_axis"] == "y":
            return cls.__get_y_rotation(alpha)
        else:
            return cls.__get_z_rotation(alpha)

    @classmethod
    def get_state(cls, rotation_matrix, init_state: np.array, **kwargs):
        return np.matmul(rotation_matrix(**kwargs), init_state)

    @staticmethod
    def __get_x_rotation(alpha: float):
        return np.array([[np.cos(alpha / 2), -1j * np.sin(alpha / 2)], [-1j * np.sin(alpha / 2), np.cos(alpha / 2)]])

    @staticmethod
    def __get_y_rotation(alpha: float):
        return np.array([[np.cos(alpha / 2), - np.sin(alpha / 2)], [-np.sin(alpha / 2), np.cos(alpha / 2)]])

    @staticmethod
    def __get_z_rotation(alpha: float):
        return np.array([[np.exp(1j * alpha / 2), 0], [0, np.exp(1j * alpha / 2)]])

    @classmethod
    def __check_theta_value(cls):
        if np.rad2deg(cls.theta) <= 0:
            cls.theta += np.deg2rad(0.1)
        elif np.rad2deg(cls.theta) >= 90:
            cls.theta -= np.deg2rad(0.1)
        return
