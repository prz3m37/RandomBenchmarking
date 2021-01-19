import numpy as np
from Utils import settings as s


# TODO: Maybe different instances will be needed for multiprocessing
class RotationHandler:
    theta = None

    @classmethod
    def __get_rotation_angle(cls, epsilon: float):
        j_function = np.sqrt(0.25 * epsilon ** 2 + 2 * s.settings["time_tc"]) - 0.5 * epsilon
        cls.theta = np.arctan(s.settings["magnetic_filed"] * s.settings["dg_factor"] *
                              s.settings["bohr_magneton"] / j_function)
        cls.theta = np.deg2rad(cls.theta)
        return

    @classmethod
    def __get_rotation_vectors(cls, epsilon: float):
        cls.__get_rotation_angle(epsilon)
        rotation_vector = np.array([0, np.cos(cls.theta), np.sin(cls.theta)])
        return rotation_vector

    @classmethod
    def get_pulse_rotation_matrix(cls, phi: float, epsilon: float):
        nx, ny, nz = cls.__get_rotation_vectors(epsilon)
        rot_matrix = np.array([[np.cos(phi) + 1j * nz * np.sin(phi), (1j * nx + ny) * np.sin(phi)],
                               [(1j * nx - ny) * np.sin(phi), np.cos(phi) - 1j * nz * np.sin(phi)]])
        return rot_matrix

    @classmethod
    def get_final_matrix(cls, alpha: float):
        if s.settings["rotation_axis"] == "x":
            return cls.__get_x_rotation(alpha)
        elif s.settings["rotation_axis"] == "y":
            return cls.__get_y_rotation(alpha)
        else:
            return cls.__get_z_rotation(alpha)

    # TODO: Check if matmul stuff is ok to use
    @classmethod
    def get_state(cls, rotation_matrix, init_state: np.array, **kwargs):
        return np.matmul(rotation_matrix(**kwargs), init_state)

    @staticmethod
    def __get_x_rotation(alpha: float):
        return np.array([[np.cos(alpha / 2), -1j * np.sin(alpha/2)], [-1j * np.sin(alpha/2), np.cos(alpha / 2)]])

    @staticmethod
    def __get_y_rotation(alpha: float):
        return np.array([[np.cos(alpha / 2), - np.sin(alpha/2)], [-np.sin(alpha/2), np.cos(alpha / 2)]])

    @staticmethod
    def __get_z_rotation(alpha: float):
        return np.array([[np.exp(1j * alpha/2), 0], [0, np.exp(1j * alpha/2)]])

    @classmethod
    def __check_theta_value(cls):
        if np.rad2deg(cls.theta) <= 0:
            cls.theta += np.deg2rad(0.1)
        elif np.rad2deg(cls.theta) >= 90:
            cls.theta -= np.deg2rad(0.1)
        return
