import numpy as np


class RotationHandler:

    def __init__(self, theta: float, phi: float):
        self.__phi = phi
        self.__theta = theta

    def __del__(self):
        pass

    def __get_rotation_vectors(self):
        vec_1 = np.array([0, np.cos(self.__phi). np.sin(self.__phi)])
        vec_2 = np.array([0, np.sin(self.__theta). np.cos(self.__theta)])
        return vec_1, vec_2

    @staticmethod
    def __get_rotation_matrix(vector_coordinates, angle):
        x, y, z = vector_coordinates
        sin, cos = np.sin(angle), np.cos(angle)
        rotation_matrix = np.array([[x**2 * (1 - cos) + cos, x*y * (1 - cos) - z * sin, x*z * (1 - cos) + y * sin],
                                    [x*y * (1 - cos) + z * sin, y**2 * (1 - cos) + cos, z*y * (1 - cos) - x * sin],
                                    [x*z * (1 - cos) - y * sin, y*z * (1 - cos) + x * sin, z**2 * (1 - cos) + cos]])
        return rotation_matrix

    def __get_joined_rotation_matrix(self):
        return
