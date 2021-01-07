from Rotations import rotation_handler
import numpy as np


class AnglesSolver(rotation_handler.RotationHandler):

    def __init__(self, theta: float, phi: float):
        rotation_handler.RotationHandler.__init__(self, theta, phi)
        self.__error = None

    def __del__(self):
        del rotation_handler.RotationHandler

    def find_angles(self):
        return

    def get_step_error(self):
        return

