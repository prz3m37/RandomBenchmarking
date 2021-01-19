from Rotations import rotation_handler
import numerical_methods as nm


class LevenbergMarquardtSolver(rotation_handler.RotationHandler, nm.NumericalMethods):

    def __init__(self):
        self.__rotations = rotation_handler.RotationHandler

    def __get_lma(self):
        return
