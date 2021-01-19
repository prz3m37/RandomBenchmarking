from Rotations import rotation_handler
import numerical_methods as nm
from Utils import settings


class LevenbergMarquardtSolver(rotation_handler.RotationHandler, nm.NumericalMethods):

    def __init__(self):
        self.__rotations = rotation_handler.RotationHandler
        self.__settings = settings.settings
        self.__num_settings = settings.numerical_settings

    def __get_lma(self):
        return

    def compare_final_results(self):
        return
