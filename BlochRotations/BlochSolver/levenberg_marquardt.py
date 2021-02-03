from Rotations import rotation_handler as rh
from BlochSolver import solver_controller as cf
from BlochSolver import numerical_methods as nm
from BlochSolver import settings_initializer as si
from Utils import settings
from Utils import utils
import numpy as np
import time


class LevenbergMarquardtSolver(rh.RotationHandler, nm.NumericalMethods):

    def __init__(self):
        self.__settings = settings.settings
        self.__numerical_settings = settings.numerical_settings
        self.__settings_init = si.SettingsInitializer()
        self.__learning_rate = self.__numerical_settings["learning_rate"]
        self.__idn = np.identity(2, dtype=complex)
        nm.NumericalMethods.load_numerical_settings(self.__numerical_settings)

    def __del__(self):
        del self.__settings_init

    def get_lma_bloch(self):
        return



