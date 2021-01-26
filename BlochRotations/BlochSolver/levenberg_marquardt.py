from Rotations import rotation_handler as rh
from BlochSolver import cost_functions as cf
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
        self.__cf = cf.CostFunctions
        self.__settings_init = si.SettingsInitializer()
        self.__learning_rate = self.__numerical_settings["learning_rate"]
        self.__idn = np.identity(2, dtype=complex)
        nm.NumericalMethods.load_numerical_settings(self.__numerical_settings)

    def __del__(self):
        del self.__settings_init

    def get_lma_bloch(self):
        return

    # TODO Think how to deal with it !
    def __spike_out_of_local_minima(self):
        return

    def __update_learning_rate(self, fidelity_i, fidelity_i1):
        if fidelity_i < fidelity_i1:
            self.__learning_rate *= self.__numerical_settings["learning_decrementation"]
        elif fidelity_i > fidelity_i1:
            self.__learning_rate *= self.__numerical_settings["learning_decrementation"]
        else:
            return
        return

    def __check_matrix_cost_function(self, error):
        error_rate = self.__numerical_settings["error"]
        if (np.abs(np.real(error)) <= error_rate) and (np.abs(np.imag(error)) <= error_rate):
            return True
        else:
            return False

    def __check_fidelity_cost_function(self, fidelity):
        if fidelity <= self.__numerical_settings["error"]:
            return True
        else:
            return False

    def __check_termination_conditions(self, time_elapsed, iteration):
        if time_elapsed >= self.__numerical_settings["time_of_termination"]:
            utils.Utils.save_log("[INFO]: Time exceeded")
            return True
        elif iteration == self.__numerical_settings["number_of_iterations"]:
            utils.Utils.save_log("[INFO]: Number of iterations exceeded")
            return True
        else:
            return False
