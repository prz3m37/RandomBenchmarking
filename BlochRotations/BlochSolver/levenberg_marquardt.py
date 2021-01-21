from Rotations import rotation_handler
import cost_functions as cf
import numerical_methods as nm
import settings_initializer as si
from Utils import settings
import numpy as np
import time


class LevenbergMarquardtSolver(rotation_handler.RotationHandler, nm.NumericalMethods):

    def __init__(self):
        self.__settings = settings.settings
        self.__numerical_settings = settings.numerical_settings
        self.__cost_fun = cf.CostFunctions
        self.__rotations = rotation_handler.RotationHandler
        self.__settings_init = si.SettingsInitializer()

    def __del__(self):
        del self.__settings_init

    def __get_lma(self):

        self.__apply_initial_pulse()
        iteration = 0
        time_start = time.time()
        while True:

            """LMA Algorithm"""

            iteration += 1
            time_elapsed = time_start - time.time()
            if self.__check_termination_conditions(time_elapsed, iteration):
                break
        return

    def __apply_initial_pulse(self):
        self.__cost_fun.pulse_state = self.__settings['init_vector']
        return

    def __update_learning_rate(self):
        return

    def __check_matrix_cost_function(self, **kwargs):
        difference = self.__cost_fun.matrix_cost_function(**kwargs)
        epsilon = self.__numerical_settings["epsilon"]
        if np.real(difference) <= epsilon and np.imag(difference) <= epsilon:
            return True
        else:
            return False

    def __check_fidelity_cost_function(self, **kwargs):
        fidelity = self.__cost_fun.fidelity_cost_function(**kwargs)
        if fidelity <= self.__numerical_settings["epsilon"]:
            return True
        else:
            return False

    def __check_termination_conditions(self, time_elapsed, iteration):
        if time_elapsed >= self.__numerical_settings["time_of_termination"]:
            return True
        elif iteration == self.__numerical_settings["number_of_iterations"]:
            return True
        else:
            return False
