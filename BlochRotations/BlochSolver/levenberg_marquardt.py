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

    def get_lma_matrix_func(self):

        self.__apply_initial_pulse()
        iteration = 0
        x_init = np.array([self.__numerical_settings["guess_pulse"],
                           self.__numerical_settings["guess_rotation"]])
        time_start = time.time()
        while True:
            print("x_init", x_init)
            hessian = self.get_hessian_matrix(f=self.__cf.matrix_cost_function,
                                              x0=x_init[0],
                                              y0=x_init[1],
                                              )
            newton_part = self.get_inverse_matrix(hessian + self.__learning_rate * self.__idn)
            gradient = self.get_gradient(f=self.__cf.matrix_cost_function, x0=x_init[0], y0=x_init[1])
            x_step = x_init - np.matmul(newton_part, gradient)

            error_init = self.__cf.matrix_cost_function(x=x_init[0], y=x_init[1])
            error_step = self.__cf.matrix_cost_function(x=x_step[0], y=x_step[1])
            result = "ITERATION " + str(iteration) + " ERROR_INIT " + str(error_init) + " ERROR_STEP " \
                     + str(error_step) + " X_INIT " + str(x_init) + " X_STEP " + str(x_step) + " LEARNING_RATE " \
                     + str(self.__learning_rate)
            utils.Utils.save_result(result)

            print("x_step", x_step)

            if self.__check_matrix_cost_function(error_init):
                utils.Utils.save_log("[INFO]: Matrix cost function condition fulfilled")
                break
            else:
                self.__update_learning_rate(fidelity_i=error_init, fidelity_i1=error_step)

            x_init = x_step
            iteration += 1
            time_elapsed = time_start - time.time()

            if self.__check_termination_conditions(time_elapsed, iteration):
                break
        return

    def __get_lma_matrix_func_hessian(self):
        return

    def __get_lma_fidelity_func(self):
        return

    def __get_lma_fidelity_func_hessian(self):
        return

    def __apply_initial_pulse(self):
        self.__cf.pulse_state = self.__settings['init_vector']
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
