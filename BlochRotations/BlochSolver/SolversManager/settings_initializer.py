from BlochSolver.Utils import settings
import numpy as np


class SettingsInitializer:

    def __init__(self):
        self.__num_settings = settings.numerical_settings
        self.__initialize_numerical_settings()

    def __del__(self):
        pass

    def __initialize_numerical_settings(self):
        self.__set_termination_time()
        self.__set_max_iteration()
        self.__set_init_learning_rate()
        self.__set_learning_incrementation()
        self.__set_learning_decrementation()
        self.__set_termination_error()
        return

    def __set_init_learning_rate(self):
        if self.__num_settings["learning_rate"] is None:
            self.__num_settings["learning_rate"] = 10.
        return

    def __set_learning_incrementation(self):
        if self.__num_settings["learning_incrementation"] is None:
            self.__num_settings["learning_incrementation"] = 2.
        return

    def __set_learning_decrementation(self):
        if self.__num_settings["learning_decrementation"] is None:
            self.__num_settings["learning_decrementation"] = 0.5
        return

    def __set_termination_error(self):
        if self.__num_settings["error"] is None:
            self.__num_settings["error"] = 1e-5
        return

    def __set_termination_time(self):
        if self.__num_settings["time_of_termination"] is None:
            self.__num_settings["time_of_termination"] = np.inf
        return

    def __set_max_iteration(self):
        if self.__num_settings["number_of_iterations"] is None:
            self.__num_settings["number_of_iterations"] = np.inf
        return
