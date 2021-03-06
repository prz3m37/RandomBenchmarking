import numpy as np

from BlochSolver.Utils import settings


class SettingsInitializer:

    def __init__(self):
        self.__num_settings = settings.numerical_settings
        self.__settings = settings.settings
        self.__initialize_numerical_settings()
        # self.__save_numerical_settings()
        # self.__save_physical_params()

    def __del__(self):
        pass

    def __initialize_numerical_settings(self):
        self.__set_max_iteration()
        self.__set_init_learning_rate()
        self.__set_learning_incrementation()
        self.__set_learning_decrementation()
        self.__set_termination_error()
        self.__set_termination_down_error()
        self.__set_const_pulses_level()
        self.__set_pulse_ranges()
        return

    def __set_pulse_ranges(self):
        if self.__num_settings["e_max"] is None:
            self.__num_settings["e_max"] = 0.004
        if self.__num_settings["e_min"] is None:
            self.__num_settings["e_min"] = 0.
        return

    def __set_init_learning_rate(self):
        if self.__num_settings["learning_rate"] is None:
            self.__num_settings["learning_rate"] = 1.
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
            self.__num_settings["error"] = 0.999
        return

    def __set_termination_down_error(self):
        if self.__num_settings["operator_error"] is None:
            self.__num_settings["operator_error"] = 1.999
        return

    def __set_const_pulses_level(self):
        if self.__num_settings["identities"] is None:
            self.__num_settings["identities"] = 0
        return

    def __set_max_iteration(self):
        if self.__num_settings["number_of_iterations"] is None:
            self.__num_settings["number_of_iterations"] = np.inf
        return
