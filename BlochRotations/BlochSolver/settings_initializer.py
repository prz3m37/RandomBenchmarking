from Utils import settings
import numpy as np


class SettingsInitializer:

    def __init__(self):
        self.__num_settings = settings.numerical_settings
        self.__initialize_numerical_settings()

    def __del__(self):
        pass

    def __initialize_numerical_settings(self):
        self.__set_init_pulse()
        self.__set_init_angle()
        self.__set_termination_time()
        self.__set_max_iteration()
        self.__set_init_learning_rate()
        return

    def __set_init_pulse(self):
        if self.__num_settings["guess_pulse"] == "None":
            self.__num_settings["guess_pulse"] = np.random.uniform(low=0.1, high=1.)
        return

    def __set_init_learning_rate(self):
        if self.__num_settings["learning_rate"] == "None":
            self.__num_settings["learning_rate"] = 10.
        return

    def __set_init_angle(self):
        if self.__num_settings["guess_rotation"] == "None":
            self.__num_settings["guess_rotation"] = np.random.uniform(low=0., high=180.)
        return

    def __set_termination_time(self):
        if self.__num_settings["time_of_termination"] == "None":
            self.__num_settings["time_of_termination"] = np.inf
        return

    def __set_max_iteration(self):
        if self.__num_settings["number_of_iterations"] == "None":
            self.__num_settings["number_of_iterations"] = 100.
        return


