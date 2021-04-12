import datetime
import os

from BlochSolver.Utils import config_parser, settings


class Utils:

    def __init__(self):
        pass

    def __del__(self):
        pass

    cfg_parser = config_parser.ConfigParser
    log_file = None
    results_file = None
    __result_file_path = None

    @classmethod
    def initialize_utilities(cls, results_path: str = "./"):
        cls.__result_file_path = results_path
        cls.cfg_parser = config_parser.ConfigParser
        cls.cfg_parser.get_params(results_path)
        cls.cfg_parser.get_numerical_params(results_path)
        cls.cfg_parser.convert_data()
        cls.__create_results_dir()
        cls.set_numerical_params()
        cls.set_physical_params()
        return

    @classmethod
    def set_physical_params(cls):
        settings.settings["magnetic_field"] = cls.cfg_parser.params["magnetic_field"]
        settings.settings["dg_factor"] = cls.cfg_parser.params["dg_factor"]
        settings.settings["time_tc"] = cls.cfg_parser.params["time_tc"]
        settings.settings["pulse_time"] = cls.cfg_parser.params["pulse_time"]
        return

    @classmethod
    def set_numerical_params(cls):
        settings.numerical_settings["number_of_iterations"] = cls.cfg_parser.numerical_params["number_of_iterations"]
        settings.numerical_settings["learning_incrementation"] = \
            cls.cfg_parser.numerical_params["learning_incrementation"]
        settings.numerical_settings["learning_decrementation"] = \
            cls.cfg_parser.numerical_params["learning_decrementation"]
        settings.numerical_settings["learning_rate"] = cls.cfg_parser.numerical_params["learning_rate"]
        settings.numerical_settings["error"] = cls.cfg_parser.numerical_params["error"]
        settings.numerical_settings["operator_error"] = cls.cfg_parser.numerical_params["operator_error"]
        settings.numerical_settings["e_min"] = cls.cfg_parser.numerical_params["e_min"]
        settings.numerical_settings["e_max"] = cls.cfg_parser.numerical_params["e_max"]
        settings.numerical_settings["identities"] = cls.cfg_parser.numerical_params["identities"]
        return

    @classmethod
    def get_png_name(cls, name: str):
        return cls.__result_file_path + str(cls.__get_current_time()) + "_" + name + ".pdf"

    @staticmethod
    def __get_current_time():
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @classmethod
    def __create_results_dir(cls):
        if not os.path.exists(cls.__result_file_path + 'bloch_solver_results/'):
            os.makedirs(cls.__result_file_path + 'bloch_solver_results/')
        cls.__result_file_path = cls.__result_file_path + 'bloch_solver_results/'
        return
