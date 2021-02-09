from BlochSolver.Utils import settings, config_parser
import datetime
import os

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
    def initialize_utilities(cls, solver_type: str, results_path: str = "./"):
        cls.__result_file_path = results_path

        cls.cfg_parser = config_parser.ConfigParser
        cls.cfg_parser.get_params(results_path)
        cls.cfg_parser.get_numerical_params(results_path)
        cls.cfg_parser.convert_data()
        cls.__create_results_dir()
        cls.set_numerical_params()
        cls.set_physical_params()
        cls.__create_log_file(solver_type)
        cls.__create_results_file(solver_type)
        return

    @classmethod
    def release_utilities(cls):
        cls.__close_results_file()
        cls.__close_log_file()
        return

    @classmethod
    def set_physical_params(cls):
        settings.settings["magnetic_field"] = cls.cfg_parser.params["magnetic_field"]
        settings.settings["dg_factor"] = cls.cfg_parser.params["dg_factor"]
        settings.settings["time_tc"] = cls.cfg_parser.params["time_tc"]
        return

    @classmethod
    def set_numerical_params(cls):
        settings.numerical_settings["number_of_iterations"] = cls.cfg_parser.numerical_params["number_of_iterations"]
        settings.numerical_settings["time_of_termination"] = cls.cfg_parser.numerical_params["time_of_termination"]
        settings.numerical_settings["learning_incrementation"] = \
            cls.cfg_parser.numerical_params["learning_incrementation"]
        settings.numerical_settings["learning_decrementation"] = \
            cls.cfg_parser.numerical_params["learning_decrementation"]
        settings.numerical_settings["learning_rate"] = cls.cfg_parser.numerical_params["learning_rate"]
        settings.numerical_settings["error"] = cls.cfg_parser.numerical_params["error"]
        settings.numerical_settings["down_error"] = cls.cfg_parser.numerical_params["down_error"]
        return

    @classmethod
    def save_log(cls, message: str):
        msg = "[" + cls.__get_current_time() + "]" + message + "\n"
        cls.log_file.write(msg)
        return

    @classmethod
    def save_result(cls, result: str):
        res = "[" + cls.__get_current_time() + "] " + result + "\n"
        cls.results_file.write(res)
        return

    @staticmethod
    def __get_current_time():
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @classmethod
    def __create_results_file(cls, solver_type: str):
        results_file_name = cls.__result_file_path + \
                            str(cls.__get_current_time()) + "_" + solver_type + "_BS_RESULTS_FILE.txt"
        cls.results_file = open(results_file_name, "a")
        cls.save_log("[INFO]: Results file created")
        return

    @classmethod
    def __close_results_file(cls):
        cls.save_log("[INFO]: Results file closed")
        cls.results_file.close()
        return

    @classmethod
    def __create_log_file(cls, solver_type: str):
        log_file_name = cls.__result_file_path + \
                        str(cls.__get_current_time()) + "_" + solver_type + "_BS_LOG_FILE.txt"
        cls.log_file = open(log_file_name, "a")
        return

    @classmethod
    def __close_log_file(cls):
        cls.save_log("[INFO]: Log file closed")
        cls.log_file.close()
        return

    @classmethod
    def __create_results_dir(cls):
        if not os.path.exists(cls.__result_file_path + 'bloch_solver_results/'):
            os.makedirs(cls.__result_file_path + 'bloch_solver_results/')
        cls.__result_file_path = cls.__result_file_path + 'bloch_solver_results/'
        return
