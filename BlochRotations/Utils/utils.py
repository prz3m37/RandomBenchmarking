from Utils import config_parser
import settings
import datetime


class Utils:

    def __init__(self):
        pass

    def __del__(self):
        pass

    cfg_parser = config_parser.ConfigParser
    cfg_parser.get_params()
    log_file = None
    results_file = None
    __result_file_path = cfg_parser.params["results_file_path"]

    @classmethod
    def initialize_utilities(cls):
        cls.cfg_parser = config_parser.ConfigParser
        cls.cfg_parser.get_params()
        cls.cfg_parser.convert_data()
        cls.log_file = None
        cls.results_file = None
        cls.__result_file_path = cls.cfg_parser.params["results_file_path"]
        cls.__create_log_file()
        cls.__create_results_file()
        return

    @classmethod
    def release_utilities(cls):
        cls.__close_results_file()
        cls.__close_log_file()
        return

    @classmethod
    def get_rotation_params(cls):
        settings.settings["magnetic_filed"] = cls.cfg_parser.params["magnetic_filed"]
        settings.settings["rotation_axis"] = cls.cfg_parser.params["rotation_axis"]
        settings.settings["dg_factor"] = cls.cfg_parser.params["dg_factor"]
        settings.settings["time_tc"] = cls.cfg_parser.params["time_tc"]
        return

    @classmethod
    def save_log(cls, message: str):
        msg = "[" + cls.__get_current_time() + "]" + message + "\n"
        cls.log_file.write(msg)
        return

    @classmethod
    def save_result(cls, result: str):
        res = "[" + cls.__get_current_time() + "]" + result + "\n"
        cls.results_file.write(res)
        return

    @staticmethod
    def __get_current_time():
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @classmethod
    def __create_results_file(cls):
        results_file_name = cls.__result_file_path + str(cls.__get_current_time()) + "_RB_RESULTS_FILE.txt"
        cls.results_file = open(results_file_name, "a")
        cls.save_log("[INFO]: Results file created")
        return

    @classmethod
    def __close_results_file(cls):
        cls.save_log("[INFO]: Results file closed")
        cls.results_file.close()
        return

    @classmethod
    def __create_log_file(cls):
        log_file_name = str(cls.__get_current_time()) + "_RB_LOG_FILE.txt"
        print(log_file_name)
        cls.log_file = open(log_file_name, "a")
        return

    @classmethod
    def __close_log_file(cls):
        cls.save_log("[INFO]: Log file closed")
        cls.log_file.close()
        return

