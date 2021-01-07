from Utils import config_parser

import datetime
import numpy as np


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
        cls.log_file = None
        cls.results_file = None
        cls.__result_file_path = cls.cfg_parser.params["results_file_path"]
        cls.__create_log_file()
        cls.__create_results_file()
        cls.__convert_data()
        return

    @classmethod
    def release_utilities(cls):
        cls.__close_results_file()
        cls.__close_log_file()
        return

    @classmethod
    def get_rotation_params(cls):
        theta, phi = cls.cfg_parser.params["theta"], cls.cfg_parser.params["phi"]
        return theta, phi

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
    def __convert_data(cls):
        cls.save_log("[INFO]: Data converted")
        cls.cfg_parser.params["theta"] = float(cls.cfg_parser.params["theta"])
        cls.cfg_parser.params["phi"] = float(cls.cfg_parser.params["phi"])
        cls.cfg_parser.params["init_vector"] = np.fromstring(cls.cfg_parser.params["init_vector"],
                                                             dtype=int, sep=',')
        cls.cfg_parser.params["final_vector"] = np.fromstring(cls.cfg_parser.params["final_vector"],
                                                              dtype=int, sep=',')
        return

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

