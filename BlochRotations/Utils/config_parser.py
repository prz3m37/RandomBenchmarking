import numpy as np


class ConfigParser:
    params = {"results_file_path": "",
              "rotation_axis": "",
              "init_vector": np.array([]),
              "target_pulse:": 0.,
              "magnetic_filed": 0.,
              "time_tc": 0.,
              "dg_factor": 0.}

    @classmethod
    def get_params(cls):
        cfg_file = open("./configFile.txt")
        for line in cfg_file:
            param_name, param_value = line.split("=")
            param_value = param_value.strip()
            cls.params[param_name] = param_value
        return

    @classmethod
    def convert_data(cls):
        print("[INFO]: Data converted")
        cls.params["magnetic_filed"] = float(cls.params["magnetic_filed"])
        cls.params["target_pulse"] = float(cls.params["target_pulse"])
        cls.params["dg_factor"] = float(cls.params["dg_factor"])
        cls.params["time_tc"] = float(cls.params["time_tc"])
        cls.__parse_string()
        return

    @classmethod
    def __parse_string(cls):
        cls.params["init_vector"] = np.fromstring(cls.params["init_vector"], dtype=float, sep=',')
        return
