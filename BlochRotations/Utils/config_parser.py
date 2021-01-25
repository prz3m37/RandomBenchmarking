import numpy as np


class ConfigParser:
    params = {"results_file_path": "",
              "rotation_axis": "",
              "rotation_angle": 0.,
              "init_vector": np.array([]),
              "magnetic_field": 0.,
              "time_tc": 0.,
              "dg_factor": 0.}

    numerical_params = {"guess_pulse": None,
                        "guess_rotation": None,
                        "hessian_diagonal": None,
                        "number_of_iterations": None,
                        "time_of_termination": None,
                        "learning_rate": None,
                        "learning_incrementation": None,
                        "learning_decrementation": None,
                        "error": None,
                        "hx": None,
                        "hy": None
                        }

    @classmethod
    def get_params(cls):
        cfg_file = open("./configFile.txt", "r")
        for line in cfg_file:
            if not line.startswith("#"):
                param_name, param_value = line.split("=")
                param_value = param_value.strip()
                cls.params[param_name] = param_value
            if line == "############## NUMERICAL SETTINGS ##############\n":
                break
        return

    @classmethod
    def get_numerical_params(cls):
        cfg_file = open("./configFile.txt")
        found_abstract = False
        for line in cfg_file:
            if line == "############## NUMERICAL SETTINGS ##############\n":
                found_abstract = True
            if found_abstract and not line.startswith("#"):
                param_name, param_value = line.split("=")
                param_value = param_value.strip()
                cls.numerical_params[param_name] = param_value
            if line == "############## TYPES ##############\n":
                break
        return

    @classmethod
    def convert_data(cls):
        print("[INFO]: Data converted")
        cls.params["magnetic_field"] = float(cls.params["magnetic_field"])
        cls.params["dg_factor"] = float(cls.params["dg_factor"])
        cls.params["time_tc"] = float(cls.params["time_tc"])
        cls.params["rotation_angle"] = float(cls.params["rotation_angle"])

        cls.numerical_params["error"] = float(cls.numerical_params["error"])
        cls.numerical_params["hx"] = float(cls.numerical_params["hx"]) \
            if cls.numerical_params["hx"] != "" else None
        cls.numerical_params["hy"] = float(cls.numerical_params["hy"]) \
            if cls.numerical_params["hy"] != "" else None
        cls.numerical_params["learning_decrementation"] = float(cls.numerical_params["learning_decrementation"]) \
            if cls.numerical_params["learning_decrementation"] != "" else None
        cls.numerical_params["learning_incrementation"] = float(cls.numerical_params["learning_incrementation"]) \
            if cls.numerical_params["learning_incrementation"] != "" else None
        cls.numerical_params["learning_rate"] = float(cls.numerical_params["learning_rate"]) \
            if cls.numerical_params["learning_rate"] != "" else None
        cls.numerical_params["guess_pulse"] = float(cls.numerical_params["guess_pulse"]) \
            if cls.numerical_params["guess_pulse"] != "" else None
        cls.numerical_params["guess_rotation"] = float(cls.numerical_params["guess_rotation"]) \
            if cls.numerical_params["guess_rotation"] != "" else None
        cls.numerical_params["hessian_diagonal"] = bool(cls.numerical_params["hessian_diagonal"]) \
            if cls.numerical_params["hessian_diagonal"] != "" else None
        cls.numerical_params["time_of_termination"] = bool(cls.numerical_params["time_of_termination"]) \
            if cls.numerical_params["time_of_termination"] != "" else None
        cls.numerical_params["number_of_iterations"] = bool(cls.numerical_params["number_of_iterations"]) \
            if cls.numerical_params["number_of_iterations"] != "" else None
        cls.__parse_string()
        return

    @classmethod
    def __parse_string(cls):
        cls.params["init_vector"] = np.fromstring(cls.params["init_vector"], dtype=float, sep=',')
        return
