import numpy as np


class ConfigParser:
    params = {"results_file_path": "",
              "rotation_axis": "",
              "rotation_angle": 0.,
              "init_vector": np.array([]),
              "bohr_magneton": 0.,
              "magnetic_filed": 0.,
              "time_tc": 0.,
              "dg_factor": 0.}

    numerical_params = {"guess_pulse": None,
                        "guess_rotation": None,
                        "cost_function": "",
                        "hessian_diagonal": False,
                        "number_of_iterations": 100,
                        "time_of_termination": 0.,
                        }

    @classmethod
    def get_params(cls):
        cfg_file = open("./configFile.txt")
        for line in cfg_file:
            param_name, param_value = line.split("=")
            param_value = param_value.strip()
            cls.params[param_name] = param_value
            if line == "############## NUMERICAL SETTINGS ##############":
                break
        return

    @classmethod
    def get_numerical_params(cls):
        cfg_file = open("./configFile.txt")
        found_abstract = False
        for line in cfg_file:
            if '############## NUMERICAL SETTINGS ##############' in line:
                found_abstract = True
            if found_abstract:
                param_name, param_value = line.split("=")
                param_value = param_value.strip()
                cls.numerical_params[param_name] = param_value
        return

    @classmethod
    def convert_data(cls):
        print("[INFO]: Data converted")
        cls.params["magnetic_filed"] = float(cls.params["magnetic_filed"])
        cls.params["target_pulse"] = float(cls.params["target_pulse"])
        cls.params["dg_factor"] = float(cls.params["dg_factor"])
        cls.params["time_tc"] = float(cls.params["time_tc"])

        cls.numerical_params["epsilon"] = float(cls.params["epsilon"])
        cls.numerical_params["learning_decrementation"] = float(cls.params["learning_decrementation"]) \
            if cls.params["learning_decrementation"] != "None" else "None"
        cls.numerical_params["learning_incrementation"] = float(cls.params["learning_incrementation"]) \
            if cls.params["learning_incrementation"] != "None" else "None"
        cls.numerical_params["learning_rate"] = float(cls.params["learning_rate"]) \
            if cls.params["learning_rate"] != "None" else "None"
        cls.numerical_params["guess_pulse"] = float(cls.params["guess_pulse"]) \
            if cls.params["guess_pulse"] != "None" else "None"
        cls.numerical_params["guess_rotation"] = float(cls.params["guess_rotation"]) \
            if cls.params["guess_rotation"] != "None" else "None"
        cls.numerical_params["hessian_diagonal"] = bool(cls.params["hessian_diagonal"]) \
            if cls.params["hessian_diagonal"] != "None" else "None"
        cls.numerical_params["time_of_termination"] = bool(cls.params["time_of_termination"]) \
            if cls.params["time_of_termination"] != "None" else "None "
        cls.numerical_params["number_of_iterations"] = bool(cls.params["number_of_iterations"]) \
            if cls.params["number_of_iterations"] != "None" else "None"
        cls.__parse_string()
        return

    @classmethod
    def __parse_string(cls):
        cls.params["init_vector"] = np.fromstring(cls.params["init_vector"], dtype=float, sep=',')
        return
