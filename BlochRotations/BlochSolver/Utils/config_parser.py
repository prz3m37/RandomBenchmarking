class ConfigParser:

    params = {"magnetic_field": 0.,
              "time_tc": 0.,
              "dg_factor": 0.}

    numerical_params = {"number_of_iterations": 0.,
                        "time_of_termination": 0.,
                        "learning_rate": 0.,
                        "learning_incrementation": 0.,
                        "learning_decrementation": 0.,
                        "down_error": 0.,
                        "error": 0.,
                        }

    @classmethod
    def get_params(cls, results_path: str):
        cfg_file = open(results_path + "configFile.txt", "r")
        for line in cfg_file:
            if not line.startswith("#"):
                param_name, param_value = line.split("=")
                param_value = param_value.strip()
                cls.params[param_name] = param_value
            if line == "############## NUMERICAL SETTINGS ##############\n":
                break
        return

    @classmethod
    def get_numerical_params(cls, results_path: str):
        cfg_file = open(results_path + "configFile.txt")
        found_abstract = False
        for line in cfg_file:
            if line == "############## NUMERICAL SETTINGS ##############\n":
                found_abstract = True
            if found_abstract and not line.startswith("#"):
                param_name, param_value = line.split("=")
                param_value = param_value.strip()
                cls.numerical_params[param_name] = param_value
            if line == "############## TYPES INFO ##############\n":
                break
        return

    @classmethod
    def convert_data(cls):
        cls.params["magnetic_field"] = float(cls.params["magnetic_field"])
        cls.params["dg_factor"] = float(cls.params["dg_factor"])
        cls.params["time_tc"] = (float(cls.params["time_tc"]) * 10**9 * 6.582119569 * 10**(-16))

        cls.numerical_params["error"] = float(cls.numerical_params["error"])
        cls.numerical_params["down_error"] = float(cls.numerical_params["down_error"])
        cls.numerical_params["learning_decrementation"] = float(cls.numerical_params["learning_decrementation"]) \
            if cls.numerical_params["learning_decrementation"] != "" else None
        cls.numerical_params["learning_incrementation"] = float(cls.numerical_params["learning_incrementation"]) \
            if cls.numerical_params["learning_incrementation"] != "" else None
        cls.numerical_params["learning_rate"] = float(cls.numerical_params["learning_rate"]) \
            if cls.numerical_params["learning_rate"] != "" else None
        cls.numerical_params["time_of_termination"] = float(cls.numerical_params["time_of_termination"]) \
            if cls.numerical_params["time_of_termination"] != "" else None
        cls.numerical_params["number_of_iterations"] = float(cls.numerical_params["number_of_iterations"]) \
            if cls.numerical_params["number_of_iterations"] != "" else None
        return
