class ConfigParser:

    params = {"theta": 0.,
              "phi": 0.,
              "results_file_path": "",
              "rotation_axes": ""}

    def __init__(self):
        pass

    def __del__(self):
        pass

    @classmethod
    def get_params(cls):
        cfg_file = open("./configFile.txt")
        for line in cfg_file:
            param_name, param_value = line.split("=")
            param_value = param_value.strip()
            cls.params[param_name] = param_value
        return
