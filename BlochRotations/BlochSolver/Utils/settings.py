import numpy as np

settings = {"results_file_path": "",
            "bohr_magneton": 5.7883818012 * 10**(-5),
            "magnetic_field": 0.,
            "time_tc": 0.,
            "pulse_time": 0,
            "dg_factor": 0,
            "h_bar": 4.135667696 * 10**(-15) / (2*np.pi)}

numerical_settings = {"number_of_iterations": None,
                      "learning_rate": None,
                      "learning_incrementation": None,
                      "learning_decrementation": None,
                      "error": None,
                      "operator_error": None,
                      "e_min": None,
                      "e_max": None,
                      "identities": None
                      }
