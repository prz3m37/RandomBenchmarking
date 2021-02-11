from BlochSolver.QuantumSolvers.solvers import quantum_grape as qp
from BlochSolver.SolversManager import settings_initializer as si
from BlochSolver.Utils import utils
import numpy as np
import time
class Solvers:

    def __init__(self):
        self.__solver = None
        self.__settings_init = None
        self.__time_start = time.time()

    def __del__(self):
        del self.__solver
        del self.__settings_init
        utils.Utils.release_utilities()
        self.__get_time()

    def get_solver(self, solver_type: str = "GRAPE", algorithm_type: str = None,
                   results_path: str = None, **kwargs):
        self.__get_info(solver_type, **kwargs)
        utils.Utils.initialize_utilities(solver_type, results_path)
        self.__settings_init = si.SettingsInitializer()
        if solver_type == "GRAPE":
            self.__solver = qp.QuantumGrape()
            return self.__solver.grape_solver(algorithm_type, **kwargs)
        else:
            print("[ERROR]: Please choose quantum solver and algorithm type!")
            return

    @staticmethod
    def __get_info(solver_type: str, initial_pulses, angles, axes, initial_state):
        np.set_printoptions(linewidth=np.inf)

        print("############################################",
              "\n ---> Solver --", solver_type, "-- running for: "
              , "\n############################################"
              , " \n ---> Target rotation: ", np.rad2deg(angles), "around: ", axes
              , "\n ---> Initial state:   ",
              "[", np.round(initial_state[0], 3), np.round(initial_state[1], 3), "]"
              , "\n ---> Initial pulses:  ", "[", initial_pulses[0], "...", initial_pulses[0], "]")
        return

    def __get_time(self):
        print("############################################")
        print(" ---> Simulation executed in: ", np.round(time.time() - self.__time_start, 3), " sec")
        print("############################################")
        return
