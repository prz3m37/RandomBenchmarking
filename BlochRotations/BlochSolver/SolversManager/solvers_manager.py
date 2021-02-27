import time

import numpy as np

from BlochSolver.QuantumSolvers.solvers import quantum_grape as qp
from BlochSolver.SolversManager import settings_initializer as si
from BlochSolver.Utils import utils


class SolversManager:

    def __init__(self):
        self.__solver = None
        self.__settings_init = None
        self.__time_start = time.time()

    def __del__(self):
        del self.__solver
        del self.__settings_init
        self.__get_time()

    def get_solver(self, solver_type: str = "GRAPE", algorithm_type: str = None,
                   penalty: bool = False, results_path: str = None, **kwargs):
        self.__get_info(solver_type, algorithm_type, penalty, **kwargs)
        utils.Utils.initialize_utilities(results_path)
        self.__settings_init = si.SettingsInitializer()
        if solver_type == "GRAPE":
            self.__solver = qp.QuantumGrape()
            return self.__solver.grape_solver(algorithm_type, penalty, **kwargs)
        else:
            print("[ERROR]: Please choose quantum solver and algorithm type!")
            return

    @staticmethod
    def __get_info(solver_type: str, algorithm_type: str, penalty: bool, initial_pulses: np.array, angles: np.array,
                   axes: np.array, initial_state: np.array):
        np.set_printoptions(linewidth=np.inf)
        if algorithm_type is None:
            algorithm_type = "default"
        print("########################################################################",
              "\n                          --", solver_type, "SOLVER -- "
              , "\n########################################################################"
              , " \n ---> Algorithm type:  ", algorithm_type
              , " \n ---> Penalty gradient:", str(penalty)
              , " \n ---> Target rotation: ", np.rad2deg(angles), "around: ", axes
              , "\n ---> Initial state:   ",
              "[", np.round(initial_state[0], 3), np.round(initial_state[1], 3), "]"
              , "\n ---> Initial pulses:  ", "[", initial_pulses[0], "...", initial_pulses[0], "]")
        return

    def __get_time(self):
        print("########################################################################")
        print(" ---> Simulation executed in: ", np.round(time.time() - self.__time_start, 3), " sec")
        print("########################################################################")
        return
