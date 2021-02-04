from BlochSolver.QuantumSolvers.solvers import quantum_grape as qp
from BlochSolver.SolversManager import settings_initializer as si
from BlochSolver.Utils import utils


class Solvers:

    def __init__(self, results_path: str = None):
        self.__solver = None
        self.__settings_init = si.SettingsInitializer()
        utils.Utils.initialize_utilities(results_path)

    def __del__(self):
        del self.__solver
        del self.__settings_init
        utils.Utils.release_utilities()

    def get_solver(self, solver_type: str = None, algorithm_type: str = None, **kwargs):
        if solver_type == "GRAPE":
            self.__solver = qp.QuantumGrape()
            return self.__solver.grape_solver(algorithm_type, **kwargs)
        else:
            print("[ERROR]: Please choose quantum solver and algorithm type!")
            return
