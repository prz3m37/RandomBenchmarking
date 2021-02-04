from BlochSolver.solvers import quantum_grape as qp


class Solvers:

    def __init__(self):
        self.solver = None

    def __del__(self):
        del self.solver

    def get_solver(self, solver_type: str = None, algorithm_type: str = None, **kwargs):
        if solver_type == "GRAPE":
            self.solver = qp.QuantumGrape()
            return self.solver.grape_solver(algorithm_type, **kwargs)
        else:
            print("[ERROR]: Please choose quantum solver and algorithm type!")
            return
