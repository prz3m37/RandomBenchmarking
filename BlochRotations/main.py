from BlochSolver.SolversManager import solvers
import numpy as np


def main():
    quantum_solvers = solvers.Solvers()

    initial_pulses = None
    angles = None
    axes = None
    initial_state = None
    quantum_solvers.get_solver(solver_type="GRAPE",
                               algorithm_type="default",
                               results_path="./",
                               initial_pulses=initial_pulses,
                               angles=angles,
                               axes=axes,
                               initial_state=initial_state)
    return


if __name__ == '__main__':
    main()
