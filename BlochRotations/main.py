from BlochSolver.SolversManager import solvers_manager
from BlochSolver.Plotter import bloch_plotter as bs
import numpy as np


def main():
    quantum_solvers = solvers_manager.SolversManager()
    bloch_plotter = bs.BlochPlotter()

    initial_pulses = 0.002 * np.ones(32)
    angles = [np.pi / 2]
    axes = ["x"]
    initial_state = np.array([1, 0])
    ideal_state, pulses = quantum_solvers.get_solver(solver_type="GRAPE",
                                                     algorithm_type="default",
                                                     results_path="./",
                                                     initial_pulses=initial_pulses,
                                                     angles=angles,
                                                     axes=axes,
                                                     initial_state=initial_state)

    bloch_plotter.plot(plot_type="evolution", pulses_final=pulses, init_state=initial_state,
                       target_state=ideal_state)

    bloch_plotter.plot(plot_type="evolution", pulses_final=pulses, init_state=ideal_state,
                       target_state=ideal_state)

    del quantum_solvers
    return


if __name__ == '__main__':
    main()
