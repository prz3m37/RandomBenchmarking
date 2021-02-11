from BlochSolver.SolversManager import solvers_manager
from BlochSolver.Plotter import bloch_plotter as bs
import numpy as np


def main():
    quantum_solvers = solvers_manager.Solvers()
    bloch_plotter = bs.BlochPlotter()

    initial_pulses = 0.001 * np.ones(32)
    angles = [np.pi/2]
    axes = ["x"]
    initial_state = np.array([1, 0])
    target_state, pulses = quantum_solvers.get_solver(solver_type="GRAPE",
                                                      algorithm_type="default",
                                                      results_path="./",
                                                      initial_pulses=initial_pulses,
                                                      angles=angles,
                                                      axes=axes,
                                                      initial_state=initial_state)
    del quantum_solvers
    bloch_plotter.plot(plot_type="evolution", pulses_final=pulses, init_state=initial_state,
                       target_state=target_state)
    # bloch_plotter.plot(plot_type="evolution", pulses_final=initial_pulses, init_state=initial_state,
    #                    target_state=target_state)
    bloch_plotter.plot(plot_type="pulses", pulses_final=pulses, pulses_init=initial_pulses)
    return


if __name__ == '__main__':
    main()
