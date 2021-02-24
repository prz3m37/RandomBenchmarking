from BlochSolver.SolversManager import solvers_manager
from BlochSolver.Plotter import bloch_plotter as bs
import numpy as np
from numpy import savez_compressed

def main():
    quantum_solvers = solvers_manager.SolversManager()
    bloch_plotter = bs.BlochPlotter()

    initial_pulses = np.ones(32) * 0.002 # np.random.uniform(0.0015, 0.004, 32)
    angles = [np.pi/2]
    axes = ["x"]
    initial_state = np.array([1, 0])
    ideal_state, pulses = quantum_solvers.get_solver(solver_type="GRAPE",
                                                     algorithm_type="unitary",
                                                     results_path="./",
                                                     initial_pulses=initial_pulses,
                                                     angles=angles,
                                                     axes=axes,
                                                     initial_state=initial_state)

    data = np.asarray([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    np.savez_compressed('./data_99.npz', data)

    # bloch_plotter.plot(plot_type="evolution", pulses_final=pulses, init_state=initial_state,
    #                    target_state=ideal_state)
    #
    # bloch_plotter.plot(plot_type="pulses", pulses_final=pulses, pulses_init=initial_pulses)

    del quantum_solvers
    return


if __name__ == '__main__':
    main()
