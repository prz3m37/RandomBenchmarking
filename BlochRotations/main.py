from BlochSolver.SolversManager import solvers_manager
from BlochSolver.Plotter import bloch_plotter as bs
# from BlochSolver.Perturbations.filters import Filters
import numpy as np


def main():
    quantum_solvers = solvers_manager.SolversManager()
    bloch_plotter = bs.BlochPlotter()

    initial_pulses = np.ones(32)*0.002 #np.random.uniform(0.001, 0.004, 32)
    angles = [np.pi/2]
    axes = ["x"]
    initial_state = np.array([1, 0])

    # signal_f, signal = Filters.get_low_pass_pulses(initial_pulses, 1.6e-9, 2e-9, 6, 64)
    ideal_state, pulses = quantum_solvers.get_solver(solver_type="GRAPE",
                                                     algorithm_type="default",
                                                     penalty=True,
                                                     results_path="./",
                                                     initial_pulses=initial_pulses,
                                                     angles=angles,
                                                     axes=axes,
                                                     initial_state=initial_state)
    bloch_plotter.plot(plot_type="evolution", pulses_final=pulses, init_state=initial_state, target_state=ideal_state)
    bloch_plotter.plot(plot_type="pulses", pulses_final=pulses, pulses_init=initial_pulses)
    # bloch_plotter.plot(plot_type="filters", signal_filtered=signal_f, signal=signal, duration=64, pulse_time=1.6e-9)

    del quantum_solvers
    return


if __name__ == '__main__':
    main()
