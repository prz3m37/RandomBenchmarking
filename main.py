from BlochSolver.SolversManager import solvers_manager
from BlochSolver.Plotter import bloch_plotter as bs
from BlochSolver.Perturbations.filters import Filters
import numpy as np
import time


def main():
    bloch_plotter = bs.BlochPlotter()
    quantum_solvers = solvers_manager.SolversManager()

    angles = [np.pi / 2]
    axes = ["x"]
    initial_state = np.array([1, 0])
    granulation = 8
    cut_off_time = 0.4e-9
    initial_pulses = np.random.uniform(0.0015, 0.004, 32)
    ideal_state, pulses = quantum_solvers.get_solver(solver_type="GRAPE",
                                                     algorithm_type="perturbation unitary",
                                                     penalty=False,
                                                     results_path="./",
                                                     initial_pulses=initial_pulses,
                                                     angles=angles,
                                                     axes=axes,
                                                     initial_state=initial_state,
                                                     cut_off_time=cut_off_time,
                                                     granulation=granulation)
    del quantum_solvers
    signal_f_init, signal_init = Filters.get_low_pass_pulses(initial_pulses, 1.64e-9, cut_off_time, granulation,
                                                             True)

    bloch_plotter.plot(plot_type="evolution", pulses_final=pulses, init_state=initial_state,
                       target_state=ideal_state,
                       granulation=granulation)
    time.sleep(1)
    bloch_plotter.plot(plot_type="evolution", pulses_final=signal_f_init, init_state=initial_state,
                       target_state=ideal_state,
                       granulation=granulation)

    bloch_plotter.plot(plot_type="pulses", pulses_final=pulses, pulses_init=signal_f_init)
    print()
    return


if __name__ == '__main__':
    main()
