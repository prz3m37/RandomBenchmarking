from BlochSolver.SolversManager import solvers_manager
from BlochSolver.Plotter import bloch_plotter as bs
from BlochSolver.Perturbations.filters import Filters
from BlochSolver.QuantumSolvers.rotations import rotation_handler
from BlochSolver.QuantumSolvers.numerics import numerical_methods
from BlochSolver.Utils.utils import Utils
import numpy as np


# INFO: Perturbation algorithms are using filter-refilter stuff to avoid problems with raising edge

# Here you can see example of code, solver options to try :
#   1. default or None - basic GRAPE algorithm
#   2. unitary - GRAPE for pulses without raising time, unitary evolution
#   3. perturbation grape - GRAPE for pulses with raising time, non-unitary evolution
#   4. perturbation unitary - GRAPE for pulses with no raising time, unitary evolution

def main():
    bloch_plotter = bs.BlochPlotter()
    quantum_solvers = solvers_manager.SolversManager()

    angles = [np.pi / 2]
    axes = ["x"]
    initial_state =np.array([1,0])
    granulation = 8
    cut_off_time = 0.4e-9

    initial_pulses = np.ones(32)* 0.002 #np.random.uniform(0.001 ,0.006, 32)#
    ideal_state, pulses = quantum_solvers.get_solver(solver_type="GRAPE",
                                                        algorithm_type="perturbation unitary",
                                                        penalty=True,
                                                        results_path="./",
                                                        initial_pulses=initial_pulses,
                                                        angles=angles,
                                                        axes=axes,
                                                        initial_state=initial_state,
                                                        cut_off_time=cut_off_time,
                                                        granulation=granulation)


    bloch_plotter.plot(plot_type="pulses", pulses_final=pulses, pulses_init=initial_pulses)
    bloch_plotter.plot(plot_type="evolution", pulses_final=pulses, init_state=initial_state, target_state=ideal_state)

    # Here before plotting you have to put filtered signals because plotting function with granulation option will calculate the effective pulses
    # bloch_plotter.plot(plot_type="evolution", pulses_final=pulses, init_state=ideal_state, granulation=granulation, target_state=ideal_state)

    del quantum_solvers

    return

if __name__ == '__main__':
    main()