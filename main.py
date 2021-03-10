from BlochSolver.SolversManager import solvers_manager
from BlochSolver.Plotter import bloch_plotter as bs
from BlochSolver.Perturbations.filters import Filters
from BlochSolver.QuantumSolvers.rotations import rotation_handler
from BlochSolver.QuantumSolvers.numerics import numerical_methods
import numpy as np
import time
import glob
import matplotlib.pyplot as plt
from BlochSolver.Utils.utils import Utils


# TODO : Third condition checking the ideal one with filtered one (convolution)
# TODO : Additional condition which links individual level of granulated pulses,
#  for each 32*granulation pulse
# TODO : Run 1000 initial pulses and filter them all and look for the smoothest curve - simulate experiment.


def get_fidelity(signal, granulation, init_state, target_operator):
    if granulation is not None:
        n = int(len(signal) / granulation)
        rotation_operators = rotation_handler.RotationHandler.get_perturbation_rotation_operators(
            signal.reshape(n, granulation), granulation)
    else:
        rotation_operators = rotation_handler.RotationHandler.get_rotation_operators(signal)

    density_operator = rotation_handler.RotationHandler.get_step_density_operator(pulse_operators=rotation_operators,
                                                                                  init_state=init_state)
    fidelity = np.real(numerical_methods.NumericalMethods.get_matrix_product(target_operator, density_operator))
    return fidelity


def process_proper_pulses(granulation, cut_time, error):
    angles = [np.pi / 2]
    axes = ["x"]
    initial_state = np.array([1, 0])
    Utils.initialize_utilities('./')
    target_prop, target_operator, target_state = rotation_handler.RotationHandler.get_target_state(angles,
                                                                                                   axes, initial_state)
    pulse_files = glob.glob("/home/pzielins/Documents/repositories/IST/RandomBenchmarking/pulses_sweep/*.npy")

    files_amount = len(pulse_files)
    target_pulses, fidelities_target, fidelities_f = [], [], []
    for i, pulse_file in enumerate(pulse_files):

        print("DONE:", np.round(i / files_amount, 1) * 100, "\r")
        pulses = np.load(pulse_file)
        signal_f = Filters.get_low_pass_pulses(pulses, 1.64e-9, cut_time, granulation, False)

        # fidelity = get_fidelity(pulses[::-1], None, initial_state, target_operator)
        fidelity_f = get_fidelity(signal_f[::-1], granulation, initial_state, target_operator)
        fidelities_f.append(fidelity_f)

        if fidelity_f >= error:
            target_pulses.append(pulses)
            fidelities_target.append(fidelity_f)

    avg_fidelity_target = np.mean(fidelities_target)
    avg_fidelity_f = np.mean(fidelities_f)
    acc = np.round(len(target_pulses) / len(pulse_files), 3)
    
    target_pulses = np.array(target_pulses)
    max_values = np.amax(target_pulses, axis=1)
    min_values = np.amin(target_pulses, axis=1)

    plt.figure(figsize=(15, 10))
    plt.axhline(y=avg_fidelity_target, color="orange", linestyle='--', label="Mean of target fidelity pulses")
    plt.axhline(y=error, color="black", linestyle='-', label="0.999 fidelity")
    plt.axhline(y=0.9999, color="grey", linestyle='-', label="0.9999 fidelity")

    plt.plot(fidelities_f, ".", color="red", label="Fidelity of filtered pulse sequence")
    plt.axhline(y=avg_fidelity_f, color="blue", linestyle='--', label="Mean fidelity of all filtered pulses")
    plt.ylim(0.990, 1.000001)

    plt.legend()
    plt.title("Pulse sweep;  cut error: " + str(error) + "  ; accuracy:  " + str(acc))
    plt.xlabel("Pulse number")
    plt.ylabel("Fidelity of filtered pulses")
    plt.savefig("/home/pzielins/Desktop/fidelity_granularity_" +str(granulation))

    plt.figure(figsize=(15, 10))
    plt.plot(max_values, ".", color="red", label="Max values")
    plt.plot(min_values, ".", color="blue", label="Min values")

    plt.axhline(y=0.004, color="black", linestyle='--', label="0.004 ev")
    plt.axhline(y=0.0015, color="grey", linestyle='--', label="0.0015 ev")
    plt.title("Pulses extremal values above " + str(error) + " fidelity")
    plt.xlabel("N")
    plt.ylabel("Amplitude eV")
    plt.legend()
    plt.savefig("/home/pzielins/Desktop/min_max_granularity_" + str(granulation))

    return


def main():
    # bloch_plotter = bs.BlochPlotter()
    # quantum_solvers = solvers_manager.SolversManager()

    angles = [np.pi / 2]
    axes = ["x"]
    initial_state = np.array([1, 0])
    # granulation = 8
    # cut_off_time = 0.4e-9
    for i in range(10000):
        print("----------------------- PULSE ITERATION : ", i, " ----------------------- \n")
        quantum_solvers = solvers_manager.SolversManager()
        initial_pulses = np.random.uniform(0.0015, 0.004, 32)
        ideal_state, pulses = quantum_solvers.get_solver(solver_type="GRAPE",
                                                         algorithm_type="unitary",
                                                         penalty=False,
                                                         results_path="./",
                                                         initial_pulses=initial_pulses,
                                                         angles=angles,
                                                         axes=axes,
                                                         initial_state=initial_state)
        del quantum_solvers

        name = '/home/pzielins/Documents/repositories/IST/RandomBenchmarking/pulses_sweep/pulses_' + str(i) + '.npy'
        with open(name, 'wb') as f:
            np.save(f, pulses)

    # signal_f_init, signal_init = Filters.get_low_pass_pulses(initial_pulses, 1.64e-9, cut_off_time, granulation,
    #                                                          True)
    #
    # bloch_plotter.plot(plot_type="evolution", pulses_final=pulses, init_state=initial_state,
    #                    target_state=ideal_state,
    #                    granulation=granulation)
    # time.sleep(1)
    # bloch_plotter.plot(plot_type="evolution", pulses_final=signal_f_init, init_state=initial_state,
    #                    target_state=ideal_state,
    #                    granulation=granulation)
    #
    # bloch_plotter.plot(plot_type="pulses", pulses_final=pulses, pulses_init=signal_f_init)
    return


if __name__ == '__main__':
    # quantum_solvers = solvers_manager.SolversManager()
    process_proper_pulses(4, 0.4e-9, 0.999)
    process_proper_pulses(8, 0.4e-9, 0.999)
    process_proper_pulses(16, 0.4e-9, 0.999)