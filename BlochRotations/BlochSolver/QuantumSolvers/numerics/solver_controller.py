import numpy as np
from BlochSolver.QuantumSolvers.numerics import numerical_methods as nm
from BlochSolver.QuantumSolvers.rotations import rotation_handler as rh


class SolverController:
    num_methods = nm.NumericalMethods
    error = None
    operator_error = None
    number_of_iterations = None
    time_of_termination = None
    learning_incrementation = None
    learning_decrementation = None

    @classmethod
    def check_pulses_potential(cls, n: int, pulses: np.array, settings: dict):
        j_func = rh.RotationHandler.get_pulse_detunings(pulses)**2
        j_func += (settings["dg_factor"] * settings["bohr_magneton"] * settings["magnetic_field"]) ** 2
        total_pulse = np.sum(j_func)
        frequency = 1 / (n * settings["pulse_time"])
        if total_pulse < frequency:
            print(" ---> Total pulse:     ", total_pulse)
            print(" ---> Frequency:       ", frequency)
            print("########################################################################")
            print(" ---> [WARNING]: Set of pulses is not sufficient to get a fully rotation")
            print(" ---> [WARNING]: Simulation terminated !")
            # exit()
        else:
            print(" ---> [INFO]: Set of pulses is sufficient to get a fully rotation")
        return

    @classmethod
    def load_control_settings(cls, control_settings):
        cls.error = control_settings["error"]
        cls.operator_error = control_settings["operator_error"]
        cls.number_of_iterations = control_settings["number_of_iterations"]
        cls.learning_incrementation = control_settings["learning_incrementation"]
        cls.learning_decrementation = control_settings["learning_decrementation"]
        return

    @classmethod
    def get_fidelity(cls, target_operator: np.array, density_operator: np.array):
        fidelity = np.real(cls.num_methods.get_matrix_product(target_operator, density_operator))
        if fidelity >= cls.error:
            return True, fidelity
        else:
            return False, fidelity

    @classmethod
    def get_operator_fidelity(cls, target_propagator: np.array, simulated_propagator: np.array):
        operator_fidelity = np.real(cls.num_methods.get_matrix_product(target_propagator, simulated_propagator))
        if operator_fidelity >= cls.operator_error:
            return True, operator_fidelity
        else:
            return False, operator_fidelity

    @classmethod
    def update_learning_rate(cls, fidelity_s: float, fidelity_e: float, learning_rate: float):
        if fidelity_s < fidelity_e:
            return learning_rate * cls.learning_decrementation
        elif fidelity_s > fidelity_e:
            return learning_rate * cls.learning_incrementation
        else:
            return learning_rate

    @classmethod
    def check_gradient_condition(cls, gradient: np.array):
        if np.max(gradient) <= cls.error:
            return True
        else:
            return False

    @classmethod
    def check_iteration_condition(cls, iteration: int):
        if iteration >= cls.number_of_iterations:
            return True
        else:
            return False

    @classmethod
    def check_time_condition(cls, time: float):
        if time >= cls.time_of_termination:
            return True
        else:
            return False
