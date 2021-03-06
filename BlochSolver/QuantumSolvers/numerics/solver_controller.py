import numpy as np

from BlochSolver.QuantumSolvers.numerics import numerical_methods as nm


class SolverController:
    num_methods = nm.NumericalMethods
    error = None
    _e_min = None
    _e_max = None
    operator_error = None
    number_of_iterations = None
    time_of_termination = None
    learning_incrementation = None
    learning_decrementation = None

    @classmethod
    def load_control_settings(cls, control_settings):
        cls.error = control_settings["error"]
        cls._e_min = control_settings["e_min"]
        cls._e_max = control_settings["e_max"]
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
    def update_learning_rate(cls, fidelity: float):
        return 1. - fidelity

    @classmethod
    def check_iteration_condition(cls, iteration: int):
        if iteration >= cls.number_of_iterations:
            return True
        else:
            return False

    @classmethod
    def check_stop_condition(cls, fidelity_status: float, pulses: np.array):
        return (fidelity_status and np.logical_and(pulses > cls._e_min,
                                                   pulses < cls._e_max).all())

    @classmethod
    def check_unitary_stop_condition(cls, fidelity_status: float,
                                     propagator_fidelity_status: float, pulses: np.array):
        return (fidelity_status and propagator_fidelity_status and
                np.logical_and(pulses > cls._e_min,
                               pulses < cls._e_max).all())

    @classmethod
    def check_time_condition(cls, time: float):
        if time >= cls.time_of_termination:
            return True
        else:
            return False
