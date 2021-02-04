import numpy as np
from BlochSolver.numerics import numerical_methods as nm


class SimulationController:
    num_methods = nm.NumericalMethods
    error = None
    number_of_iterations = None
    time_of_termination = None
    learning_incrementation = None
    learning_decrementation = None

    @classmethod
    def load_control_settings(cls, control_settings):
        cls.error = control_settings.numerical_settings["error"]
        cls.number_of_iterations = control_settings.numerical_settings["number_of_iterations"]
        cls.time_of_termination = control_settings.numerical_settings["time_of_termination"]
        cls.learning_incrementation = control_settings.numerical_settings["learning_incrementation"]
        cls.learning_decrementation = control_settings.numerical_settings["learning_decrementation"]
        return

    @classmethod
    def get_fidelity(cls, target_operator: np.array, density_operator: np.array):
        fidelity = cls.num_methods.get_matrix_product(target_operator, density_operator)
        if fidelity <= cls.error:
            return True
        else:
            return False

    @classmethod
    def update_learning_rate(cls, fidelity_a: float, fidelity_b: float, learning_rate: float):
        if fidelity_a < fidelity_b:
            learning_rate /= cls.learning_decrementation
            return
        elif fidelity_b > fidelity_a:
            learning_rate *= cls.learning_decrementation
            return
        else:
            return

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
