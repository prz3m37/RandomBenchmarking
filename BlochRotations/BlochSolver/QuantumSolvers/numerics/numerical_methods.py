from BlochSolver.QuantumSolvers.rotations import rotation_handler
import numpy as np


class NumericalMethods:
    dt = None
    h_bar = None
    h_k = None
    gamma = None
    j_max = None
    j_min = None

    @classmethod
    def load_numerical_settings(cls, control_hamiltonian: np.array, settings: dict, num_settings: dict) -> None:
        cls.dt = settings["pulse_time"]
        cls.h_bar = settings["h_bar"]
        cls.gamma = num_settings["gamma"]
        cls.h_k = control_hamiltonian
        cls.j_min = rotation_handler.RotationHandler.get_pulse_detuning(num_settings["e_min"])
        cls.j_max = rotation_handler.RotationHandler.get_pulse_detuning(num_settings["e_max"])
        print(cls.j_max, cls.j_min)
        return

    @classmethod
    def get_inverse_matrix(cls, matrix: np.array):
        return np.linalg.inv(matrix)

    @classmethod
    def get_commutator(cls, operator_a: np.array, operator_b: np.array):
        return np.dot(operator_a, operator_b) - np.dot(operator_b, operator_a)

    @classmethod
    def get_matrix_product(cls, operator_a: np.array, operator_b: np.array):
        return np.trace(np.dot(np.conj(operator_a.T), operator_b))

    @classmethod
    def get_gradient(cls, back_operators: np.array, prop_operators: np.array):
        grad = -1 * np.array(
            [cls.get_matrix_product(back_op, 1j * (cls.dt / cls.h_bar) * cls.get_commutator(cls.h_k, prop_op))
             for back_op, prop_op in zip(back_operators, prop_operators)])
        return grad

    @classmethod
    def __get_penalty(cls, j: float):
        if j > cls.j_max:
            return np.log(j-cls.j_max)
        elif j < cls.j_min:
            return np.log(j - cls.j_min)
        else:
            return 0

    @classmethod
    def get_penalty_gradient(cls, back_operators: np.array, prop_operators: np.array, detunings: np.array):
        penalty_gradient = -1 * np.array(
            [cls.get_matrix_product(back_op, 1j * (cls.dt / cls.h_bar) * cls.get_commutator(cls.h_k, prop_op)) +
             cls.__get_penalty(detunning)
             for back_op, prop_op, detunning in zip(back_operators, prop_operators, detunings)])

        return penalty_gradient

    @classmethod
    def get_density_operator(cls, vector_a: np.array):
        return np.outer(vector_a, np.conj(vector_a))

    @classmethod
    def get_hermit_sequence(cls, operator_sequence: np.array):
        return np.array([np.conj(operator.T) for operator in operator_sequence])
