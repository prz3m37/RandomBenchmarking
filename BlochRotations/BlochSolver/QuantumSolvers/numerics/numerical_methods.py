import numpy as np


class NumericalMethods:
    dt = None
    ctr_h = None
    h_bar = None

    @classmethod
    def load_numerical_settings(cls, settings: dict, control_hamiltonian:np.array):
        cls.dt = 0.41 * 10**(-9)
        cls.h_bar = settings["h_bar"]
        cls.ctr_h = control_hamiltonian
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
        grad = -1 * np.array([cls.get_matrix_product(back_op, 1j * (cls.dt / cls.h_bar) * cls.get_commutator(cls.ctr_h,
                                                                                                             prop_op))
                              for back_op, prop_op in zip(back_operators, prop_operators)])
        return grad

    @classmethod
    def get_density_operator(cls, vector_a: np.array):
        return np.outer(vector_a, np.conj(vector_a))

    @classmethod
    def get_hermit_sequence(cls, operator_sequence: np.array):
        return np.array([np.conj(operator.T) for operator in operator_sequence])
