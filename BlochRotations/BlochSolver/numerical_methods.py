import numpy as np


class NumericalMethods:
    dt = None

    @classmethod
    def load_numerical_settings(cls, settings):
        cls.dt = settings["time_tc"]
        return

    @classmethod
    def get_inverse_matrix(cls, matrix: np.array):
        return np.linalg.inv(matrix)

    @classmethod
    def get_commutator(cls, operator_a: np.array, operator_b: np.array):
        return np.dot(operator_a, operator_b) - np.dot(operator_b, operator_a)

    @classmethod
    def get_matrix_product(cls, operator_a: np.array, operator_b: np.array):
        return np.trace(np.dot(np.conj(operator_a).T, operator_b))

    @classmethod
    def get_quantum_gradient(cls, ctr_field_h: np.array, backward_operators: np.array,
                             propagation_operators: np.array):
        gradient = np.fromiter((-1 * cls.get_matrix_product(backward_operator,
                                                            1j * cls.dt * cls.get_commutator(ctr_field_h,
                                                                                             propagation_operator))
                                for backward_operator, propagation_operator
                                in zip(backward_operators, propagation_operators)), np.complex)
        return gradient

    @classmethod
    def get_density_operator(cls, vector_a: np.array):
        return np.outer(vector_a, np.conj(vector_a))

    @classmethod
    def get_hermit_sequence(cls, operator_sequence: np.array):
        return np.fromiter((np.conj(operator).T for operator in operator_sequence), np.complex)
