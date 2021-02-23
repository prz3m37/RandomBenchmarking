from BlochSolver.QuantumSolvers.rotations import rotation_handler as rh
import numpy as np


class NumericalMethods:
    n_shape = None
    dt = None
    h_bar = None
    h_k = None
    idn_num = None
    j_max = None
    j_min = None

    @classmethod
    def load_numerical_settings(cls, control_hamiltonian: np.array, settings: dict, num_settings: dict) -> None:
        cls.dt = settings["pulse_time"]
        cls.h_bar = settings["h_bar"]
        cls.idn_num = num_settings["identities"]
        cls.h_k = control_hamiltonian
        cls.j_min = rh.RotationHandler.get_pulse_detuning(num_settings["e_min"])
        cls.j_max = rh.RotationHandler.get_pulse_detuning(num_settings["e_max"])
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
    def get_gradient(cls, back_operators: np.array, forward_operators: np.array):
        grad = -1 * np.array(
            [cls.get_matrix_product(back_op, 1j * cls.dt * cls.get_commutator(cls.h_k, fwd_op))
             for back_op, fwd_op in zip(back_operators, forward_operators)])
        return grad

    @classmethod
    def get_penalty_gradient(cls, backward_operators: np.array, forward_operators: np.array, detunings: np.array):
        penalty_gradient = np.array(
            [-1 * cls.get_matrix_product(back_op, 1j * cls.dt * cls.get_commutator(cls.h_k, fwd_op)) -
             cls.__get_penalty(detunning) if k < cls.n_shape - cls.idn_num
             else
             -1 * cls.get_matrix_product(back_op,
                                         1j * cls.dt * cls.get_commutator(rh.RotationHandler.idn,
                                                                          fwd_op)) -
             cls.__get_penalty(detunning)
             for k, (back_op, fwd_op, detunning) in enumerate(zip(backward_operators, forward_operators, detunings))])
        return np.real(penalty_gradient)

    @classmethod
    def get_propagator_gradient(cls, backward_propagator: np.array, forward_propagator: np.array):
        propagator_gradient = np.array(
            [-1 * cls.get_matrix_product(back_prop, 1j * cls.dt * rh.RotationHandler.get_dot_product(cls.h_k, fwd_prop))
             if k < cls.n_shape - cls.idn_num else
             -1 * cls.get_matrix_product(back_prop, 1j * cls.dt *
                                         rh.RotationHandler.get_dot_product(rh.RotationHandler.idn, fwd_prop))
             for k, (back_prop, fwd_prop) in enumerate(zip(backward_propagator, forward_propagator))])
        return np.real(propagator_gradient)

    @classmethod
    def __get_penalty(cls, j: float):
        if j > cls.j_max:
            return (j - cls.j_max) ** 2  # np.log(np.abs(j-(1+cls.j_max)))
        elif j < cls.j_min:
            return -(cls.j_min - j) ** 2  # np.log(-(j - (1+cls.j_min)))
        else:
            return 0

    @classmethod
    def get_density_operator(cls, vector_a: np.array):
        return np.outer(vector_a, np.conj(vector_a))

    @classmethod
    def get_hermit_sequence(cls, operator_sequence: np.array):
        return np.array([np.conj(operator.T) for operator in operator_sequence])
