import numpy as np

from BlochSolver.QuantumSolvers.numerics import numerical_methods as nm, solver_controller as sc
from BlochSolver.QuantumSolvers.rotations import rotation_handler as rh


class Operators(rh.RotationHandler, nm.NumericalMethods):
    _n = None
    _ideal_state = None
    _granulation = None
    _initial_state = None
    _target_operator = None

    @classmethod
    def adopt_operators(cls, n: int, initial_state: np.array, ideal_state: np.array,
                        target_operator: np.array, granulation: int = None) -> None:
        cls._n = n
        cls._ideal_state = ideal_state
        cls._initial_state = initial_state
        cls._target_operator = target_operator
        if granulation is not None:
            cls._granulation = granulation
        return

    @classmethod
    def evaluate_operators(cls, inv_pulses: np.array):
        rotation_sequence = cls.get_rotation_operators(inv_pulses)
        hermit_operators = cls.get_hermit_sequence(rotation_sequence)
        fwd_operators = cls._evaluate_forward_operators(rotation_sequence)
        bwd_operators = cls._evaluate_backward_operators(hermit_operators)
        return fwd_operators, bwd_operators

    @classmethod
    def evaluate_perturbated_operators(cls, inv_perturbated_signal: np.array):
        rotation_sequence = cls.get_perturbation_rotation_operators(inv_perturbated_signal, cls._granulation)
        hermit_operators = cls.get_hermit_sequence(rotation_sequence)
        fwd_operators = cls._evaluate_forward_operators(rotation_sequence)
        bwd_operators = cls._evaluate_backward_operators(hermit_operators)
        return fwd_operators, bwd_operators

    @classmethod
    def evaluate_fidelity(cls, inv_pulses: np.array):
        rotation_operators_sequence = cls.get_rotation_operators(inv_pulses)
        density_operator = cls.get_step_density_operator(pulse_operators=rotation_operators_sequence,
                                                         init_state=cls._initial_state)
        fidelity_status, fidelity = sc.SolverController.get_fidelity(cls._target_operator, density_operator)
        return fidelity_status, fidelity

    @classmethod
    def evaluate_effective_fidelity(cls, inv_pulses: np.array):
        rotation_operators_sequence = cls.get_perturbation_rotation_operators(inv_pulses.reshape(cls._n,
                                                                                                 cls._granulation),
                                                                              cls._granulation)
        density_operator = cls.get_step_density_operator(pulse_operators=rotation_operators_sequence,
                                                         init_state=cls._initial_state)
        fidelity_status, fidelity = sc.SolverController.get_fidelity(cls._target_operator, density_operator)
        return fidelity_status, fidelity

    @classmethod
    def _evaluate_backward_operators(cls, hermit_operators: np.array):
        reverse_hermit_operators = hermit_operators[::-1]
        backward_operators = np.array([
            cls._get_step_rotation(cls._ideal_state, reverse_hermit_operators[step:])
            if step < cls._n else cls._target_operator
            for step in range(1, cls._n + 1)], dtype=object)
        return backward_operators

    @classmethod
    def _evaluate_forward_operators(cls, rotation_sequence: np.array):
        forward_operators = np.array([
            cls._get_step_rotation(cls._initial_state, rotation_sequence[-r_length - 1:])
            for r_length in range(cls._n)])
        return forward_operators

    @classmethod
    def _get_step_rotation(cls, state: np.array, rotation_sequence: np.array):
        rotation_evolution = cls.get_evolution(rotation_sequence)
        rotated_state = cls.get_dot_product(rotation_evolution, state)
        rotation_operator = cls.get_density_operator(rotated_state)
        return rotation_operator
