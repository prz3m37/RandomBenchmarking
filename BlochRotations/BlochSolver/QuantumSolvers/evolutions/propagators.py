import numpy as np

from BlochSolver.QuantumSolvers.numerics import numerical_methods as nm, solver_controller as sc
from BlochSolver.QuantumSolvers.rotations import rotation_handler as rh


class Propagators(rh.RotationHandler, nm.NumericalMethods):
    _n = None
    _initial_state = None
    _target_prop = None
    _granulation = None

    @classmethod
    def adopt_propagators(cls, n: int, initial_state: np.array,
                          target_propagator: np.array, granulation: int = None) -> None:
        cls._n = n
        cls._initial_state = initial_state
        cls._target_prop = target_propagator
        if granulation is not None:
            cls._granulation = granulation
        return

    @classmethod
    def evaluate_propagators(cls, inv_pulses: np.array):
        rotation_sequence = cls.get_rotation_operators(inv_pulses)
        hermit_propagators = cls.get_hermit_sequence(rotation_sequence)
        fwd_propagators = cls._evaluate_forward_propagators(rotation_sequence)
        bwd_propagators = cls._evaluate_backward_propagators(hermit_propagators)
        return fwd_propagators, bwd_propagators

    @classmethod
    def evaluate_perturbated_propagators(cls, inv_perturbated_signal: np.array):
        rotation_sequence = cls.get_perturbation_rotation_operators(inv_perturbated_signal, cls._granulation)
        hermit_propagators = cls.get_hermit_sequence(rotation_sequence)
        fwd_propagators = cls._evaluate_forward_propagators(rotation_sequence)
        bwd_propagators = cls._evaluate_backward_propagators(hermit_propagators)
        return fwd_propagators, bwd_propagators

    @classmethod
    def evaluate_propagator_fidelity(cls, inv_pulses):
        rotation_operators_sequence = cls.get_rotation_operators(inv_pulses)
        simulated_prop = cls.get_step_density_operator(pulse_operators=rotation_operators_sequence)
        prop_fidelity_status, prop_fidelity = sc.SolverController.get_operator_fidelity(cls._target_prop,
                                                                                        simulated_prop)
        return prop_fidelity_status, prop_fidelity


    @classmethod
    def evaluate_effective_propagator_fidelity(cls, inv_pulses):
        rotation_operators_sequence = cls.get_perturbation_rotation_operators(inv_pulses.reshape(cls._n,
                                                                                                 cls._granulation),
                                                                              cls._granulation)
        simulated_prop = cls.get_step_density_operator(pulse_operators=rotation_operators_sequence)
        prop_fidelity_status, prop_fidelity = sc.SolverController.get_operator_fidelity(cls._target_prop,
                                                                                        simulated_prop)
        return prop_fidelity_status, prop_fidelity

    @classmethod
    def _evaluate_backward_propagators(cls, hermit_operators: np.array):
        reverse_hermit_operators = hermit_operators[::-1]
        backward_operators = np.array([
            cls._get_step_propagator(initial_state=cls._target_prop,
                                     rotation_sequence=reverse_hermit_operators[step:])
            if step < cls._n else cls._target_prop
            for step in range(1, cls._n + 1)], dtype=object)
        return backward_operators

    @classmethod
    def _evaluate_forward_propagators(cls, rotation_sequence: np.array):
        forward_operators = np.array([
            cls._get_step_propagator(rotation_sequence=rotation_sequence[-r_length - 1:])
            for r_length in range(cls._n)])
        return forward_operators

    @classmethod
    def _get_step_propagator(cls, rotation_sequence: np.array, initial_state: np.array = None):
        rotation_evolution = cls.get_evolution(rotation_sequence)
        if initial_state is None:
            return rotation_evolution
        else:
            rotated_state = cls.get_dot_product(rotation_evolution, initial_state)
            return rotated_state

    @classmethod
    def _get_step_propagator(cls, rotation_sequence: np.array, initial_state: np.array = None):
        rotation_evolution = cls.get_evolution(rotation_sequence)
        if initial_state is None:
            return rotation_evolution
        else:
            rotated_state = cls.get_dot_product(rotation_evolution, initial_state)
            return rotated_state
