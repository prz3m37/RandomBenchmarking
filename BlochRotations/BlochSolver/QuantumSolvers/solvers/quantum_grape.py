from BlochSolver.QuantumSolvers.rotations import rotation_handler as rh
from BlochSolver.QuantumSolvers.numerics import numerical_methods as nm, solver_controller as sc
from BlochSolver.Utils import settings, utils
import numpy as np


# TODO: Add PROPAGATOR evaluating methods !

class QuantumGrape(rh.RotationHandler, nm.NumericalMethods):

    def __init__(self):
        self._sc = sc.SolverController
        self._utils = utils.Utils
        self._settings = settings.settings
        self._num_sets = settings.numerical_settings
        self._l_rate = self._num_sets["learning_rate"]
        h_k = self.get_control_hamiltonian()
        self.load_numerical_settings(h_k, self._settings, self._num_sets)
        self._sc.load_control_settings(self._num_sets)

        self._n = None
        self._target_operator = None
        self._target_prop = None
        self._ideal_state = None
        self._inv_pulses = None
        self._j = None
        self._pulses = None
        self._return = None

        self._fidelity = None
        self._prop_fidelity = None
        self._fidelity_status = None
        self._prop_fidelity_status = None

    def grape_solver(self, algorithm_type: str = None, **kwargs):
        if algorithm_type is None or algorithm_type == "default":
            self._utils.save_log("[INFO]: GRAPE algorithm - default, iteration termination condition")
            return self._get_grape(**kwargs)
        elif algorithm_type == "lr":
            self._utils.save_log("[INFO]: GRAPE algorithm - learning rate, iteration termination condition")
            return  # self._get_lr_grape(**kwargs)
        elif algorithm_type == "unitary":
            self._utils.save_log("[INFO]: GRAPE algorithm - learning rate, iteration termination condition")
            return self._get_unitary_grape(**kwargs)

    def _get_grape(self, initial_pulses: np.array, angles: np.array, axes: np.array, initial_state: np.array):
        self._return = self._pulses = initial_pulses
        self._n = nm.NumericalMethods.n_shape = initial_pulses.shape[0]
        _, self._target_operator, self._ideal_state = self.get_target_state(angles, axes, initial_state)
        print(" ---> Ideal state:    ", np.around(self._ideal_state, 3))
        self._sc.check_pulses_potential(self._n, initial_pulses, self._settings)
        iteration = 0
        while True:
            print(" ---> FIDELITY", iteration, "th :", self._fidelity)
            self._get_order(self._pulses)
            f_operator, b_operator = self._evaluate_operators(initial_state)
            self._j = self.get_pulse_detunings(self._pulses)
            self._j = self._j + (self._l_rate * self.get_penalty_gradient(b_operator, f_operator, self._j))
            self._pulses = self.get_pulse_args(self._j)
            self._get_order(self._pulses)
            self._evaluate_fidelity(initial_state)

            if self._fidelity_status:
                print(" ---> FINAL FIDELITY", iteration, ":", self._fidelity)
                self._utils.save_log("[INFO]: Fidelity condition fulfilled")
                break
            elif self._sc.check_iteration_condition(iteration):
                self._utils.save_log("[INFO]: Iteration condition fulfilled")
                break
            else:
                self._update_pulse()
                iteration += 1
        return self._ideal_state, self._return

    def _get_unitary_grape(self, initial_pulses: np.array, angles: np.array, axes: np.array, initial_state: np.array):
        self._return = self._pulses = initial_pulses
        self._n = nm.NumericalMethods.n_shape = initial_pulses.shape[0]
        self._target_prop, self._target_operator, self._ideal_state = self.get_target_state(angles, axes, initial_state)
        print(" ---> Ideal state:    ", np.around(self._ideal_state, 3))
        self._sc.check_pulses_potential(self._n, initial_pulses, self._settings)
        iteration = 0
        while True:
            print(" ---> FIDELITY", iteration, "th :", self._fidelity, "OPERATOR FIDELITY: ", self._prop_fidelity)
            self._get_order(self._pulses)
            f_propagator, b_propagator = self._evaluate_propagators()
            self._j = self.get_pulse_detunings(self._pulses)
            self._j = self._j + (self._l_rate * self.get_propagator_gradient(b_propagator, f_propagator))
            self._pulses = self.get_pulse_args(self._j)
            self._get_order(self._pulses)
            self._evaluate_fidelity(initial_state)
            self._evaluate_operator_fidelity()

            if self._fidelity_status and self._prop_fidelity_status:
                print(" ---> FIDELITY", iteration, "th :", self._fidelity,
                      "FINAL OPERATOR FIDELITY: ", self._prop_fidelity)
                self._utils.save_log("[INFO]: Fidelity condition fulfilled")
                break
            elif self._sc.check_iteration_condition(iteration):
                self._utils.save_log("[INFO]: Iteration condition fulfilled")
                break
            else:
                self._update_pulse()
                iteration += 1
        return self._ideal_state, self._return

    def _evaluate_fidelity(self, initial_state: np.array):
        rotation_operators_sequence = self.get_rotation_operators(self._inv_pulses)
        density_operator = self.get_step_density_operator(rotation_operators_sequence, init_state=initial_state)
        self._fidelity_status, self._fidelity = self._sc.get_fidelity(self._target_operator, density_operator)
        return

    def _evaluate_operator_fidelity(self):
        rotation_operators_sequence = self.get_rotation_operators(self._inv_pulses)
        simulated_prop = self.get_step_density_operator(pulse_operators=rotation_operators_sequence)
        self._prop_fidelity_status, self._prop_fidelity = self._sc.get_operator_fidelity(self._target_prop,
                                                                                         simulated_prop)
        return

    def _evaluate_operators(self, initial_state: np.array):
        rotation_sequence = self.get_rotation_operators(self._inv_pulses)
        hermit_operators = self.get_hermit_sequence(rotation_sequence)
        fwd_operators = self._evaluate_forward_operators(initial_state, rotation_sequence)
        bwd_operators = self._evaluate_backward_operators(hermit_operators)
        return fwd_operators, bwd_operators

    def _evaluate_backward_operators(self, hermit_operators: np.array):
        reverse_hermit_operators = hermit_operators[::-1]
        backward_operators = np.array([
            self._get_step_rotation(initial_state=self._ideal_state, rotation_sequence=reverse_hermit_operators[step:])
            if step < self._n else self._target_operator
            for step in range(1, self._n + 1)], dtype=object)
        return backward_operators

    def _evaluate_forward_operators(self, initial_state: np.array, rotation_sequence: np.array):
        forward_operators = np.array([
            self._get_step_rotation(initial_state=initial_state, rotation_sequence=rotation_sequence[-r_length - 1:])
            for r_length in range(self._n)])
        return forward_operators

    def _evaluate_propagators(self):
        rotation_sequence = self.get_rotation_operators(self._inv_pulses)
        hermit_propagators = self.get_hermit_sequence(rotation_sequence)
        fwd_propagators = self._evaluate_forward_propagators(rotation_sequence)
        bwd_propagators = self._evaluate_backward_propagators(hermit_propagators)
        return fwd_propagators, bwd_propagators

    def _evaluate_backward_propagators(self, hermit_operators: np.array):
        reverse_hermit_operators = hermit_operators[::-1]
        backward_operators = np.array([
            self._get_step_rotation(self._ideal_state, reverse_hermit_operators[step:])
            if step < self._n else self._target_operator
            for step in range(1, self._n + 1)], dtype=object)
        return backward_operators

    def _evaluate_forward_propagators(self, rotation_sequence: np.array):
        forward_operators = np.array([
            self._get_step_rotation(rotation_sequence=rotation_sequence[-r_length - 1:])
            for r_length in range(self._n)])
        return forward_operators

    def _get_step_rotation(self, rotation_sequence: np.array, initial_state: np.array = None):
        rotation_evolution = self.get_evolution(rotation_sequence)
        if initial_state is None:
            return rotation_evolution
        else:
            rotated_state = self.get_state(rotation_evolution, initial_state)
            rotation_operator = self.get_density_operator(rotated_state)
            return rotation_operator

    def _update_pulse(self):
        if np.logical_and(self._pulses > self._num_sets["e_min"], self._pulses < self._num_sets["e_max"]).all():
            self._return = self._pulses
        else:
            pass

    def _get_order(self, pulses: np.array):
        self._inv_pulses = pulses[::-1]
        return
