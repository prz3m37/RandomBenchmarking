import numpy as np

from BlochSolver.QuantumSolvers.evolutions import operators, propagators
from BlochSolver.QuantumSolvers.numerics import numerical_methods as nm, solver_controller as sc
from BlochSolver.QuantumSolvers.rotations import rotation_handler as rh
from BlochSolver.Utils import settings, utils


# TODO: PHD thesis filter functions


class QuantumGrape(rh.RotationHandler, nm.NumericalMethods):

    def __init__(self):
        self._sc = sc.SolverController
        self._prop = propagators.Propagators
        self._operators = operators.Operators
        self._utils = utils.Utils
        self._settings = settings.settings
        self._num_sets = settings.numerical_settings
        self._l_rate = self._num_sets["learning_rate"]
        h_k = self.get_control_hamiltonian()
        self.load_numerical_settings(h_k, self._settings, self._num_sets)
        self._sc.load_control_settings(self._num_sets)

        self._penalty = False
        self._inv_pulses = None
        self._j = None
        self._pulses = None
        self._return = None

    def grape_solver(self, algorithm_type: str = None, penalty: bool = False, **kwargs):
        self._penalty = penalty
        if algorithm_type is None or algorithm_type == "default":
            return self._get_grape(**kwargs)
        elif algorithm_type == "unitary":
            return self._get_unitary_grape(**kwargs)
        elif algorithm_type == "lr unitary":
            return self._get_lr_unitary_grape(**kwargs)
        elif algorithm_type == "lr":
            return self._get_lr_grape(**kwargs)

    def _get_grape(self, initial_pulses: np.array, angles: np.array, axes: np.array, initial_state: np.array):
        self._return = self._pulses = initial_pulses
        n = nm.NumericalMethods.n_shape = initial_pulses.shape[0]
        _, target_operator, ideal_state = self.get_target_state(angles, axes, initial_state)
        print(" ---> Ideal state:     ", np.around(ideal_state, 3), "\n")
        self._operators.adopt_operators(n, initial_state, ideal_state, target_operator)
        iteration = 0
        while True:
            self._get_order(self._pulses)
            f_operator, b_operator = self._operators.evaluate_operators(self._inv_pulses)
            self._j = self.get_pulse_detunings(self._pulses)
            if self._penalty:
                self._j = self._j + (self._l_rate * self.get_penalty_gradient(b_operator, f_operator, self._j))
            else:
                self._j = self._j + (self._l_rate * self.get_gradient(b_operator, f_operator))
            self._pulses = self.get_pulse_args(self._j)
            self._get_order(self._pulses)
            fidelity_status, fidelity = self._operators.evaluate_fidelity(self._inv_pulses)

            if self._check_stop_condition(fidelity_status):
                print(" **** FINAL FIDELITY", iteration, ":", fidelity)
                break
            elif self._sc.check_iteration_condition(iteration):
                print(" **** STOP CONDITION REACHED")
                break
            else:
                print(" **** FIDELITY", iteration, "th :", fidelity)
                self._update_pulse()
                iteration += 1
        return ideal_state, self._return

    def _get_unitary_grape(self, initial_pulses: np.array, angles: np.array, axes: np.array, initial_state: np.array):
        self._return = self._pulses = initial_pulses
        n = nm.NumericalMethods.n_shape = initial_pulses.shape[0]
        target_prop, target_operator, ideal_state = self.get_target_state(angles, axes, initial_state)
        print(" ---> Ideal state:     ", np.around(ideal_state, 3), "\n")
        self._prop.adopt_propagators(n, initial_state, target_prop)
        self._operators.adopt_operators(n, initial_state, ideal_state, target_operator)
        iteration = 0
        while True:
            self._get_order(self._pulses)
            fwd_propagators, bwd_propagators = self._prop.evaluate_propagators(self._inv_pulses)
            self._j = self.get_pulse_detunings(self._pulses)
            if self._penalty:
                self._j = self._j + (self._l_rate *
                                     self.get_penalty_propagator_gradient(bwd_propagators, fwd_propagators, self._j))
            else:
                self._j = self._j + (self._l_rate * self.get_propagator_gradient(bwd_propagators, fwd_propagators))
            self._pulses = self.get_pulse_args(self._j)
            self._get_order(self._pulses)
            fidelity_status, fidelity = self._operators.evaluate_fidelity(self._inv_pulses)
            prop_fidelity_status, prop_fidelity = self._prop.evaluate_propagator_fidelity(self._inv_pulses)

            if self._check_unitary_stop_condition(fidelity_status, prop_fidelity_status):
                print("  **** FINAL FIDELITY", iteration, "th :", fidelity,
                      "FINAL OPERATOR FIDELITY: ", prop_fidelity)
                break
            elif self._sc.check_iteration_condition(iteration):
                print(" **** STOP CONDITION REACHED")
                break
            else:
                print(" **** FIDELITY", iteration, "th :", fidelity, "OPERATOR FIDELITY: ", prop_fidelity)
                self._update_pulse()
                iteration += 1
        return ideal_state, self._return

    def _get_lr_grape(self, initial_pulses: np.array, angles: np.array, axes: np.array, initial_state: np.array):
        self._return = self._pulses = initial_pulses
        n = nm.NumericalMethods.n_shape = initial_pulses.shape[0]
        _, target_operator, ideal_state = self.get_target_state(angles, axes, initial_state)
        print(" ---> Ideal state:     ", np.around(ideal_state, 3), "\n")
        self._operators.adopt_operators(n, initial_state, ideal_state, target_operator)
        iteration = 0
        while True:
            self._get_order(self._pulses)
            f_operator, b_operator = self._operators.evaluate_operators(self._inv_pulses)
            self._j = self.get_pulse_detunings(self._pulses)
            if self._penalty:
                self._j = self._j + (self._l_rate * self.get_penalty_gradient(b_operator, f_operator, self._j))
            else:
                self._j = self._j + (self._l_rate * self.get_gradient(b_operator, f_operator))
            self._pulses = self.get_pulse_args(self._j)
            self._get_order(self._pulses)
            fidelity_status, fidelity = self._operators.evaluate_fidelity(self._inv_pulses)
            self._l_rate = self._sc.update_learning_rate(fidelity)

            if self._check_stop_condition(fidelity_status):
                print(" **** FINAL FIDELITY", iteration, ":", fidelity)
                break
            elif self._sc.check_iteration_condition(iteration):
                print(" **** STOP CONDITION REACHED")
                break
            else:
                print(" ---> FIDELITY", iteration, "th :", fidelity)
                self._update_pulse()
                iteration += 1
        return ideal_state, self._return

    def _get_lr_unitary_grape(self, initial_pulses: np.array, angles: np.array, axes: np.array,
                              initial_state: np.array):
        self._return = self._pulses = initial_pulses
        n = nm.NumericalMethods.n_shape = initial_pulses.shape[0]
        target_prop, target_operator, ideal_state = self.get_target_state(angles, axes, initial_state)
        print(" ---> Ideal state:     ", np.around(ideal_state, 3), "\n")
        self._prop.adopt_propagators(n, initial_state, target_prop)
        self._operators.adopt_operators(n, initial_state, ideal_state, target_operator)
        iteration = 0
        while True:
            self._get_order(self._pulses)
            fwd_propagators, bwd_propagators = self._prop.evaluate_propagators(self._inv_pulses)
            self._j = self.get_pulse_detunings(self._pulses)
            if self._penalty:
                self._j = self._j + (self._l_rate *
                                     self.get_penalty_propagator_gradient(bwd_propagators, fwd_propagators, self._j))
            else:
                self._j = self._j + (self._l_rate * self.get_propagator_gradient(bwd_propagators, fwd_propagators))
            self._pulses = self.get_pulse_args(self._j)
            self._get_order(self._pulses)
            fidelity_status, fidelity = self._operators.evaluate_fidelity(self._inv_pulses)
            prop_fidelity_status, prop_fidelity = self._prop.evaluate_propagator_fidelity(self._inv_pulses)
            self._l_rate = self._sc.update_learning_rate(fidelity)

            if self._check_unitary_stop_condition(fidelity_status, prop_fidelity_status):
                print(" **** FINAL FIDELITY", iteration, "th :", fidelity, "FINAL OPERATOR FIDELITY: ", prop_fidelity)
                break
            elif self._sc.check_iteration_condition(iteration):
                print(" **** STOP CONDITION REACHED")
                break
            else:
                print(" **** FIDELITY", iteration, "th :", fidelity, "OPERATOR FIDELITY: ", prop_fidelity)
                self._update_pulse()
                iteration += 1
        return ideal_state, self._return

    def _update_pulse(self):
        if np.logical_and(self._pulses > self._num_sets["e_min"], self._pulses < self._num_sets["e_max"]).all():
            self._return = self._pulses
        else:
            pass

    def _check_stop_condition(self, fidelity_status: float):
        return (fidelity_status and np.logical_and(self._pulses > self._num_sets["e_min"],
                                                   self._pulses < self._num_sets["e_max"]).all())

    def _check_unitary_stop_condition(self, fidelity_status: float, propagator_fidelity_status: float):
        return (fidelity_status and propagator_fidelity_status and
                np.logical_and(self._pulses > self._num_sets["e_min"], self._pulses < self._num_sets["e_max"]).all())

    def _get_order(self, pulses: np.array):
        self._inv_pulses = pulses[::-1]
        return
