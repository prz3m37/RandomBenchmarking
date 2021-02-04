from Rotations import rotation_handler as rh
from BlochSolver.numerics import numerical_methods as nm, solver_controller as sc
from BlochSolver import settings_initializer as si
from Utils import settings
import numpy as np
import time


class QuantumGrape(rh.RotationHandler, nm.NumericalMethods):

    def __init__(self):
        self.__settings_init = si.SettingsInitializer()
        self.__sc = sc.SimulationController

        self.__settings = settings.settings
        self.__numerical_settings = settings.numerical_settings
        self.__l_rate = self.__numerical_settings["learning_rate"]
        self.load_numerical_settings(self.__numerical_settings)
        self.__sc.load_control_settings(self.__numerical_settings)

        self.__n_steps = None
        self.__inv_pulses = None
        self.__pulses_s = None
        self.__pulses_e = None
        self.__ctrl_h = np.identity(2) + np.array([[1, 0], [0, -1]])

    def __del__(self):
        del self.__settings_init

    def grape_solver(self, algorithm_type: str = None, **kwargs):
        if algorithm_type is None:
            return self.__get_grape_solver(**kwargs)
        elif algorithm_type == "time":
            return self.__get_grape_solver_time(**kwargs)
        elif algorithm_type == "rate":
            return self.__get_grape_solver_learning_rate(**kwargs)
        elif algorithm_type == "rate time":
            return self.__get_grape_solver_learning_rate_time(**kwargs)

    def __get_grape_solver(self, initial_pulses: np.array, angles: np.array, axes: np.array, initial_state: np.array):
        self.__pulses_s = initial_pulses
        target_operator = self.get_target_operator(angles, axes, initial_state)
        iteration = 0
        while True:
            self.__get_order(self.__pulses_s)
            pulse_operators_sequence = self.get_pulse_operators(self.__inv_pulses)
            step_density_operator = self.get_step_density_operator(initial_state, pulse_operators_sequence)

            if self.__sc.get_fidelity(target_operator, step_density_operator):
                break
            if self.__sc.check_iteration_condition(iteration):
                break

            hermit_pulse_operators_sequence = self.get_hermit_sequence(pulse_operators_sequence)
            backward_operators = self.__get_backward_operators(target_operator, pulse_operators_sequence,
                                                               hermit_pulse_operators_sequence)
            propagation_operators = self.__get_propagation_operators(initial_state, pulse_operators_sequence)

            self.__pulses_e = self.__pulses_s + self.__l_rate * self.get_gradient(self.__ctrl_h, backward_operators,
                                                                                  propagation_operators)
            self.__pulses_s = self.__pulses_e
            iteration += 1

        return self.__pulses_s

    def __get_grape_solver_time(self, initial_pulses: np.array, angles: np.array, axes: np.array,
                              initial_state: np.array):
        self.__pulses_s = initial_pulses
        target_operator = self.get_target_operator(angles, axes, initial_state)
        time_start = time.time()
        while True:
            self.__get_order(self.__pulses_s)
            pulse_operators_sequence = self.get_pulse_operators(self.__inv_pulses)
            step_density_operator = self.get_step_density_operator(initial_state, pulse_operators_sequence)

            time_elapsed = time_start - time.time()
            if self.__sc.get_fidelity(target_operator, step_density_operator):
                break
            if self.__sc.check_time_condition(time_elapsed):
                break

            hermit_pulse_operators_sequence = self.get_hermit_sequence(pulse_operators_sequence)
            backward_operators = self.__get_backward_operators(target_operator, pulse_operators_sequence,
                                                               hermit_pulse_operators_sequence)
            propagation_operators = self.__get_propagation_operators(initial_state, pulse_operators_sequence)

            self.__pulses_e = self.__pulses_s + self.__l_rate * self.get_gradient(self.__ctrl_h, backward_operators,
                                                                                  propagation_operators)
            self.__pulses_s = self.__pulses_e

        return self.__pulses_s

    def __get_grape_solver_learning_rate(self, initial_pulses: np.array, angles: np.array, axes: np.array,
                                       initial_state: np.array):
        self.__pulses_s = initial_pulses
        self.__pulses_e = initial_pulses
        target_operator = self.get_target_operator(angles, axes, initial_state)
        iteration = 0
        while True:
            self.__get_order(self.__pulses_s)
            pulse_operators_sequence_s = self.get_pulse_operators(self.__inv_pulses)
            step_density_operator_s = self.get_step_density_operator(initial_state, pulse_operators_sequence_s)
            self.__get_order(self.__pulses_e)
            pulse_operators_sequence_e = self.get_pulse_operators(self.__inv_pulses)
            step_density_operator_e = self.get_step_density_operator(initial_state, pulse_operators_sequence_e)

            hermit_pulse_operators_sequence = self.get_hermit_sequence(step_density_operator_s)
            backward_operators = self.__get_backward_operators(target_operator, step_density_operator_s,
                                                               hermit_pulse_operators_sequence)
            propagation_operators = self.__get_propagation_operators(initial_state, step_density_operator_s)

            self.__pulses_e = self.__pulses_s + self.__l_rate * self.get_gradient(self.__ctrl_h, backward_operators,
                                                                                  propagation_operators)
            if self.__sc.get_fidelity(target_operator, step_density_operator_e):
                break
            if self.__sc.check_iteration_condition(iteration):
                break
            self.__sc.update_learning_rate(step_density_operator_s, step_density_operator_e, self.__l_rate)

            self.__pulses_s = self.__pulses_e
            iteration += 1

        return self.__pulses_e

    def __get_grape_solver_learning_rate_time(self, initial_pulses: np.array, angles: np.array, axes: np.array,
                                            initial_state: np.array):
        self.__pulses_s = initial_pulses
        self.__pulses_e = initial_pulses
        target_operator = self.get_target_operator(angles, axes, initial_state)
        time_start = time.time()
        while True:
            self.__get_order(self.__pulses_s)
            pulse_operators_sequence_s = self.get_pulse_operators(self.__inv_pulses)
            step_density_operator_s = self.get_step_density_operator(initial_state, pulse_operators_sequence_s)
            self.__get_order(self.__pulses_e)
            pulse_operators_sequence_e = self.get_pulse_operators(self.__inv_pulses)
            step_density_operator_e = self.get_step_density_operator(initial_state, pulse_operators_sequence_e)

            hermit_pulse_operators_sequence = self.get_hermit_sequence(step_density_operator_s)
            backward_operators = self.__get_backward_operators(target_operator, step_density_operator_s,
                                                               hermit_pulse_operators_sequence)
            propagation_operators = self.__get_propagation_operators(initial_state, step_density_operator_s)

            self.__pulses_e = self.__pulses_s + self.__l_rate * self.get_gradient(self.__ctrl_h, backward_operators,
                                                                                  propagation_operators)
            time_elapsed = time_start - time.time()
            if self.__sc.get_fidelity(target_operator, step_density_operator_e):
                break
            if self.__sc.check_time_condition(time_elapsed):
                break
            self.__sc.update_learning_rate(step_density_operator_s, step_density_operator_e, self.__l_rate)

            self.__pulses_s = self.__pulses_e

        return self.__pulses_e

    def __get_backward_operators(self, target_operator: np.array,
                                 pulse_sequence: np.array, hermit_pulse_sequence: np.array):
        backward_sequences = np.fromiter(
            (np.concatenate[hermit_pulse_sequence[step:], target_operator, pulse_sequence[step:]]
             if step < self.__n_steps else target_operator for step in range(self.__n_steps)), np.complex)
        backward_operators = np.fromiter(
            (self.get_evolution(backward_sequence) for backward_sequence in backward_sequences),
            np.complex)
        return backward_operators

    def __get_step_propagation_operator(self, init_state: np.array, pulse_sequence: np.array):
        pulse_evolution = self.get_evolution(pulse_sequence)
        pulse_state = self.get_state(pulse_evolution, init_state)
        return self.get_density_operator(pulse_state)

    def __get_propagation_operators(self, init_state: np.array, pulse_sequence: np.array):
        propagation_operators = np.fromiter((
            self.__get_step_propagation_operator(init_state, pulse_sequence[0:step])
            for step in range(self.__n_steps)), np.complex)
        return propagation_operators

    def __get_order(self, pulses: np.array):
        self.__inv_pulses = pulses[::-1]
        return
