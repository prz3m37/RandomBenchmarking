from BlochSolver.QuantumSolvers.rotations import rotation_handler as rh
from BlochSolver.QuantumSolvers.numerics import numerical_methods as nm, solver_controller as sc
from BlochSolver.Utils import settings, utils
import numpy as np


class QuantumGrape(rh.RotationHandler, nm.NumericalMethods):

    def __init__(self):
        self.__sc = sc.SolverController
        self.__utils = utils.Utils
        self.__settings = settings.settings
        self.__num_sets = settings.numerical_settings
        self.__l_rate = self.__num_sets["learning_rate"]
        h_k = self.get_control_hamiltonian()
        self.load_numerical_settings(h_k, self.__settings, self.__num_sets)
        self.__sc.load_control_settings(self.__num_sets)

        self.__n_steps = None
        self.__target_operator = None
        self.__ideal_state = None
        self.__inv_pulses = None
        self.__j_func = None
        self.__pulses = None
        self.__return = None

    def grape_solver(self, algorithm_type: str = None, **kwargs):
        if algorithm_type is None or algorithm_type == "default":
            self.__utils.save_log("[INFO]: GRAPE algorithm - default, iteration termination condition")
            return self.__get_grape_solver(**kwargs)
        elif algorithm_type == "lr":
            self.__utils.save_log("[INFO]: GRAPE algorithm - learning rate, iteration termination condition")
            return self.__get_grape_solver_learning_rate(**kwargs)

    def __get_grape_solver(self, initial_pulses: np.array, angles: np.array, axes: np.array, initial_state: np.array):
        self.__return = self.__pulses = initial_pulses
        self.__n_steps = initial_pulses.shape[0]
        self.__ideal_state, self.__target_operator = self.get_target_state(angles, axes, initial_state)
        print(" ---> Target state:    ", np.around(self.__ideal_state, 3))
        iteration = 0
        while True:
            self.__j_func = self.get_pulse_detunings(self.__pulses)
            forward_operators, backward_operators = self.__get_operators(initial_state)
            self.__j_func = self.__j_func + (self.__l_rate * self.get_penalty_gradient(backward_operators,
                                                                                       forward_operators,
                                                                                       self.__j_func))
            self.__pulses = self.get_pulse_args(self.__j_func)
            self.__get_order(self.__pulses)
            status_f, fidelity = self.__evaluate_fidelity(initial_state)
            results_msg = str(iteration) + " PULSES: " + str(np.around(self.__pulses, 3)) + " FID: " + \
                          str(np.round(fidelity, 5))

            self.__utils.save_result(results_msg)
            print("pulses:", fidelity, self.__pulses)
            print("j:",self.__j_func)
            if status_f:
                self.__utils.save_log("[INFO]: Fidelity condition fulfilled")
                break
            elif self.__sc.check_iteration_condition(iteration):
                self.__utils.save_log("[INFO]: Iteration condition fulfilled")
                break
            else:
                self.__update_pulse()
                iteration += 1

        return self.__ideal_state, self.__return

    def __get_grape_solver_learning_rate(self, initial_pulses: np.array, angles: np.array, axes: np.array,
                                         initial_state: np.array):
        self.__return = self.__pulses = initial_pulses
        self.__n_steps = initial_pulses.shape[0]
        self.__ideal_state, self.__target_operator = self.get_target_state(angles, axes, initial_state)
        print(" ---> Target state:    ", np.around(self.__ideal_state, 3))
        iteration = 0
        while True:
            self.__j_func = self.get_pulse_detunings(self.__pulses)
            forward_operators, backward_operators = self.__get_operators(initial_state)
            _, fidelity_s = self.__evaluate_fidelity(initial_state)
            self.__j_func = self.__j_func + (self.__l_rate * self.get_penalty_gradient(backward_operators,
                                                                                       forward_operators,
                                                                                       self.__j_func))
            self.__pulses = self.get_pulse_args(self.__j_func)
            self.__get_order(self.__pulses)
            status_f, fidelity_e = self.__evaluate_fidelity(initial_state)
            self.__l_rate = self.__sc.update_learning_rate(fidelity_s, fidelity_e, self.__l_rate)
            results_msg = str(iteration) + " PULSES: " + str(np.around(self.__pulses, 3)) + \
                          " FID: " + str(np.round(fidelity_e, 3)) + " L_RATE: " + str(self.__l_rate)
            self.__utils.save_result(results_msg)
            print(self.__l_rate, self.__pulses)
            if status_f:
                self.__utils.save_log("[INFO]: Fidelity condition fulfilled")
                break
            elif self.__sc.check_iteration_condition(iteration):
                self.__utils.save_log("[INFO]: Iteration condition fulfilled")
                break
            else:
                self.__update_pulse()
                iteration += 1
        return self.__ideal_state, self.__return

    def __get_step_rotation_operator(self, init_state: np.array, pulse_sequence: np.array):
        rotation_evolution = self.get_evolution(pulse_sequence)
        rotated_state = self.get_state(rotation_evolution, init_state)
        return self.get_density_operator(rotated_state)

    def __get_forward_operators(self, init_state: np.array, operators_sequence: np.array):
        forward_operators = np.array([
            self.__get_step_rotation_operator(init_state, operators_sequence[-step - 1:])
            for step in range(self.__n_steps)])
        return forward_operators

    def __get_backward_operators(self, hermit_operators_sequence: np.array):
        hermit_operators_sequence = hermit_operators_sequence[::-1]
        backward_operators = np.array([
            self.__get_step_rotation_operator(self.__ideal_state, hermit_operators_sequence[step:])
            if step < self.__n_steps else self.__target_operator
            for step in range(1, self.__n_steps + 1)], dtype=object)
        return backward_operators

    def __get_operators(self, initial_state: np.array):
        self.__get_order(self.__pulses)
        rotation_operators_sequence = self.get_rotation_operators(self.__inv_pulses)
        hermit_rotation_operators_sequence = self.get_hermit_sequence(rotation_operators_sequence)
        forward_operators = self.__get_forward_operators(initial_state, rotation_operators_sequence)
        backward_operators = self.__get_backward_operators(hermit_rotation_operators_sequence)
        return forward_operators, backward_operators

    def __evaluate_fidelity(self, initial_state: np.array):
        self.__inv_pulses = np.real(self.__inv_pulses)
        rotation_operators_sequence = self.get_rotation_operators(self.__inv_pulses)
        step_density_operator = self.get_step_density_operator(initial_state, rotation_operators_sequence)
        return self.__sc.get_fidelity(self.__target_operator, step_density_operator)

    def __update_pulse(self):
        if np.logical_and(self.__pulses > self.__num_sets["e_min"], self.__pulses < self.__num_sets["e_max"]).all():
            self.__return = self.__pulses
        else:
            pass

    def __get_order(self, pulses: np.array):
        self.__inv_pulses = pulses[::-1]
        return
