from BlochSolver.QuantumSolvers.rotations import rotation_handler as rh
from BlochSolver.QuantumSolvers.numerics import numerical_methods as nm, solver_controller as sc
from BlochSolver.Utils import settings, utils
import numpy as np
import time


class QuantumGrape(rh.RotationHandler, nm.NumericalMethods):

    def __init__(self):
        self.__sc = sc.SimulationController
        self.__utils = utils.Utils
        self.__settings = settings.settings
        self.__numerical_settings = settings.numerical_settings
        self.__l_rate = self.__numerical_settings["learning_rate"]
        self.__get_control_hamiltonian()
        self.load_numerical_settings(self.__settings, self.__ctrl_h)
        self.__sc.load_control_settings(self.__numerical_settings)

        self.__ctrl_h = None
        self.__n_steps = None
        self.__target_operator = None
        self.__ideal_state = None
        self.__inv_pulses = None
        self.__e_s = None
        self.__e_e = None

    def grape_solver(self, algorithm_type: str = None, **kwargs):
        if algorithm_type is None or algorithm_type == "default":
            self.__utils.save_log("[INFO]: GRAPE algorithm - default, iteration termination condition")
            return self.__get_grape_solver(**kwargs)
        elif algorithm_type == "time":
            self.__utils.save_log("[INFO]: GRAPE algorithm - time termination condition")
            return self.__get_grape_solver_time(**kwargs)
        elif algorithm_type == "rate":
            self.__utils.save_log("[INFO]: GRAPE algorithm - learning rate, iteration termination condition")
            return self.__get_grape_solver_learning_rate(**kwargs)
        elif algorithm_type == "rate time":
            self.__utils.save_log("[INFO]: GRAPE algorithm - learning rate, time termination condition")
            return self.__get_grape_solver_learning_rate_time(**kwargs)

    def __get_grape_solver(self, initial_pulses: np.array, angles: np.array, axes: np.array, initial_state: np.array):
        self.__e_s = initial_pulses
        self.__n_steps = initial_pulses.shape[0]
        self.__ideal_state, self.__target_operator = self.get_target_state(angles, axes, initial_state)
        print(" ---> Target state:    ", np.around(self.__ideal_state, 3))
        iteration = 0
        while True:
            propagation_operators, backward_operators = self.__get_operators(initial_state)
            self.__e_e = self.__e_s + self.__l_rate * self.get_gradient(backward_operators, propagation_operators)
            # self.__e_e = np.clip(self.__e_e, a_min=0.0015, a_max=0.004)

            self.__get_order(self.__e_e)
            status_f, fidelity = self.__evaluate_fidelity(initial_state)
            results_msg = str(iteration) + " PULSES: " + str(self.__e_e) + " FID: " + str(np.round(fidelity, 5))
            self.__utils.save_result(results_msg)

            if status_f:
                self.__utils.save_log("[INFO]: Fidelity condition fulfilled")
                break
            elif self.__sc.check_iteration_condition(iteration):
                self.__utils.save_log("[INFO]: Iteration condition fulfilled")
                break
            else:
                self.__e_s = self.__e_e
                iteration += 1

        return self.__ideal_state, self.__e_e

    def __get_grape_solver_learning_rate(self, initial_pulses: np.array, angles: np.array, axes: np.array,
                                         initial_state: np.array):
        self.__e_s = initial_pulses
        self.__n_steps = initial_pulses.shape[0]
        self.__ideal_state, self.__target_operator = self.get_target_state(angles, axes, initial_state)
        print(" ---> Target state:    ", np.around(self.__ideal_state, 3))
        iteration = 0
        while True:
            propagation_operators, backward_operators = self.__get_operators(initial_state)
            _, fidelity_s = self.__evaluate_fidelity(initial_state)
            self.__e_e = self.__e_s + self.__l_rate * self.get_gradient(backward_operators, propagation_operators)

            status_f, fidelity_e = self.__evaluate_fidelity(initial_state)
            self.__l_rate = self.__sc.update_learning_rate(fidelity_s, fidelity_e, self.__l_rate)

            results_msg = str(iteration) + " PULSES: " + str(self.__e_s) + " STATE: " + str(self.__e_e) + \
                          " FID: " + str(np.round(fidelity_e, 3)) + " L_RATE: " + str(self.__l_rate)
            self.__utils.save_result(results_msg)

            if status_f:
                self.__utils.save_log("[INFO]: Fidelity condition fulfilled")
                break
            elif self.__sc.check_iteration_condition(iteration):
                self.__utils.save_log("[INFO]: Iteration condition fulfilled")
                break
            else:
                self.__e_s = self.__e_e
                iteration += 1

        return self.__ideal_state, self.__e_e

    def __get_grape_solver_time(self, initial_pulses: np.array, angles: np.array, axes: np.array,
                                initial_state: np.array):
        self.__e_s = initial_pulses
        self.__n_steps = initial_pulses.shape[0]
        self.__ideal_state, self.__target_operator = self.get_target_state(angles, axes, initial_state)
        print(" ---> Target state:    ", np.around(self.__ideal_state, 2))
        time_start = time.time()
        iteration = 0
        while True:
            propagation_operators, backward_operators = self.__get_operators(initial_state)
            self.__e_e = self.__e_s + self.__l_rate * self.get_gradient(backward_operators, propagation_operators)
            status_f, fidelity = self.__evaluate_fidelity(initial_state)
            results_msg = str(iteration) + " PULSES: " + str(self.__e_e) + " FID: " + str(np.round(fidelity, 3))
            self.__utils.save_result(results_msg)

            time_elapsed = time_start - time.time()
            if status_f:
                self.__utils.save_log("[INFO]: Fidelity condition fulfilled")
                break
            elif self.__sc.check_time_condition(time_elapsed):
                self.__utils.save_log("[INFO]: Time condition fulfilled")
                break
            else:
                self.__e_s = self.__e_e
                iteration += 1

        return self.__ideal_state, self.__e_e

    def __get_grape_solver_learning_rate_time(self, initial_pulses: np.array, angles: np.array, axes: np.array,
                                              initial_state: np.array):
        self.__e_s = initial_pulses
        self.__n_steps = initial_pulses.shape[0]
        self.__ideal_state, self.__target_operator = self.get_target_state(angles, axes, initial_state)
        print(" ---> Target state:    ", np.around(self.__ideal_state, 2))
        time_start = time.time()
        iteration = 0
        while True:
            propagation_operators, backward_operators = self.__get_operators(initial_state)
            _, fidelity_s = self.__evaluate_fidelity(initial_state)
            self.__e_e = self.__e_s + self.__l_rate * self.get_gradient(backward_operators, propagation_operators)
            status_f, fidelity_e = self.__evaluate_fidelity(initial_state)
            self.__l_rate = self.__sc.update_learning_rate(fidelity_s, fidelity_e, self.__l_rate)

            results_msg = str(iteration) + " PULSES: " + str(self.__e_s) + " STATE: " + str(self.__e_e) + \
                          " FID: " + str(np.round(fidelity_e, 3)) + " L_RATE: " + str(self.__l_rate)
            self.__utils.save_result(results_msg)

            time_elapsed = time_start - time.time()
            if status_f:
                self.__utils.save_log("[INFO]: Fidelity condition fulfilled")
                break
            elif self.__sc.check_time_condition(time_elapsed):
                self.__utils.save_log("[INFO]: Time condition fulfilled")
                break
            else:
                self.__e_s = self.__e_e
                iteration += 1

        return self.__ideal_state, self.__e_e

    def __get_step_evolution_operator(self, init_state: np.array, pulse_sequence: np.array):
        pulse_evolution = self.get_evolution(pulse_sequence)
        pulse_state = self.get_state(pulse_evolution, init_state)
        return self.get_density_operator(pulse_state)

    def __get_propagation_operators(self, init_state: np.array, pulse_sequence: np.array):
        propagation_operators = np.array([
            self.__get_step_evolution_operator(init_state, pulse_sequence[-step - 1:])
            for step in range(self.__n_steps)])
        return propagation_operators

    def __get_backward_operators(self, hermit_pulse_sequence: np.array):
        hermit_pulse_sequence = hermit_pulse_sequence[::-1]
        backward_operators = np.array([
            self.__get_step_evolution_operator(self.__ideal_state, hermit_pulse_sequence[step:])
            if step < self.__n_steps else self.__target_operator
            for step in range(1, self.__n_steps + 1)], dtype=object)
        return backward_operators

    def __get_operators(self, initial_state: np.array):
        self.__get_order(self.__e_s)
        pulse_operators_sequence = self.get_pulse_operators(self.__inv_pulses)
        hermit_pulse_operators_sequence = self.get_hermit_sequence(pulse_operators_sequence)
        propagation_operators = self.__get_propagation_operators(initial_state, pulse_operators_sequence)
        backward_operators = self.__get_backward_operators(hermit_pulse_operators_sequence)
        return propagation_operators, backward_operators

    def __evaluate_fidelity(self, initial_state: np.array):
        self.__inv_pulses = np.real(self.__inv_pulses)
        pulse_operators_sequence = self.get_pulse_operators(self.__inv_pulses)
        step_density_operator = self.get_step_density_operator(initial_state, pulse_operators_sequence)
        return self.__sc.get_fidelity(self.__target_operator, step_density_operator)

    def __get_order(self, pulses: np.array):
        self.__inv_pulses = pulses[::-1]
        return

    def __get_control_hamiltonian(self):
        h = self.idn + np.array([[1, 0], [0, -1]])
        # self.__ctrl_h = np.kron(h, idn)
        self.__ctrl_h = h
        return
