from Rotations import rotation_handler as rh
from BlochSolver import numerical_methods as nm
from BlochSolver import settings_initializer as si
from BlochSolver import solver_controller as sc
from Utils import settings
import numpy as np


# TODO: Check order of pulses !!!
class QuantumGrape(rh.RotationHandler, nm.NumericalMethods):

    def __init__(self):
        self.__settings_init = si.SettingsInitializer()
        self.__sc = sc.SimulationController

        self.__settings = settings.settings
        self.__numerical_settings = settings.numerical_settings
        self.__learning_rate = self.__numerical_settings["learning_rate"]
        self.load_numerical_settings(self.__numerical_settings)
        self.__sc.load_control_settings(self.__numerical_settings)

        self.__inv_pulses = None
        self.__n_steps = None
        self.__pulse_sequence_e = None
        self.__ctrl_h = np.identity(2) + np.array([[1, 0], [0, -1]])

    def __del__(self):
        del self.__settings_init

    def grape_solver(self, initial_pulses: np.array, angles: np.array, axes: np.array, initial_state: np.array):
        self.__get_order(initial_pulses)
        self.__pulse_sequence_e = self.get_pulse_operators(self.__inv_pulses)
        target_operator = self.get_target_operator(angles, axes, initial_state)

        while True:
            pulse_sequence = self.get_pulse_operators(self.__pulse_sequence_e)
            hermit_pulse_sequence = self.get_hermit_sequence(pulse_sequence)

            backward_operators = self.__get_backward_operators(target_operator, pulse_sequence, hermit_pulse_sequence)
            propagation_operators = self.__get_propagation_operators(initial_state, pulse_sequence)

            self.__pulse_sequence_e = \
                pulse_sequence + self.__learning_rate * self.get_quantum_gradient(self.__ctrl_h, backward_operators,
                                                                                  propagation_operators)

            break

        return

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

    def __get_order(self, initial_pulses: np.array):
        self.__inv_pulses = initial_pulses[::-1]
        return
