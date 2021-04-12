cimport numpy as np
import numpy as np
from BlochSolver.Utils import settings as s

cdef class RotationHandler:

    def __cinit__(self):
        self.idn = np.identity(2)
        self.pauli_x = np.array([[0, 1], [1, 0]])
        self.pauli_y = np.array([[0, -1j], [1j, 0]])
        self.pauli_z = np.array([[1, 0], [0, -1]])

    def __dealloc__(self):
        del self.idn
        del self.pauli_x
        del self.pauli_y
        del self.pauli_z

    cdef public np.ndarray get_pulse_detuning(self, float pulse):
        return np.sqrt((0.25 * pulse ** 2) + (2 * s.settings["time_tc"] ** 2)) - 0.5 * pulse

    cdef public np.ndarray get_pulse_detunings(self, np.ndarray pulses):
        return np.array([self.get_pulse_detuning(pulse) for pulse in pulses])

    cdef np.ndarray _get_pulse_arg(self, float pulse_detuning):
        return (2 * s.settings["time_tc"] ** 2 - pulse_detuning ** 2) / pulse_detuning

    cdef public np.ndarray get_pulse_args(self, np.ndarray pulse_detunings):
        return np.array([self._get_pulse_arg(pulse_detuning) for pulse_detuning in pulse_detunings])

    cdef np.ndarray _get_rotation_operator(self, float pulse):
        cdef np.ndarray j_f = self.get_pulse_detuning(pulse)
        cdef float f_term = s.settings["dg_factor"] * s.settings["bohr_magneton"] * s.settings["magnetic_field"]
        cdef float omega = np.sqrt(j_f ** 2 + f_term ** 2)
        cdef float phi = (omega * s.settings["pulse_time"]) / (2 * s.settings["h_bar"])
        cdef float alpha = -j_f / omega
        cdef float beta = f_term / omega
        return np.array([[np.cos(phi) + 1j * alpha * np.sin(phi), -1j * beta * np.sin(phi)],
                         [-1j * beta * np.sin(phi), np.cos(phi) - 1j * alpha * np.sin(phi)]])

    cdef public np.ndarray get_control_hamiltonian(self):
        return self.pauli_z 

    cdef public np.ndarray get_rotation_operators(self, np.ndarray pulses):
        return np.array([self._get_rotation_operator(pulse) for pulse in pulses])

    cdef public np.ndarray get_perturbation_rotation_operators(self, np.ndarray perturbated_signal, int granulation):
        cdef np.ndarray perturbated_pulses = np.array([self.get_evolution(
            np.array([self._get_perturbated_rotation_operator(pulse, granulation) for pulse in chunk_pulses]))
                                       for chunk_pulses in perturbated_signal])
        return perturbated_pulses
    
    cdef public np.ndarray get_evolution(self, np.ndarray pulse_sequence):
        if pulse_sequence.shape[0] == 1:
            return pulse_sequence[0]
        else:
            return np.linalg.multi_dot(pulse_sequence)

    cdef public np.ndarray get_step_density_operator(self, np.ndarray pulse_operators, np.ndarray init_state):
        cdef np.ndarray evolution = self.get_evolution(pulse_operators)
        cdef np.ndarray step_state
        if init_state is None:
            return evolution
        else:
            step_state = self.get_dot_product(evolution, init_state)
            return np.array([]) #nm.NumericalMethods.get_density_operator(step_state)

    
    cdef np.ndarray _get_rotation_matrix(self, float alpha, str axis):
        if axis == "x":
            return RotationHandler._get_x_rotation(alpha)
        elif axis == "y":
            return RotationHandler._get_y_rotation(alpha)
        else:
            return RotationHandler._get_z_rotation(alpha)
    
    cdef public np.ndarray get_target_state(self, np.ndarray angles, np.ndarray axes, np.ndarray init_state):
        angles, axes = angles[::-1], axes[::-1]
        cdef np.ndarray rotations_sequence = self._get_rotation_sequence(angles, axes)
        cdef np.ndarray target_rotation = self.get_evolution(rotations_sequence)
        cdef np.ndarray ideal_state = self.get_dot_product(target_rotation, init_state)
        return target_rotation, np.array([]), ideal_state#nm.NumericalMethods.get_density_operator(ideal_state), ideal_state

    cdef np.ndarray _get_rotation_sequence(self, np.ndarray angles, np.ndarray axes):
        return np.array([self._get_rotation_matrix(alpha, axis) for alpha, axis in zip(angles, axes)])
    
    cdef public np.ndarray get_dot_product(self, np.ndarray evolution_operator, np.ndarray init_state):
        return np.dot(evolution_operator, init_state)

    @staticmethod
    cdef np.ndarray get_state_stat(np.ndarray evolution_operator, np.ndarray init_state):
        return np.dot(evolution_operator, init_state)

    @staticmethod
    cdef np.ndarray _get_x_rotation(float alpha):
        return np.array([[np.cos(alpha / 2), -1j * np.sin(alpha / 2)],
                         [-1j * np.sin(alpha / 2), np.cos(alpha / 2)]])

    @staticmethod
    cdef np.ndarray _get_y_rotation(float alpha):
        return np.array([[np.cos(alpha / 2), - np.sin(alpha / 2)],
                         [np.sin(alpha / 2), np.cos(alpha / 2)]])

    @staticmethod
    cdef np.ndarray _get_z_rotation(float alpha):
        return np.array([[np.exp(-1j * alpha / 2), 0],
                         [0, np.exp(1j * alpha / 2)]])