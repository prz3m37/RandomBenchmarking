cimport numpy as np
import numpy as np

cdef class RotationHandler:

    cdef public np.ndarray get_control_hamiltonian(self)
    cdef public np.ndarray get_pulse_detuning(self, float)
    cdef public np.ndarray get_pulse_detunings(self, np.ndarray)
    cdef public np.ndarray get_pulse_args(self, np.ndarray)
    cdef public np.ndarray get_dot_product(self, np.ndarray, np.ndarray)
    cdef public np.ndarray get_rotation_operators(self, np.ndarray)
    cdef public np.ndarray get_perturbation_rotation_operators(self, np.ndarray, int)
    cdef public np.ndarray get_evolution(self, np.ndarray)
    cdef public np.ndarray get_step_density_operator(self, np.ndarray, np.ndarray)
    cdef public np.ndarray get_target_state(self, np.ndarray, np.ndarray, np.ndarray)

    cdef np.ndarray _get_pulse_arg(self, float)
    cdef np.ndarray _get_rotation_operator(self, float)
    cdef np.ndarray _get_rotation_sequence(self, np.ndarray, np.ndarray)
    cdef np.ndarray _get_rotation_matrix(self, float, str)


    @staticmethod
    cdef np.ndarray get_state_stat(np.ndarray, np.ndarray)

    @staticmethod
    cdef np.ndarray _get_x_rotation(float)

    @staticmethod
    cdef np.ndarray _get_y_rotation(float)

    @staticmethod
    cdef np.ndarray _get_z_rotation(float)
