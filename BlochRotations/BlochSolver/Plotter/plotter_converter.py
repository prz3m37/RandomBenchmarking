from BlochSolver.QuantumSolvers.rotations import rotation_handler as rh
from BlochSolver.QuantumSolvers.numerics import numerical_methods as nm
import numpy as np


class PlotterConverter(rh.RotationHandler, nm.NumericalMethods):

    @classmethod
    def convert_bloch_coordinates(cls, pulses: np.array, init_state: np.array, target_state: np.array):
        bloch_states = cls.get_pulse_states(pulses, init_state)
        init_vector = cls.get_real_vector(init_state)
        target_final_state = cls.get_real_vector(target_state)
        real_vectors = cls.get_pulse_real_vectors(bloch_states)
        real_vectors = np.concatenate((np.array([init_vector]), real_vectors), axis=0)
        return init_vector, target_final_state, real_vectors.T

    @classmethod
    def get_real_vector(cls, bloch_state: np.array):
        alpha = 2 * np.arccos(np.real(bloch_state[0]))
        if alpha == 0:
            phi = 0
        else:
            phi = np.real(- 1j * np.log(bloch_state[1] / np.sin(alpha / 2)))
        x = np.cos(phi) * np.sin(alpha)
        y = np.sin(phi) * np.sin(alpha)
        z = np.cos(alpha)
        return np.array([x, y, z])

    @classmethod
    def get_pulse_real_vectors(cls, bloch_states: np.array):
        return np.array([cls.get_real_vector(bloch_state) for bloch_state in bloch_states])

    @classmethod
    def get_pulse_states(cls, pulses: np.array, init_state: np.array):
        N = len(pulses)
        pulse_operators = cls.get_pulse_operators(pulses)
        evolution_operators = np.array([cls.get_evolution(pulse_operators[:step + 1])
                                        for step in range(N)])
        bloch_states = np.array([cls.get_state(evolution_operator, init_state)
                                 for evolution_operator in evolution_operators])
        return bloch_states
