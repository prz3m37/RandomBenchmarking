import numpy as np

from BlochSolver.QuantumSolvers.numerics import numerical_methods as nm
from BlochSolver.QuantumSolvers.rotations import rotation_handler as rh


class PlotterConverter(rh.RotationHandler, nm.NumericalMethods):

    @classmethod
    def convert_bloch_coordinates(cls, pulses: np.array, init_state: np.array, target_state: np.array):
        pulses = pulses[::-1]
        bloch_states = cls.get_pulse_states(pulses, init_state)
        init_vector = cls.get_real_vector(init_state)
        target_final_state = cls.get_real_vector(target_state)
        real_vectors = cls.get_pulse_real_vectors(bloch_states)
        real_vectors = np.concatenate((np.array([init_vector]), real_vectors), axis=0)
        return init_vector.T, target_final_state.T, real_vectors.T

    @classmethod
    def get_real_vector(cls, bloch_state: np.array):
        density_m = cls.get_density_operator(bloch_state)
        x = 2 * np.real(density_m[0][1])
        y = 2 * np.imag(density_m[1][0])
        z = density_m[0][00] - density_m[1][1]
        return np.array([x, y, z])

    @classmethod
    def get_pulse_real_vectors(cls, bloch_states: np.array):
        return np.array([cls.get_real_vector(bloch_state) for bloch_state in bloch_states])

    @classmethod
    def get_pulse_states(cls, pulses: np.array, init_state: np.array):
        n = len(pulses)
        rotation_operators = cls.get_rotation_operators(pulses)
        evolution_operators = np.array([cls.get_evolution(rotation_operators[:step + 1])
                                        for step in range(n)])
        bloch_states = np.array([cls.get_dot_product(evolution_operator, init_state)
                                 for evolution_operator in evolution_operators])
        return bloch_states
