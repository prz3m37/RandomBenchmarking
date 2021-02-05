from BlochSolver.QuantumSolvers.rotations import rotation_handler as rh
from BlochSolver.QuantumSolvers.numerics import numerical_methods as nm
import numpy as np


class PlotterConverter(rh.RotationHandler, nm.NumericalMethods):

    @classmethod
    def convert_bloch_coordinates(cls, pulses: np.array, init_state: np.array):
        density_operators = cls.get_pulse_evolution(pulses, init_state)
        return cls.get_pulse_real_vectors(density_operators)

    @classmethod
    def get_real_vector(cls, density_operator: np.array):
        x = 2 * np.real(density_operator[0][1])
        y = 2 * np.imag(density_operator[1][0])
        z = density_operator[0][0] - density_operator[1][1]
        return np.array([x, y, z])

    @classmethod
    def get_pulse_real_vectors(cls, density_operators):
        return np.fromiter((cls.get_real_vector(density_operator) for density_operator in density_operators), np.complex)

    @classmethod
    def get_pulse_evolution(cls, pulses: np.array, init_state: np.array):
        e_operators = cls.get_pulse_operators(pulses)
        e_states = np.fromiter((cls.get_state(e_operator, init_state)
                                for e_operator in e_operators), np.complex)
        e_density_operators = np.fromiter((cls.get_density_operator(state)
                                           for state in e_states), np.complex)
        return e_density_operators
