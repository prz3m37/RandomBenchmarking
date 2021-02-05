from BlochSolver.Plotter import plotter_converter as pc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


class BlochPlotter:

    def __init__(self):
        self.__size = None
        self.__x_axis = None

    def plot(self, pulses_final: np.array, plot_type: str = None, **kwargs):
        self.__size = len(pulses_final)
        self.__x_axis = np.arange(1, self.__size + 1, 1)
        if plot_type == "evolution":
            pulse_vectors = pc.PlotterConverter.convert_bloch_coordinates(pulses_final, **kwargs)
            self.__plot_evolution(pulse_vectors)
        elif plot_type == "numerical":
            self.__plot_numerical_data(**kwargs)
        elif plot_type == "pulses":
            self.__plot_pulses_diff(pulses_final=pulses_final, **kwargs)
        else:
            return

    def __plot_evolution(self, pulse_vectors: np.array):
        return

    def __plot_pulses_diff(self, pulses_init: np.array, pulses_final: np.array):
        diff = pulses_init - pulses_final
        sns.set_style("dark")
        plt.figure(figsize=(17, 12))
        plt.subplots_adjust(hspace=0.3)

        plt.subplot(311)
        plt.title("Initial pulses")
        plt.ylabel("\u03B5")
        sns.barplot(y=pulses_init, x=self.__x_axis, edgecolor='black', color="steelblue")
        plt.xticks(rotation=45)

        plt.subplot(312)
        plt.title("Final pulses")
        plt.ylabel("\u03B5")
        sns.barplot(y=pulses_final, x=self.__x_axis, edgecolor='black', color="green")
        plt.xticks(rotation=45)

        plt.subplot(313)
        plt.title("Pulses difference")
        plt.ylabel("\u0394 \u03B5")
        plt.xlabel("steps [N]")
        sns.barplot(y=diff, x=self.__x_axis, edgecolor='black', color="darkred")
        plt.xticks(rotation=45)
        plt.show()
        return

    @staticmethod
    def __plot_numerical_data(fidelities, learning_rate, iterations):
        plt.figure(figsize=(17, 12))

        plt.subplot(211)
        plt.title("Fidelity")
        plt.ylabel("Fidelity [a.u]")
        plt.plot(iterations, fidelities, "--D")
        plt.xticks(rotation=45)

        plt.subplot(212)
        plt.title("Learning rates")
        plt.ylabel("Learning rate [a.u]")
        plt.xlabel("steps [N]")
        plt.plot(iterations, learning_rate, "--D")
        plt.xticks(rotation=45)
        plt.show()
        return
