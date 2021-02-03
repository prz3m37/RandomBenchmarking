import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


class SolverPlotter:

    def plot_evolution(self):
        return

    @staticmethod
    def plot_pulses(pulses: np.array):
        size = len(pulses)
        x_axis = np.arange(1, size+1, 1)
        sns.set_style("dark")
        plt.figure(figsize=(15, 10))
        sns.barplot(y=pulses, x=x_axis, edgecolor='black', color="grey")
        plt.xticks(rotation=45)
        plt.show()
        return

    def plot_errors(self):
        return

    def plot_learning_rate_curve(self):
        return


