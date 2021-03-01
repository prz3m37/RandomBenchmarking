import numpy as np


class SignalConverter:

    @staticmethod
    def extract_signal_chunks(perturbated_signal: np.array, pulses_num: int):
        return np.array_split(perturbated_signal, pulses_num)

    @staticmethod
    def get_pulse_mask(chunked_signal: np.array):
        const_pulse_mask = np.where(np.sum(np.diff(chunked_signal), axis=1) == 0)[0].size
        if const_pulse_mask == 0:
            return None
        else:
            return const_pulse_mask