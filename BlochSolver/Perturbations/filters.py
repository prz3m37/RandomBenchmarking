import numpy as np

class Filters:

    @staticmethod
    def extract_signal_chunks(perturbated_signal: np.array, pulses_num: int):
        signal_chunks = np.array_split(perturbated_signal, pulses_num)
        return np.array(signal_chunks)

    def _get_signal(pulses, signal_len):
        signal = np.ones(signal_len)
        n = int(signal_len / len(pulses))
        ones = np.ones(n)
        for i, pulse in enumerate(pulses):
            signal[i * n:(i + 1) * n] = pulse * ones
        return signal

    def _get_low_pass_filter(pulse_time, t_0, cut_off_time, amplitude, pulse_amplitude):
        return amplitude - (amplitude - pulse_amplitude) * (1 - np.exp(-(pulse_time - t_0) / cut_off_time))

    @staticmethod
    def _get_time_domain(pulse_time, granulation, signal_len):
        return np.linspace(0, signal_len * pulse_time, signal_len * granulation)

    @staticmethod
    def get_low_pass_pulses(pulses, pulse_time, cut_off_time, granulation, return_original_signal: bool = False):
        signal_len = pulses.shape[0] * granulation
        signal = Filters._get_signal(pulses, signal_len)
        signal_f = np.ones(signal.shape[0])
        signal_f[-1] = signal[0]
        t = Filters._get_time_domain(pulse_time, granulation, signal_len)
        t_0 = 0
        for i, A0 in enumerate(signal):
            if A0 != signal[i - 1]:
                t_0 = t[i]
            signal_f[i] = Filters._get_low_pass_filter(t[i], t_0, cut_off_time, signal_f[i - 1], signal[i])
        if return_original_signal:
            return signal_f, signal
        else:
            return signal_f

    @staticmethod
    def filter_out_signal(signal_filtered: np.array, n: int, granulation: int):
        pulses_f = signal_filtered.reshape(n, granulation)
        for pulse in pulses_f:
            pulse.fill(pulse[-1])
        return np.mean(pulses_f, axis=1)