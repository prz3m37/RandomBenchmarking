import numpy as np


class Filters:
    signal_len = 0

    @staticmethod
    def _get_signal(pulses):
        signal = np.ones(Filters.signal_len)
        n = int(Filters.signal_len / len(pulses))
        ones = np.ones(n)
        for i, A in enumerate(pulses):
            signal[i * n:(i + 1) * n] = A * ones
        return signal

    @staticmethod
    def _get_low_pass_filter(pulse_time, t_0, cut_off_time, amplitude, pulse_amplitude):
        return amplitude - (amplitude - pulse_amplitude) * (1 - np.exp(-(pulse_time - t_0) / cut_off_time))

    @staticmethod
    def _get_time_domain(pulse_time, duration):
        return np.linspace(0, duration * pulse_time, Filters.signal_len)

    @staticmethod
    def get_low_pass_pulses(pulses, pulse_time, cut_off_time, granulation, duration):
        Filters.signal_len = pulses.shape[0] * granulation
        signal = Filters._get_signal(pulses)
        signal_f = np.ones(signal.shape[0])
        signal_f[-1] = signal[0]
        t = Filters._get_time_domain(pulse_time, duration)
        t_0 = 0
        for i, A0 in enumerate(signal):
            if A0 != signal[i - 1]:
                t_0 = t[i]
            signal_f[i] = Filters._get_low_pass_filter(t[i], t_0, cut_off_time, signal_f[i - 1], signal[i])
        return signal_f, signal
