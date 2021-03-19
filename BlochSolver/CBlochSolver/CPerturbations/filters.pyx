import numpy as np
cimport numpy as np

cdef class Filters:

    @staticmethod
    cdef np.ndarray extract_signal_chunks(np.ndarray perturbated_signal, int pulses_num):
        cdef np.ndarray signal_chunks = np.array_split(perturbated_signal, pulses_num)
        return np.array(signal_chunks)

    @staticmethod
    cdef np.ndarray _get_signal(np.ndarray pulses, int signal_len):
        cdef np.ndarray signal = np.ones(signal_len)
        cdef int n = int(signal_len / len(pulses))
        cdef np.ndarray ones = np.ones(n)

        for i, pulse in enumerate(pulses):
            signal[i * n:(i + 1) * n] = pulse * ones
        return signal

    @staticmethod
    cdef double _get_low_pass_filter(double pulse_time, double t_0, double cut_off_time, double amplitude, double pulse_amplitude):
        return amplitude - (amplitude - pulse_amplitude) * (1 - np.exp(-(pulse_time - t_0) / cut_off_time))

    @staticmethod
    cdef np.ndarray _get_time_domain(double pulse_time, int granulation, int signal_len):
        return np.linspace(0, signal_len * pulse_time, signal_len * granulation)
    
    @staticmethod
    cdef np.ndarray get_low_pass_pulses(np.ndarray pulses, double pulse_time, double cut_off_time, int granulation):
        cdef int signal_len = pulses.shape[0] * granulation
        cdef np.ndarray signal = Filters._get_signal(pulses, signal_len)
        cdef np.ndarray signal_f = np.ones(signal.shape[0])
        signal_f[-1] = signal[0]
        cdef np.ndarray t = Filters._get_time_domain(pulse_time, granulation, signal_len)
        cdef double t_0 = 0
        for i, A0 in enumerate(signal):
            if A0 != signal[i - 1]:
                t_0 = t[i]
            signal_f[i] = Filters._get_low_pass_filter(t[i], t_0, cut_off_time, signal_f[i - 1], signal[i])
        return signal_f 
    
    @staticmethod
    cdef np.ndarray filter_out_signal(np.ndarray signal_filtered, int n, int granulation):
        cdef np.ndarray pulses_f = signal_filtered.reshape(n, granulation)
        for pulse in pulses_f:
            pulse.fill(pulse[-1])
        return np.mean(pulses_f, axis=1)