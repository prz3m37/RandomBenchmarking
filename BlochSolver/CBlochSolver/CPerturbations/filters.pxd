cimport numpy as np

cdef class Filters:

    @staticmethod
    cdef np.ndarray extract_signal_chunks(np.ndarray , int)

    @staticmethod
    cdef np.ndarray _get_signal(np.ndarray, int)

    @staticmethod
    cdef double _get_low_pass_filter(double, double, double, double, double)

    @staticmethod
    cdef np.ndarray _get_time_domain(double, int, int)

    @staticmethod
    cdef np.ndarray get_low_pass_pulses(np.ndarray, double, double, int)

    @staticmethod
    cdef np.ndarray filter_out_signal(np.ndarray, int, int)