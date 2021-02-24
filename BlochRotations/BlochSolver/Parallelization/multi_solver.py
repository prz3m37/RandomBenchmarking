import multiprocessing as mp


class MultiSolver:

    def __init__(self, n: int = None):
        self._processes = []
        self._num_of_processes = None
        self._max_cpu = mp.cpu_count()
        self._set_num_of_processes(n)

    def __del__(self):
        pass

    def parallelize_solvers(self, *solvers):
        return

    def parallelize_solver_functions(self, *solver_functions):
        self._start_solver_functions(*solver_functions)
        self._stop_solver_functions()
        return

    def _start_solver_functions(self, *solver_functions):
        proc_num = 0
        while proc_num <= self._num_of_processes:
            process = mp.Process(target=solver_functions[proc_num])
            process.start()
            self._processes.append(process)
            proc_num += 1
        return

    def _stop_solver_functions(self):
        for proc in self._processes:
            proc.join()
        return

    def _set_num_of_processes(self, n: int = None):
        if (n is None) or (n > self._max_cpu):
            self._num_of_processes = self._max_cpu - 1
        else:
            self._num_of_processes = n
        return
