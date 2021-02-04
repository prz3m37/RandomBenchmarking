from BlochSolver import solvers
from Utils import utils


def main():
    # utilities = utils.Utils
    # utilities.initialize_utilities()

    pulse_solvers = solvers.Solvers()
    pulse_solvers.get_solver("GRAPE")
    # utilities.release_utilities()
    return


if __name__ == '__main__':
    main()
    exit()
