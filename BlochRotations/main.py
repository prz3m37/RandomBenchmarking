from BlochSolver import levenberg_marquardt
from Utils import utils


def main():
    utilities = utils.Utils
    utilities.initialize_utilities()
    lma = levenberg_marquardt.LevenbergMarquardtSolver()
    lma.get_lma_bloch()
    utilities.release_utilities()
    return


if __name__ == '__main__':
    main()
    exit()
