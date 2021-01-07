from Utils import utils
import angles_solver


def main():
    utilities = utils.Utils
    utilities.initialize_utilities()

    theta, phi = utilities.get_rotation_params()
    aslv = angles_solver.AnglesSolver(theta, phi)

    utilities.release_utilities()
    del aslv
    return 1


if __name__ == '__main__':
    main()
