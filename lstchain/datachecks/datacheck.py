import argparse
from lstchain.datachecks import plot_dl1b


def main():
    plot_dl1b.plot_dl1_params(args.infile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data-check plots")

    requiredNamed = parser.add_argument_group('required arguments')

    requiredNamed.add_argument("--infile", '-f', type=str,
                               dest='infile',
                               help="Path to the DL1 input file",
                               required=True)

    args = parser.parse_args()

    main()
