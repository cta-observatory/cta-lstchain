"""Pipeline for training three Random Forests destinated to Energy, disp_
reconstruction and Gamma/Hadron separation.
The resulting RF models can be stored in files for later use on data.

Usage:

$> python lst-trainpipe arg1 arg2 ...

"""

import argparse
from lstchain.reco import dl1_to_dl2
from distutils.util import strtobool
from lstchain.io.config import read_configuration_file

parser = argparse.ArgumentParser(description="Train Random Forests.")

# Required argument
parser.add_argument('--gammafile', '-fg', type=str,
                    dest='gammafile',
                    help='path to the dl1 file of gamma events for training')

parser.add_argument('--protonfile', '-fp', type=str,
                    dest='protonfile',
                    help='path to the dl1 file of proton events for training')

parser.add_argument('--storerf', '-s', action='store', type=lambda x: bool(strtobool(x)),
                    dest='storerf',
                    help='Boolean. True for storing trained models in 3 files'
                    'Deafult=True, use False otherwise',
                    default=True)

# Optional arguments
parser.add_argument('--opath', '-o', action='store', type=str,
                     dest='path_models',
                     help='Path to store the resulting RF',
                     default='./trained_models/')

parser.add_argument('--config_file', '-conf', action='store', type=str,
                    dest='config_file',
                    help='Path to a configuration file. If none is given, a standard configuration is applied',
                    default=None
                    )

args = parser.parse_args()

def main():
    #Train the models
        
    config = {}
    if args.config_file is not None:
        try:
            config = read_configuration_file(args.config_file)
        except("Custom configuration could not be loaded !!!"):
            pass

    dl1_to_dl2.build_models(args.gammafile,
                            args.protonfile,
                            save_models=args.storerf,
                            path_models=args.path_models,
                            custom_config=config,
                            )


if __name__ == '__main__':
    main()
