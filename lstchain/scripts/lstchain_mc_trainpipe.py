#!/usr/bin/env python3

"""
Pipeline to train three Random Forests destinated to Energy, disp
reconstruction and Gamma/Hadron separation.
The resulting RF models can be stored in files for later use on data.

Inputs are DL1 gamma and proton files
Outputs are RF trained models

Usage:

$> python lst-trainpipe 
--input_file_gamma dl1_gamma_20deg_180deg_cta-prod3-demo-2147m-LaPalma-baseline-mono_off0.4_merge_test.h5 
--input-file-proton dl1_proton_20deg_180degcta-prod3-demo-2147m-LaPalma-baseline-mono_merge_test.h5 

"""

import argparse
from lstchain.reco import dl1_to_dl2
from distutils.util import strtobool
from lstchain.io.config import read_configuration_file

parser = argparse.ArgumentParser(description="Train Random Forests.")

# Required argument
parser.add_argument('--input-file-gamma', '--fg', type=str,
                    dest='gammafile',
                    help='Path to the dl1 file of gamma events for training')

parser.add_argument('--input-file-proton', '--fp', type=str,
                    dest='protonfile',
                    help='Path to the dl1 file of proton events for training')

# Optional arguments
parser.add_argument('--store-rf', '-s', action='store', type=lambda x: bool(strtobool(x)),
                    dest='storerf',
                    help='Boolean. True for storing trained models in 3 files'
                    'Default=True, use False otherwise',
                    default=True)

parser.add_argument('--output-dir', '-o', action='store', type=str,
                     dest='path_models',
                     help='Path to store the resulting RF',
                     default='./trained_models/')

parser.add_argument('--config', '-c', action='store', type=str,
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
