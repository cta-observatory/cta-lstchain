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

from lstchain.io.config import read_configuration_file
from lstchain.reco import dl1_to_dl2

parser = argparse.ArgumentParser(description="Train Random Forests.")

# Required argument
parser.add_argument(
    '--input-file-gamma', '--fg',
    dest='gammafile',
    required=True,
    help='Path to the dl1 file of gamma events for training'
)

parser.add_argument(
    '--input-file-proton', '--fp',
    dest='protonfile',
    required=True,
    help='Path to the dl1 file of proton events for training',
)

# Optional arguments
parser.add_argument(
    '--no-save-models',
    dest='save_models',
    action='store_false',
    help='Disable storing trained models',
)

parser.add_argument(
    '--output-dir', '-o',
    dest='path_models',
    default='./trained_models/',
    help='Path to store the resulting RF',
)

parser.add_argument(
    '--config', '-c',
    dest='config_file',
    help='Path to a configuration file. If none is given, a standard configuration is applied',
)


def main():
    args = parser.parse_args()

    #Train the models

    config = {}
    if args.config_file is not None:
        config = read_configuration_file(args.config_file)

    dl1_to_dl2.build_models(
        args.gammafile,
        args.protonfile,
        save_models=args.save_models,
        path_models=args.path_models,
        custom_config=config,
    )


if __name__ == '__main__':
    main()
