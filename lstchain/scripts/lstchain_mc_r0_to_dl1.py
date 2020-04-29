#!/usr/bin/env python3

"""
Pipeline to calibrate and compute image parameters at single telescope 
level for MC.
- Inputs are simtelarray files.
- Output is a dataframe with dl1 data.

Usage: 

$> python lstchain_mc_r0_to_dl1.py 
--input-file gamma_20deg_0deg_run8___cta-prod3-lapalma-2147m-LaPalma-FlashCam.simtel.gz

"""

import argparse
from ctapipe.utils import get_dataset_path
from lstchain.reco import r0_to_dl1
from lstchain.io.config import read_configuration_file
import os

parser = argparse.ArgumentParser(description="R0 to DL1")

# Required arguments
parser.add_argument('--input-file', '-f', type=str,
                    dest='input_file',
                    help='Path to the simtelarray file',
                    default=get_dataset_path('gamma_test_large.simtel.gz'))

# Optional arguments
parser.add_argument('--output-dir', '-o', action='store', type=str,
                    dest='output_dir',
                    help='Path where to store the reco dl2 events',
                    default='./dl1_data/')

parser.add_argument('--config-file', '-c', action='store', type=str,
                    dest='config_file',
                    help='Path to a configuration file. If none is given, a standard configuration is applied',
                    default=None
                    )


args = parser.parse_args()


def main():

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    r0_to_dl1.allowed_tels = {1, 2, 3, 4}
    output_filename = os.path.join(
        output_dir, 'dl1_' + os.path.basename(
            args.input_file).rsplit('.', 1)[0] + '.h5')

    config = {}
    if args.config_file is not None:
        try:
            config = read_configuration_file(os.path.abspath(args.config_file))
        except("Custom configuration could not be loaded !!!"):
            pass

    r0_to_dl1.r0_to_dl1(args.input_file, output_filename=output_filename, custom_config=config)


if __name__ == '__main__':
    main()
