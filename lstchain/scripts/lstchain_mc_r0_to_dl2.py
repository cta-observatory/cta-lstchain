#!/usr/bin/env python3

"""
Pipeline to calibrate and compute image parameters at single telescope 
level for MC.

Inputs are simtelarray files and trained Random Forests.
Output is a dataframe with DL2 data.

Usage:

$> python lstchain_mc_r0_to_dl2.py
--input-file gamma_20deg_0deg_run8___cta-prod3-lapalma-2147m-LaPalma-FlashCam.simtel.gz
--path-models ./trained_models

"""

from ctapipe.utils import get_dataset_path
import argparse
import os
from distutils.util import strtobool
from lstchain.paths import r0_to_dl1_filename
from pathlib import Path

parser = argparse.ArgumentParser(description="MC Pipeline R0 to DL2.")

# Required arguments
parser.add_argument('--input-file', '-f', type=Path,
                    dest='datafile',
                    help='path to the file with simtelarray events',
                    default=get_dataset_path('gamma_test_large.simtel.gz'))

parser.add_argument('--path-models', '-p', action='store', type=str,
                    dest='path_models',
                    help='Path where to find the trained RF',
                    default='./trained_models')

# Optional argument
parser.add_argument('--store-dl1', '-s1', action='store', type=lambda x: bool(strtobool(x)),
                    dest='store_dl1',
                    help='Boolean. True for storing DL1 file'
                    'Default=False, use True otherwise',
                    default=True)

parser.add_argument('--output-dir', '-o', type=Path,
                    dest='outdir',
                    help='Path where to store the reco dl2 events',
                    default='./dl2_data')

parser.add_argument('--config', '-c', action='store', type=str,
                    dest='config_file',
                    help='Path to a configuration file. If none is given, a standard configuration is applied',
                    default=None
                    )

args = parser.parse_args()


def main():
    outdir = args.outdir.absolute()
    dl1_file = outdir / r0_to_dl1_filename(args.datafile.name)

    cmd_r0_to_dl1 = f'lstchain_mc_r0_to_dl1 -f {args.datafile} -o {outdir}'
    if args.config_file is not None:
        cmd_r0_to_dl1 = cmd_r0_to_dl1 + f' -conf {args.config_file}'

    cmd_dl1_to_dl2 = f'lstchain_dl1_to_dl2 -f {dl1_file} -p {args.path_models} -o {outdir}'
    if args.config_file is not None:
        cmd_dl1_to_dl2 = cmd_dl1_to_dl2 + f' -conf {args.config_file}'

    os.system(cmd_r0_to_dl1)
    os.system(cmd_dl1_to_dl2)

    if not args.store_dl1:
        os.remove(dl1_file)


if __name__ == '__main__':
    main()
