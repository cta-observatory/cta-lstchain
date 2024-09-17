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

import argparse
import os
from pathlib import Path
import subprocess as sp

from ctapipe.utils import get_dataset_path

from lstchain.paths import r0_to_dl1_filename

parser = argparse.ArgumentParser(description="MC Pipeline R0 to DL2.")

# Required arguments
parser.add_argument(
    '--input-file', '-f', type=Path,
    dest='datafile',
    help='path to the file with simtelarray events',
)

parser.add_argument(
    '--path-models', '-p',
    help='Path where to find the trained RF',
    default='./trained_models'
)

# Optional argument
parser.add_argument(
    '--no-dl1',
    action='store_true',
    help='If given, the dl1 file is removed after creating the dl2 output',
)

parser.add_argument(
    '--output-dir', '-o', type=Path,
    help='Path where to store the reco dl2 events',
    default='./dl2_data'
)

parser.add_argument(
    '--config', '-c',
    dest='config_file',
    help='Path to a configuration file. If none is given, a standard configuration is applied',
)


def main():
    args = parser.parse_args()

    # using a default of None and only using get_dataset_path here
    # prevents downloading when an input file is actually given
    # or just --help is called.
    if args.datafile is None:
        args.datafile = get_dataset_path('gamma_lstprod2.simtel.gz')

    output_dir = args.output_dir.absolute()
    dl1_file = output_dir / r0_to_dl1_filename(args.datafile.name)

    cmd_r0_to_dl1 = [
        'lstchain_mc_r0_to_dl1',
        '-f', str(args.datafile),
        '-o', str(output_dir),
    ]
    if args.config_file is not None:
        cmd_r0_to_dl1.extend(['--config', str(args.config_file)])

    cmd_dl1_to_dl2 = [
        'lstchain_dl1_to_dl2',
        '-f', str(dl1_file),
        '-p', str(args.path_models),
        '-o', str(output_dir),
    ]
    if args.config_file is not None:
        cmd_dl1_to_dl2.extend(['--config', str(args.config_file)])

    sp.run(cmd_r0_to_dl1, check=True)
    sp.run(cmd_dl1_to_dl2, check=True)

    if args.no_dl1:
        os.remove(dl1_file)


if __name__ == '__main__':
    main()
