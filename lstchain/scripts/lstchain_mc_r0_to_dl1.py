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
import logging
import sys
from pathlib import Path

from ctapipe.utils import get_dataset_path

from lstchain.io.config import read_configuration_file
from lstchain.paths import r0_to_dl1_filename
from lstchain.reco import r0_to_dl1

log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="R0 to DL1")

# Required arguments
parser.add_argument(
    '--input-file', '-f', type=Path,
    dest='input_file',
    help='Path to the simtelarray file',
)

# Optional arguments
parser.add_argument('--output-dir', '-o', action='store', type=Path,
                    dest='output_dir',
                    help='Path where to store the reco dl1 events',
                    default='./dl1_data/')

parser.add_argument('--config', '-c', action='store', type=Path,
                    dest='config_file',
                    help='Path to a configuration file. If none is given, a standard configuration is applied',
                    default=None
                    )


def main():

    args = parser.parse_args()

    # using a default of None and only using get_dataset_path here
    # prevents downloading when an input file is actually given
    # or just --help is called.
    if args.input_file is None:
        args.input_file = get_dataset_path('gamma_lstprod2.simtel.gz')

    output_dir = args.output_dir.absolute()
    output_dir.mkdir(exist_ok=True, parents=True)
    output_file = output_dir / r0_to_dl1_filename(args.input_file.name)

    r0_to_dl1.allowed_tels = {1, 2, 3, 4}

    config = {}
    if args.config_file is not None:
        try:
            config = read_configuration_file(args.config_file.absolute())
        except Exception as e:
            log.error(f'Config file {args.config_file} could not be read: {e}')
            sys.exit(1)

    r0_to_dl1.r0_to_dl1(
        args.input_file,
        output_filename=output_file,
        custom_config=config,
    )


if __name__ == '__main__':
    main()
