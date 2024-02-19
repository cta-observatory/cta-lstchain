#!/usr/bin/env python3

"""
Usage:

$> python lstchain_tune_nsb_waveform.py
--config  config_file.json     (must be the one used in the DL1 production)
--input-mc  simtel_file.simtel.gz    simulation simtel file
--input-data dl1_data.h5         real data DL1 file

Calculates the parameters needed to tune the NSB in the waveforms (in the
R0 to DL1 stage) to the level of NSB in a given data file

"""

import argparse
import json
import logging
import sys
import numpy as np
from pathlib import Path

from lstchain.image.modifier import calculate_required_additional_nsb
from lstchain.io.config import dump_config, read_configuration_file
from traitlets.config import Config

log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Tune NSB in waveforms")

# Required arguments
parser.add_argument(
    '--config', type=Path,
    help='Path to the configuration file for the production (must be the '
         'one used for calibration and DL1 creation)',
    required=True,
)

parser.add_argument(
    '--input-mc', type=Path,
    help='Path to a simtel file of the production (must include the true '
         'p.e. images)',
    required=True,
)

parser.add_argument(
    '--input-data', type=Path,
    help='Path to a data DL1 file of the production (must include DL1a)',
    required=True,
)

parser.add_argument(
    '--output-file', '-o',
    type=Path,
    help='Path to a output file where to dump the update config',
)

parser.add_argument(
    '--overwrite',
    action='store_true',
    help='Use to overwrite output-file',
)


def main():
    args = parser.parse_args()

    if not args.config.is_file():
        log.error('Config file does not exist or is not a file')
        sys.exit(1)
    if not args.input_mc.is_file():
        log.error('MC simtel file does not exist or is not a file')
        sys.exit(1)
    if not args.input_data.is_file():
        log.error('DL1 data file does not exist or is not a file')
        sys.exit(1)

    log.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    logging.getLogger().addHandler(handler)

    config = read_configuration_file(args.config)

    nsb_correction_ratio, data_ped_variance, mc_ped_variance = \
        calculate_required_additional_nsb(args.input_mc, args.input_data,
                                          config=Config(config))

    dict_nsb = {
        "nsb_tuning": True,
        "nsb_tuning_ratio": np.round(nsb_correction_ratio, decimals=2),
        "spe_location": "lstchain/data/SinglePhE_ResponseInPhE_expo2Gaus.dat"
    }

    log.info(f'\ndata_ped_stdev: {data_ped_variance**0.5:.3f} p.e.')
    log.info(f'mc_ped_stdev: {mc_ped_variance**0.5:.3f} p.e.\n')

    log.info(json.dumps(dict_nsb, indent=2))
    log.info('\n')

    if args.output_file:
        cfg = read_configuration_file(args.config)
        cfg['waveform_nsb_tuning'].update(dict_nsb)
        dump_config(cfg, args.output_file, overwrite=args.overwrite)


if __name__ == '__main__':
    main()
