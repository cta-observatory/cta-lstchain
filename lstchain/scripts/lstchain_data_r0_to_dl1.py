#!/usr/bin/env python3

"""
Pipeline to calibrate and compute image parameters at single telescope
level for real data.
- Inputs are a protozfits input file and a drs4 pedestal/calibration/time
calibration files
- Output is a dataframe with dl1 data

Usage:

$> python lstchain_data_r0_to_dl1.py
--input-file LST-1.1.Run02030.0000.fits.fz
--output-dir ./
--pedestal-file drs4_pedestal.Run2028.0000.fits
--calibration-file calibration.Run2029.0000.hdf5
--time-calibration-file time_calibration.Run2029.0000.hdf5

"""

import argparse
import logging
import sys
from pathlib import Path

from lstchain.io.config import read_configuration_file
from lstchain.paths import parse_r0_filename, run_to_dl1_filename, r0_to_dl1_filename
from lstchain.reco import r0_to_dl1

log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="R0 to DL1")

# Required arguments
parser.add_argument(
    '-f', '--input-file', type=Path,
    help='Path to the .fits.fz file with the raw events',
    required=True,
)

parser.add_argument(
    '-o', '--output-dir', type=Path,
    help='Path where to store the reco dl1 events',
    default='./dl1_data/'
)

parser.add_argument(
    '-p', '--pedestal-file', '-p', type=Path,
    dest='pedestal_file',
    help='Path to a pedestal file',
    required=True
)

parser.add_argument(
    '--calibration-file', '--calib', type=Path,
    help='Path to a calibration file',
    required=True,
)

parser.add_argument(
    '--time-calibration-file', '-t', type=Path,
    help='Path to a calibration file for pulse time correction',
    required=True
)

# Optional arguments
parser.add_argument(
    '--config', '-c', type=Path,
    dest='config_file',
    help='Path to a configuration file. If none is given, a standard configuration is applied',
)

parser.add_argument(
    '--pointing-file', '--pointing', action='store', type=Path,
    help='Path to the Drive log file with the pointing information.',
)

parser.add_argument(
    '--ucts-t0-dragon', type=int,
    help=(
        'UCTS timestamp in nsecs, unix format and TAI scale of the'
        ' first event of the run with valid timestamp. If none is'
        ' passed, the start-of-the-run timestamp is provided, hence'
        ' Dragon timestamp is not reliable.'
    ),
)

parser.add_argument(
    '--dragon-counter0', type=int,
    help=(
        'Dragon counter (pps + 10MHz) in nsecs corresponding'
        ' to the first reliable UCTS of the run. To be provided'
        ' along with ucts_t0_dragon.'
    ),
)

parser.add_argument(
    '--ucts-t0-tib', type=int,
    help=(
        'UCTS timestamp in nsecs, unix format and TAI scale of the'
        ' first event of the run with valid timestamp. If none is'
        ' passed, the start-of-the-run timestamp is provided, hence'
        ' TIB timestamp is not reliable.'
    ),
)

parser.add_argument(
    '--tib-counter0', type=int,
    help=(
        'First valid TIB counter (pps + 10MHz) in nsecs corresponding'
        ' to the first reliable UCTS of the run when TIB is available.'
        ' To be provided along with ucts_t0_tib.'
    ),
)

parser.add_argument(
    '--max-events', type=int,
    help='Maximum number of events to be processed.',
)

args = parser.parse_args()


def main():
    output_dir = args.output_dir.absolute()
    output_dir.mkdir(exist_ok=True)

    if not args.input_file.is_file():
        log.error('Input file does not exist or is not a file')
        sys.exit(1)

    log.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    logging.getLogger().addHandler(handler)

    r0_to_dl1.allowed_tels = {1, 2, 3, 4}

    # test if this matches data file name pattern
    try:
        run = parse_r0_filename(args.input_file)
        output_filename = output_dir / run_to_dl1_filename(run.tel_id,
                                                           run.run, run.subrun)
    except ValueError:
        # for arbitrary filenames, including mc
        output_filename = output_dir / r0_to_dl1_filename(args.input_file.name)

    config = {}
    if args.config_file is not None:
        try:
            config = read_configuration_file(args.config_file)
        except Exception as e:
            log.error(f'Configuration file could not be read: {e}')
            sys.exit(1)

    config["max_events"] = args.max_events

    r0_to_dl1.r0_to_dl1(
        args.input_file,
        output_filename=output_filename,
        custom_config=config,
        pedestal_path=args.pedestal_file,
        calibration_path=args.calibration_file,
        time_calibration_path=args.time_calibration_file,
        pointing_file_path=args.pointing_file,
        ucts_t0_dragon=args.ucts_t0_dragon,
        dragon_counter0=args.dragon_counter0,
        ucts_t0_tib=args.ucts_t0_tib,
        tib_counter0=args.tib_counter0
    )



if __name__ == '__main__':
    main()
