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
parser.add_argument('--input-file', '-f', type=Path,
                    dest='input_file',
                    help='Path to the .fits.fz file with the raw events',
                    default=None, required=True)

parser.add_argument('--output-dir', '-o', type=Path,
                    dest='output_dir',
                    help='Path where to store the reco dl1 events',
                    default='./dl1_data/')

parser.add_argument('--pedestal-file', '-p', action='store', type=str,
                    dest='pedestal_file',
                    help='Path to a pedestal file',
                    default=None, required=True
                    )

parser.add_argument('--calibration-file', '--calib', action='store', type=str,
                    dest='calibration_file',
                    help='Path to a calibration file',
                    default=None, required=True
                    )

parser.add_argument('--time-calibration-file', '-t', action='store', type=str,
                    dest='time_calibration_file',
                    help='Path to a calibration file for pulse time correction',
                    default=None, required=True
                    )

# Optional arguments
parser.add_argument('--config', '-c', action='store', type=str,
                    dest='config_file',
                    help='Path to a configuration file. If none is given, a standard configuration is applied',
                    default=None
                    )

parser.add_argument('--pointing-file', '--pointing', action='store', type=str,
                    dest='pointing_file',
                    help='Path to the Drive log file with the pointing information.',
                    default=None
                    )

parser.add_argument('--ucts-t0-dragon', action='store', type=float,
                    dest='ucts_t0_dragon',
                    help='UCTS timestamp in nsecs, unix format and TAI scale of the \
                          first event of the run with valid timestamp. If none is \
                          passed, the start-of-the-run timestamp is provided, hence \
                          Dragon timestamp is not reliable.',
                    default="NaN"
                    )

parser.add_argument('--dragon-counter0', action='store', type=float,
                    dest='dragon_counter0',
                    help='Dragon counter (pps + 10MHz) in nsecs corresponding \
                          to the first reliable UCTS of the run. To be provided \
                          along with ucts_t0_dragon.',
                    default="NaN"
                    )

parser.add_argument('--ucts-t0-tib', action='store', type=float,
                    dest='ucts_t0_tib',
                    help='UCTS timestamp in nsecs, unix format and TAI scale of the \
                          first event of the run with valid timestamp. If none is \
                          passed, the start-of-the-run timestamp is provided, hence \
                          TIB timestamp is not reliable.',
                    default="NaN"
                    )

parser.add_argument('--tib-counter0', action='store', type=float,
                    dest='tib_counter0',
                    help='First valid TIB counter (pps + 10MHz) in nsecs corresponding \
                          to the first reliable UCTS of the run when TIB is available. \
                          To be provided along with ucts_t0_tib.',
                    default="NaN"
                    )

parser.add_argument('--max-events', '--maxevts', action='store', type=int,
                    dest='max_events',
                    help='Maximum number of events to be processed.',
                    default=int(1e15)
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
