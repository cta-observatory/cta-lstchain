#!/usr/bin/env python3

"""
Script to create drs4 time correction coefficients

- Input: fits.fz file
- Output: time_calibration.hdf5 file

Usage:
$> python lstchain_data_create_time_calibration_file.py
--input-file LST-1.1.Run01625.0000.fits.fz or input-file LST-1.1.Run01625.000*.fits.fz
--output-file time_calibration.Run1625.0000.hdf5

"""

import argparse
import glob
import logging
import numpy as np
from traitlets.config.loader import Config
from ctapipe.io import EventSource
from lstchain.io.config import read_configuration_file
from lstchain.calib.camera.time_correction_calculate import TimeCorrectionCalculate

log = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

# Required arguments
parser.add_argument("--input-file", action='store', type=str,
                    dest='input_file',
                    help="Path to fits.fz file used to create the time calibration file. \
                    Allowed to use regular expression in given path to process subruns",
                    default=None, required=True)

parser.add_argument("--output-file", action='store', type=str,
                    dest='output_file',
                    help="Path where script creates the time calibration file",
                    default=None, required=True)

# Optional argument
parser.add_argument("--max-events",
                    dest='max_events',
                    help="Maximum numbers of events to read. Default = 20000",
                    type=int,
                    default=20000)

parser.add_argument('--config', '-c', action='store', type=str,
                    dest='config_file',
                    help='Path to a configuration file. If none is given, a standard configuration is applied',
                    default=None
                    )

parser.add_argument('--pedestal-file', '-p', action='store', type=str,
                    dest='pedestal_file',
                    help='Path to drs4 pedestal file ',
                    default=None
                    )
parser.add_argument(
    '--dragon-reference-time', type=int,
    help=(
        'UCTS timestamp in nsecs, unix format and TAI scale of the'
        ' first event of the run with valid timestamp. If none is'
        ' passed, the start-of-the-run timestamp is provided, hence'
        ' Dragon timestamp is not reliable.'
    )
)

parser.add_argument(
    '--dragon-reference-counter', type=int,
    help=(
        'Dragon counter (pps + 10MHz) in nsecs corresponding'
        'to the first reliable UCTS of the run. To be provided'
        'along with ucts_t0_dragon.'
    ),
)

args = parser.parse_args()


def main():
    log.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    logging.getLogger().addHandler(handler)

    log.info(f'Input file: {args.input_file}')
    log.info(f'Number of events in each subrun: {args.max_events}')
    path_list = sorted(glob.glob(args.input_file))
    log.info(f'list of files: {path_list}')

    config_dic = {}
    # read the configuration file
    if args.config_file is not None:
        config_dic = read_configuration_file(args.config_file)

    config = Config(config_dic)

    source_config = Config({
        "LSTEventSource": {
            "max_events" : args.max_events,
            "EventTimeCalculator": {
                "dragon_reference_time": args.dragon_reference_time,
                "dragon_reference_counter": args.dragon_reference_counter,
            },
            "LSTR0Corrections": {
                "drs4_pedestal_path": args.pedestal_file,
            }
        }
    })

    config.merge(source_config)

    for i, path in enumerate(path_list):
        log.info(f'File {i+1} out of {len(path_list)}')
        log.info(f'Processing: {path}')

        reader = EventSource(input_url=path, config=config)

        if i==0:
            timeCorr = TimeCorrectionCalculate(calib_file_path=args.output_file,
                                               config=config,
                                               subarray=reader.subarray)

        for event in reader:
            if event.index.event_id % 5000 == 0:
                log.info(f'event id = {event.index.event_id}')

            timeCorr.calibrate_peak_time(event)

    # write output
    timeCorr.finalize()


if __name__ == '__main__':
    main()
