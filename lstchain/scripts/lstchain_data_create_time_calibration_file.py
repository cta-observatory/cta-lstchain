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
parser.add_argument("--input-file",
                    help="Path to fits.fz file used to create the time calibration file. \
                    Allowed to use regular expression in given path to process subruns",
                    required=True)

parser.add_argument("--output-file",
                    help="Path where script creates the time calibration file",
                    required=True)

# Optional argument
parser.add_argument("--max-events", type=int,
                    help="Maximum numbers of events to read. Default = 20000",
                    default=20000)

parser.add_argument('--config', '-c',
                    help='Path to a configuration file. If none is given, a standard configuration is applied')

parser.add_argument('--pedestal-file', '-p',
                    help='Path to drs4 pedestal file ')

parser.add_argument('--run-summary-path',
                    help='Path to run summary file ')


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
    if args.config is not None:
        config_dic = read_configuration_file(args.config)

    config = Config(config_dic)

    source_config = Config({
        "LSTEventSource": {
            "max_events" : args.max_events,
            "default_trigger_type" : 'tib',
            "EventTimeCalculator": {
                "run_summary_path": args.run_summary_path,
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
