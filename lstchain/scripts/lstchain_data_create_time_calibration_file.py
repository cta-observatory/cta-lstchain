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

from ctapipe.io import EventSource
from tqdm.auto import tqdm
from traitlets.config.loader import Config

from lstchain.calib.camera.time_correction_calculate import TimeCorrectionCalculate
from lstchain.io.config import read_configuration_file

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
                    help='Path to drs4 pedestal file ', required=True)

parser.add_argument('--run-summary-path',
                    help='Path to run summary file ', required=True)

parser.add_argument('--no-progress',
                    action='store_true',
                    help='Do not display a progress bar during event processing')

parser.add_argument(
    '--flatfield-heuristic', action='store_const', const=True, dest="use_flatfield_heuristic",
    help=(
        "If given, try to identify flatfield events from the raw data."
        " Should be used only for data from before 2022"
    )
)
parser.add_argument(
    '--no-flatfield-heuristic', action='store_const', const=False, dest="use_flatfield_heuristic",
    help=(
        "If given, do not to identify flatfield events from the raw data."
        " Should be used only for data from before 2022"
    )
)


def main():
    args = parser.parse_args()
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
            "max_events": args.max_events,
            "pointing_information": False,
            "default_trigger_type": 'tib',
            "use_flatfield_heuristic": args.use_flatfield_heuristic,
            "EventTimeCalculator": {
                "run_summary_path": args.run_summary_path,
            },
            "LSTR0Corrections": {
                "drs4_pedestal_path": args.pedestal_file,
            }
        }
    })

    config.merge(source_config)

    with EventSource(path_list[0]) as s:
        subarray = s.subarray

    timeCorr = TimeCorrectionCalculate(
        calib_file_path=args.output_file,
        config=config,
        subarray=subarray
    )

    for i, path in enumerate(path_list):
        log.info(f'File {i + 1} out of {len(path_list)}')
        log.info(f'Processing: {path}')

        reader = EventSource(input_url=path, config=config)

        for event in tqdm(reader, disable=args.no_progress):
            timeCorr.calibrate_peak_time(event)

    # write output
    timeCorr.finalize()


if __name__ == '__main__':
    main()
