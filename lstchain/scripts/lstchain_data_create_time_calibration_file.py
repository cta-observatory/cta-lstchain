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
from ctapipe_io_lst import LSTEventSource
from lstchain.calib.camera.r0 import LSTR0Corrections
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
    if args.config_file is not None:
        try:
            config_dic = read_configuration_file(args.config_file)
        except("Custom configuration could not be loaded !!!"):
            pass
    # read the configuration file
    config = Config(config_dic)

    # declare the pedestal calibrator
    lst_r0 = LSTR0Corrections(pedestal_path=args.pedestal_file, config=config)

    reader = LSTEventSource(input_url=path_list[0], max_events=args.max_events)
    # declare the time corrector
    timeCorr = TimeCorrectionCalculate(calib_file_path=args.output_file,
                                       config=config,
                                       subarray=reader.subarray)

    tel_id = timeCorr.tel_id

    for i, path in enumerate(path_list):
        log.info(f'File {i+1} out of {len(path_list)}')
        log.info(f'Processing: {path}')
        reader = LSTEventSource(input_url=path, max_events=args.max_events)
        for event in reader:
            if event.index.event_id % 5000 == 0:
                log.info(f'event id = {event.index.event_id}')
            lst_r0.calibrate(event)

            # Cut in signal to avoid cosmic events
            if event.r1.tel[tel_id].trigger_type == 4 or (
                    np.median(np.sum(event.r1.tel[tel_id].waveform[0], axis=1))> 300):

                    timeCorr.calibrate_peak_time(event)

    # write output
    timeCorr.finalize()


if __name__ == '__main__':
    main()
