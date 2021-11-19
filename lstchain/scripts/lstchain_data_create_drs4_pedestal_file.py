#!/usr/bin/env python3

"""
Script to create pedestal file for low level calibration.

To set start sample in waveform --start_r0_waveform i (default i = 11)
not to use deltaT correction add --deltaT False

- Input: fits.fz file
- Output: drs4_pedestal.fits file


Usage:

$> python lstchain_data_create_pedestal_file.py
--input-file LST-1.1.Run00097.0000.fits.fz
--output_file drs4_pedestalRun2028.0000.fits
--max_events 9000

"""
from distutils.util import strtobool
import argparse

import numpy as np
from astropy.io import fits
from traitlets.config import Config
from tqdm.auto import tqdm

from ctapipe_io_lst import LSTEventSource
from lstchain.calib.camera.drs4 import DragonPedestal


parser = argparse.ArgumentParser()

# Required arguments
parser.add_argument("--input-file", '-f', type=str, action='store',
                    dest='input_file',
                    help="Path to fitz.fz file to create pedestal file.",
                    default=None, required=True)

parser.add_argument("--output-file", '-o', type=str, action='store',
                    dest='output_file',
                    help="Path where script create pedestal file",
                    default=None, required=True)


# Optional arguments
parser.add_argument("--max-events",
                    help="Maximum numbers of events to read. Default = 20000",
                    type=int,
                    default=20000)

parser.add_argument("--start-sample",
                    help="Start sample for waveform. Default = 11",
                    type=int,
                    default=11)

parser.add_argument('--deltaT', '-s',
                    type=strtobool,
                    help='Boolean. True for use deltaT correction'
                    'Default=True, use False otherwise',
                    default=True)

parser.add_argument('--overwrite', action='store_true', help='Overwrite output file without asking')



def main():
    args = parser.parse_args()
    print("--> Input file: {}".format(args.input_file))
    print("--> Number of events: {}".format(args.max_events))

    source_config = {
        "LSTEventSource": {
            "max_events": args.max_events,
        },
        "LSTR0Corrections": {
            "offset": 0,
            "apply_drs4_pedestal_correction": False,
            "apply_timelapse_correction": args.deltaT,
            "apply_spike_correction": False,
            "select_gain": False,
            "r1_sample_start": 0,
            "r1_sample_end": 40,
        }
    }

    reader = LSTEventSource(input_url=args.input_file, config=Config(source_config))
    print("--> Number of files", reader.multi_file.num_inputs())

    camera_config = reader.camera_config
    n_modules = camera_config.lstcam.num_modules
    pedestal = DragonPedestal(
        tel_id=reader.tel_id,
        n_module=n_modules,
        start_sample=args.start_sample,
    )

    for event in tqdm(reader):
        pedestal.fill_pedestal_event(event)

    # Finalize pedestal and write to fits file
    pedestal.finalize_pedestal()

    expected_pixel_id = fits.PrimaryHDU(camera_config.expected_pixels_id)
    pedestal_array = fits.ImageHDU(
        pedestal.meanped.astype(np.int16),
        name="pedestal array"
    )
    failing_pixels_column = fits.Column(
        name='failing pixels',
        array=pedestal.failing_pixels_array,
        format='K'
    )
    failing_pixels = fits.BinTableHDU.from_columns(
        [failing_pixels_column],
        name="failing pixels"
    )
    hdulist = fits.HDUList([expected_pixel_id, pedestal_array, failing_pixels])
    hdulist.writeto(args.output_file, overwrite=args.overwrite)


if __name__ == '__main__':
    main()
