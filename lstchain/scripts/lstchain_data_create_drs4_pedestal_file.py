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

from traitlets.config import Config
import argparse
import numpy as np
from astropy.io import fits

from ctapipe.io import EventSource

from distutils.util import strtobool
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

parser.add_argument("--start-r0-waveform",
                    help="Start sample for waveform. Default = 11",
                    type=int,
                    default=11)

parser.add_argument('--deltaT', '-s',
                    type=lambda x: bool(strtobool(x)),
                    help='Boolean. True for use deltaT correction'
                    'Default=True, use False otherwise',
                    default=True)

args = parser.parse_args()

source_config = {
    "LSTEventSource": {
        "max_events":args.max_events,
    }
}
def main():
    print("--> Input file: {}".format(args.input_file))
    print("--> Number of events: {}".format(args.max_events))

    reader = EventSource(input_url=args.input_file, config=Config(source_config))
    print("--> Number of files", reader.multi_file.num_inputs())

    for i, event in enumerate(reader):
        for tel_id in event.trigger.tels_with_trigger:

            if i==0:
                n_modules = event.lst.tel[tel_id].svc.num_modules
                pedestal = DragonPedestal(tel_id=tel_id, n_module=n_modules, r0_sample_start=args.start_r0_waveform)

            if args.deltaT:
                reader.r0_r1_calibrator.update_first_capacitors(event)
                reader.r0_r1_calibrator.time_lapse_corr(event, tel_id)

            pedestal.fill_pedestal_event(event)
            if i%500 == 0:
                print("i = {}, ev id = {}".format(i, event.index.event_id))

    # Finalize pedestal and write to fits file
    pedestal.finalize_pedestal()

    expected_pixel_id = fits.PrimaryHDU(event.lst.tel[tel_id].svc.pixel_ids)
    pedestal_array = fits.ImageHDU(np.int16(pedestal.meanped),
                                   name="pedestal array")
    failing_pixels_column = fits.Column(name='failing pixels',
                                        array=pedestal.failing_pixels_array,
                                        format='K')
    failing_pixels = fits.BinTableHDU.from_columns([failing_pixels_column],
                                                    name="failing pixels")
    hdulist = fits.HDUList([expected_pixel_id, pedestal_array, failing_pixels])
    hdulist.writeto(args.output_file)


if __name__ == '__main__':
    main()
