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


import argparse
import numpy as np
from astropy.io import fits
from numba import prange

from ctapipe.io import event_source
from ctapipe.io import EventSeeker
from distutils.util import strtobool
from traitlets.config.loader import Config
from lstchain.calib.camera.r0 import LSTR0Corrections
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


def main():
    print("--> Input file: {}".format(args.input_file))
    print("--> Number of events: {}".format(args.max_events))
    reader = event_source(input_url=args.input_file, max_events=args.max_events)
    print("--> Number of files", reader.multi_file.num_inputs())

    seeker = EventSeeker(reader)
    ev = seeker[0]
    tel_id = ev.r0.tels_with_data[0]
    n_modules = ev.lst.tel[tel_id].svc.num_modules

    config = Config({
        "LSTR0Corrections": {
            "tel_id": tel_id
        }
    })
    lst_r0 = LSTR0Corrections(config=config)

    pedestal = DragonPedestal(tel_id=tel_id, n_module=n_modules, r0_sample_start=args.start_r0_waveform)

    if args.deltaT:
        print("DeltaT correction active")
        for i, event in enumerate(reader):
            for tel_id in event.r0.tels_with_data:
                lst_r0.time_lapse_corr(event, tel_id)
                pedestal.fill_pedestal_event(event)
                if i%500 == 0:
                    print("i = {}, ev id = {}".format(i, event.index.event_id))
    else:
        print("DeltaT correction no active")
        for i, event in enumerate(reader):
            pedestal.fill_pedestal_event(event)
            if i%500 == 0:
                print("i = {}, ev id = {}".format(i, event.index.event_id))

    # Finalize pedestal and write to fits file
    pedestal.finalize_pedestal()

    primaryhdu = fits.PrimaryHDU(ev.lst.tel[tel_id].svc.pixel_ids)
    secondhdu = fits.ImageHDU(np.int16(pedestal.meanped))

    hdulist = fits.HDUList([primaryhdu, secondhdu])
    hdulist.writeto(args.output_file)


if __name__ == '__main__':
    main()

