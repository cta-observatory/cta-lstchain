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

<<<<<<< HEAD
from ctapipe_io_lst import LSTEventSource
from ctapipe.io import EventSeeker
=======

from ctapipe.io import EventSource
from ctapipe_io_lst.calibration import LSTR0Corrections
>>>>>>> 365258afc85e779a77e9d993bda64a9327afd0f7
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
        "fill_timestamp": False,
    }
}
def main():
    print("--> Input file: {}".format(args.input_file))
    print("--> Number of events: {}".format(args.max_events))
<<<<<<< HEAD
    reader = LSTEventSource(input_url=args.input_file, max_events=args.max_events, fill_timestamp=False)
    print("--> Number of files", reader.multi_file.num_inputs())

    tel_id = 1
    seeker = EventSeeker(reader)
    ev = seeker.get_event_index(1)
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
            lst_r0.time_lapse_corr(event, tel_id)
            pedestal.fill_pedestal_event(event)
            if i%500 == 0:
                print("i = {}, ev id = {}".format(i, event.index.event_id))
=======
    reader = EventSource(input_url=args.input_file, config=Config(source_config))
    print("--> Number of files", reader.multi_file.num_inputs())

    if args.deltaT:
        print("DeltaT correction active")
>>>>>>> 365258afc85e779a77e9d993bda64a9327afd0f7
    else:
        print("DeltaT correction no active")

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

<<<<<<< HEAD
    expected_pixel_id = fits.PrimaryHDU(ev.lst.tel[tel_id].svc.pixel_ids)
    pedestal_mean_array = fits.ImageHDU(np.int16(pedestal.meanped))
    failing_pixels = fits.ImageHDU(pedestal.failing_pixels_list)
    hdulist = fits.HDUList([expected_pixel_id, pedestal_mean_array, failing_pixels])
=======
    primaryhdu = fits.PrimaryHDU(event.lst.tel[tel_id].svc.pixel_ids)
    secondhdu = fits.ImageHDU(np.int16(pedestal.meanped))

    hdulist = fits.HDUList([primaryhdu, secondhdu])
>>>>>>> 365258afc85e779a77e9d993bda64a9327afd0f7
    hdulist.writeto(args.output_file)


if __name__ == '__main__':
    main()
