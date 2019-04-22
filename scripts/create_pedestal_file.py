import sys
sys.path.insert(0,'/home/pawel1/Pulpit/Astrophysics/CTA/soft/ctapipe_io_lst')
from ctapipe_io_lst import LSTEventSource

import argparse
import numpy as np
from astropy.io import fits
from ctapipe.io import EventSeeker
from traitlets.config.loader import Config

from lstchain.calib.camera.r0 import LSTR0Corrections
from lstchain.calib.camera.drs4 import DragonPedestal


''' 
Script to create pedestal file for low level calibration. 
To run script in console:
python create_pedestal_file.py --input_file LST-1.\*.Run00097.0000.fits.fz --output_file pedestal.fits --max_events 9000
'''

parser = argparse.ArgumentParser()


parser.add_argument("--input_file", help="Path to fitz.fz file to create pedestal file.",
                    type=str, default="")

parser.add_argument("--output_file", help="Path where script create pedestal file",
                    type=str)

parser.add_argument("--max_events", help="Maximum numbers of events to read",
                    type=int, default=5000)

args = parser.parse_args()

if __name__ == '__main__':
    print("input file: {}".format(args.input_file))
    print("max events: {}".format(args.max_events))
    reader = LSTEventSource(input_url=args.input_file, max_events=args.max_events)
    print("---> Number of files", reader.multi_file.num_inputs())

    seeker = EventSeeker(reader)
    ev = seeker[0]
    n_modules = ev.lst.tel[0].svc.num_modules
    telid = ev.r0.tels_with_data[0]

    config = Config({
        "LSTR0Corrections": {
            "tel_id": telid
        }
    })
    lst_r0 = LSTR0Corrections(config=config)

    ped = DragonPedestal(telid=telid)
    PedList = []
    pedestal_value_array = np.zeros((n_modules, 2, 7, 4096), dtype=np.uint16)

    for i in range(0, n_modules):
        PedList.append(DragonPedestal())

    for i, event in enumerate(reader):
        print("i = {}, ev id = {}".format(i, event.r0.event_id))
        lst_r0.time_lapse_corr(ev)
        for nr_module in range(0, n_modules):
            PedList[nr_module].fill_pedestal_event(event, nr=nr_module)

    # Finalize pedestal and write to fits file
    for i in range(0, n_modules):
        PedList[i].finalize_pedestal()
        PedList[i].meanped[np.isnan(PedList[i].meanped)] = 300  # fill 300 where occurs NaN
        pedestal_value_array[i, :, :, :] = PedList[i].meanped

    # re-order offset values according to expected pixel id
    expected_pixel_id = ev.lst.tel[telid].svc.pixel_ids
    ped_array = np.zeros((2, 1855, 4096), dtype=np.uint16)
    for nr in range(0, n_modules):
        for gain in range(0, 2):
            for pix in range(0, 7):
                pixel = expected_pixel_id[nr * 7 + pix]
                ped_array[:, pixel, :] = pedestal_value_array[nr, :, pix, :]

    primaryhdu = fits.PrimaryHDU(ev.lst.tel[telid].svc.pixel_ids)
    secondhdu = fits.ImageHDU(ped_array)

    hdulist = fits.HDUList([primaryhdu, secondhdu])
    hdulist.writeto(args.output_file)