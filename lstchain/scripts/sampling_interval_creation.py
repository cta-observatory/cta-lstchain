"""
Authors: Yuto Nogami (Ibaraki Univ.), Hide Katagri (Ibaraki Univ.)
last update: April 14, 2020
#  This script was validated with ctapipe(0.7.0.post171), cta-lstchain(March 2) 
# and ctapipe_io_lst(Dec. 17) in April 14.
#  This script creates the sampling intervals for each DRS4 cell of all pixels
# only high gain or low gain, and the result outputs as fits file.
# Important 1.: The interval table creation is needed 10^6 events (or more) of test pulse data.
# Important 2.: This script subtracts pedestal and corrects the spikes & time-lapse before timing-calibration.
# Important 3.: The sampling interval table should be re-created if the LST camera modules are changed.

"""
import os
import argparse
import numpy as np
from astropy.io import fits
from ctapipe.io import event_source
from lstchain.calib.camera.r0 import LSTR0Corrections
from traitlets.config.loader import Config

parser = argparse.ArgumentParser(
    usage='$python sampling_interval_creation.py --input_file /.../LST.1.1.RunXXX.xxx.fits.fz --output_file sampling_interval_RunXXX_HG.fits --pedestal_file pedestal_RunYYY.fits --gain_selection 0 --num_events 1000000',
    description='This script creates DRS4 sampling interval for each capacitor of all pixels. We need ~10^6 test pulse events to correct the sampling interval.',
    add_help=True,
    )

parser.add_argument('-i', '--input_file', help='Input event file', action='store',
                    type=str, dest='path_to_data')
parser.add_argument('-o', '--output_file', help='Output file, file format is .fits',
                    action='store', type=str, dest='output_file')
parser.add_argument('-ped', '--pedestal_file', help='Pedestal file for the subtraction', 
                    action='store', type=str, dest='pedestal_file')
parser.add_argument('-g', '--gain_selection', help='Select gain to create. High-gain is 0, low-gain is 1)', 
                    action='store', type=int, choices=[0,1], dest='gain')
parser.add_argument('-n', '--num_events', help='Set the number of events you want to use. The default is "None", i.e. all events are used.', 
                    action='store', type=str, default=None, dest='n_events')
args = parser.parse_args()

# set the number of events to calculate the sampling interval.
if args.n_events != None:
    args.n_events = int(args.n_events)
else:
    pass

# serach the event file for the sampling interval calculation.
# Example filename was as LST-1.1.Run00001.0000.fits.fz.
path, name = os.path.split(os.path.abspath(args.path_to_data))
if 'Run' in name:
    stream, run = name.split('Run', 1) # ex) divide to "LST-1.1." and "0001.0000.fits.fz"
    run_num, run_sub = run.split('.',1) # ex) divide to "0001" and "0000.fits.fz"
    run_num = stream + 'Run' + run_num
else:
    print("This process maybe fail due to inconsistent file name like a LST-1.1.Run01732.0000.fits.fz")

ls = os.listdir(path)
file_list = []
for file_name in ls:
    if run_num in file_name:
        full_name = os.path.join(path,file_name)
        file_list.append(full_name)

# setting for reading and analyzing events
n_gain = 2 # number of gain channels
n_cell = 1024 # number of DRS4 capacitors per 1 domino ring.
n_pixels = 7 # number of pixels per 1 module.
n_module = 265 # number of modules.
num_pixels = n_pixels*n_module # number of all pixels for LST.
peak_counts = np.zeros((num_pixels, n_cell)) # fill the counts for all pixels.
sampling_interval = np.zeros((num_pixels, n_cell))

config = Config({
    "LSTR0Corrections": {
        "pedestal_path": args.pedestal_file,
        "offset": 0,
        "tel_id": 1,
        "r1_sample_start": None,
        "r1_sample_end": None
    }
})
lst_r0 = LSTR0Corrections(config=config)

if args.gain == 0:
    chan = 0
else:
    chan = 4

flag = False # this flag controls the number of events to readout.
# read events and counting test pulse peak for sampling interval creation.
num_loop = 0 # control the number of events to create the interval.
for filename in file_list: # read all events.
    with event_source(filename) as source:
        for event in source:
            telID = event.lst.tels_with_data[0]
            fc_all = event.lst.tel[telID].evt.first_capacitor_id%1024 # convert ids 0-4095 to 0-1023
            pid = event.lst.tel[telID].svc.pixel_ids
            lst_r0.calibrate(event)# DRS4's properties correction process
            # reorder the DRS4 first capacitor ids array to (2 gains, 1855 pixels)
            for n_mod in range(n_module):
                for n_pix in range(n_pixels):
                    fc = fc_all[8*n_mod+n_pix//2 + chan]
                    event.r1.tel[telID].waveform[args.gain, pid[n_mod*7+n_pix],:2] = 0# The counts of first & last 2cells become strange..
                    event.r1.tel[telID].waveform[args.gain, pid[n_mod*7+n_pix],38:] = 0
                    ADC = event.r1.tel[telID].waveform[args.gain, pid[n_mod*7+n_pix]]
                    max_cell = ADC.argmax() + fc
                    true_max = max_cell
                    if true_max >= n_cell:
                        true_max = max_cell - n_cell
                    peak_counts[pid[n_mod*7+n_pix]][true_max] += 1

            num_loop += 1

            if args.n_events != None:
                if num_loop == args.n_events:
                    flag = True
                    break

        if flag == True:
            break # when event_id reach to number of events that user set, the counting process finishes.

print('The counting process finished!')
print('{} events are used to obtain the sampling interval.'.format(num_loop))
# calculate the sampling intervals
for cellID in range(n_cell):
    sampling_interval[:,cellID] = (peak_counts[:,cellID]/num_loop)*n_cell

# output as fits format.
primaryhdu = fits.PrimaryHDU(sampling_interval)
primaryhdu.header['EVENTS'] = (num_loop, 'Number of events to obtain sampling intervals.')
hdulist = fits.HDUList([primaryhdu])
hdulist.writeto(args.output_file)
print('The process compleated!!')
