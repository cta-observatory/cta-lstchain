"""
Authors: Yuto Nogami (Ibaraki Univ.), Hide Katagri (Ibaraki Univ.)
Date: Dec 20, 2019
#  This script was validated with ctapipe(0.7.0.post128), cta-lstchain(Dec. 18) 
# and ctapipe_io_lst(Dec. 17) in Dec 20.
#  This script counts the test pulse peak for each capacitor of all pixels only HG or LG.
#  This script subtracts pedestal and corrects the spikes & time-lapse before timing-calibration.
#  This script is made for the multiple job such as batch job.
# Important 1.: The interval table creation is needed 10^6 events (or more) of test pulse data.
# Important 2.: The output file is not complete yet, so please use ConvertCountsToInterval.py!
# Important 3.: please set the output file name with one pattarn such as below.
#  ex) RunXXX_PeakCount_HG_0.fits, RunXXX_PeakCount_HG_1.fits, ....
# Important 4.: The sampling interval table should be re-created if the LST camera modules are changed.
"""
import argparse
import numpy as np
from astropy.io import fits
from ctapipe.io import event_source
from lstchain.calib.camera.r0 import LSTR0Corrections
from traitlets.config.loader import Config

parser = argparse.ArgumentParser(
    usage='$python create_sampling_interval.py --input_file /.../LST.1.1.RunXXX.xxx.fits.fz --output_file RunXXX_PeakCount_HG_xxx.fits --pedestal_file pedestal_RunYYY.fits --gain_selection 0 --num_events 53000',
    description='This script counts test pulse peak for each capacitor of all pixels. \
                 We need ~10^6 test pulse events to correct the sampling interval. \
                 **Recommendation**: multiple job submission like a batch jobs.',
    add_help=True,
    )

parser.add_argument('-i', '--input_file', help='Input event file', action='store',
                    type=str, dest='path_to_data')
parser.add_argument('-o', '--output_file', help='Output file, file format is .fits',
                    action='store', type=str, dest='output_file')
parser.add_argument('-ped', '--pedestal_file', help='Pedestal file for the subtraction', 
                    action='store', type=str, dest='pedestal_file')
parser.add_argument('-g', '--gain_selection', help='Select gain to use (high-gain is 0, low-gain is 1)', 
                    action='store', type=int, choices=[0,1], dest='gain')
parser.add_argument('-n', '--num_events', help='Set the number of events you want to use. The default is "None", i.e. all events are used.', 
                    action='store', type=str, default=None, dest='n_events')
args = parser.parse_args()

# conversion to integer if the default didn't use.
if args.n_events != None:
    args.n_events = int(args.n_events)
else:
    pass

# setting for reading and analyzing events
n_gain = 2 # number of gain channels
n_cell = 1024 # number of DRS4 capacitors per 1 domino ring.
n_pixels = 7 # number of pixels per 1 module.
n_module = 265 # number of modules.
num_pixels = n_pixels*n_module #number of all pixels for LST.
hg_id = 0 # High gain index in ctapipe container.
lg_id = 1 # Low gain index in ctapipe container.
fc=np.zeros((n_gain, num_pixels), dtype =np.int16)
fc_reorder = np.zeros((n_gain, num_pixels),dtype=np.int16)
peak_counts = np.zeros((num_pixels, n_cell))#fill the counts for all pixels.

reader = event_source(input_url=args.path_to_data, max_events=args.n_events)

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

for event in reader:
    telID = event.lst.tels_with_data[0]
    fc_all = event.lst.tel[telID].evt.first_capacitor_id%1024 # convert ids 0-4095 to 0-1023
    pid = event.lst.tel[telID].svc.pixel_ids
    lst_r0.calibrate(event)# DRS4's properties correction process
    #lst_r0.subtract_pedestal(event, telID)# only pedestal subtract process
    for n_mod in range(n_module):
        for n_pix in range(n_pixels):
            fc[hg_id][n_mod*7+n_pix] = fc_all[8*n_mod+n_pix//2]
            fc[lg_id][n_mod*7+n_pix] = fc_all[8*n_mod+n_pix//2 + 4]
            
    for n_pix in range(num_pixels):
        fc_reorder[hg_id][pid[n_pix]] = fc[hg_id][n_pix]
        fc_reorder[lg_id][pid[n_pix]] = fc[lg_id][n_pix]

    # counting process
    for n_pix in range(num_pixels):
        event.r1.tel[telID].waveform[args.gain, n_pix,:2] = 0# The counts of first & last 2cells become strange..
        event.r1.tel[telID].waveform[args.gain, n_pix,38:] = 0
        ADC = event.r1.tel[telID].waveform[args.gain, n_pix]
        max_cell = ADC.argmax() + fc_reorder[args.gain][n_pix]
        if max_cell < n_cell:
            peak_counts[n_pix][max_cell] += 1
        if max_cell >= n_cell:
            true_max = max_cell - n_cell
            peak_counts[n_pix][true_max] += 1

# output as fits format.
primaryhdu = fits.PrimaryHDU(np.int16(peak_counts))
hdulist = fits.HDUList([primaryhdu])
hdulist.writeto(args.output_file)
print('The process compleated!!')
