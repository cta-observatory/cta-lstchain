"""
Authors: Yuto Nogami (Ibaraki Univ.), Hide Katagiri (Ibaraki Univ.)
Date: Dec. 23, 2019
#  This example script was validated with ctapipe(0.7.0.post128), cta-lstchain(Dec. 18)
# and ctapipe_io_lst(Dec. 17) in Dec 23.
#  This script can apply sampling coeffcients (timing-calibration) and it creates 
# charge histgram before/after timing-calibration only HG or LG in a root file using ROOT. 
#  This script applied pedestal subtraction, spikes correction and time-lapse correction
# before timing-calibration and obtained the charge using LocalPeakWindowSum class.
#  The algolithm of applying sampling coefficients will be merged to LSTR0Corrections class.
# Important : please set the output file name with one pattern such as below.
# ex) RunXXXHistNoCalib_0000.root, RunXXXHistNoCalib_0001.root, ...

# The order of execution is as follows:
# 1. sampling_interval_count_pulse_per_cell.py
#  This script creates the counts of pulse to obtain DRS4 sampling intervals for
# each DRS4 cell of all pixels, only high gain or low gain. The counts table outputs
# as a fits file.
# 2. sampling_interval_counts_to_interval.py
#  This script merges sevelar counts-files and converts the counts to sampling intervals
# as a fits file.
# 3. sampling_interval_charge_dist.py
#  This script obtains the charge and outputs charge distriution as 2 root-files using
# the ROOT.
# 4. sampling_interval_merge_charge_dist.py
#  This script merges the root files to 2 root-files using the ROOT.
# 5. sampling_interval_check_calib.py
#  This script checks the charge distribution and resolution using the ROOT.
"""
import argparse
from astropy.io import fits
import numpy as np
from ctapipe.io import event_source
from lstchain.calib.camera.r0 import LSTR0Corrections
from ctapipe.image import LocalPeakWindowSum
from traitlets.config.loader import Config
from ROOT import PyConfig
PyConfig.IgnoreCommandLineOptions=True # ignore cmd options for ROOT if it sets "True".
from ROOT import gROOT, gStyle
gROOT.SetBatch() # this is not to show the graph in GUI.
from ROOT import TCanvas, TH1D, TF1, TFile

parser = argparse.ArgumentParser(
    usage='$python charge_calculator.py --input_file /.../LST1.1.Run00XXX.00xx.fits.fz --hist_file1 RunXXXHistNoCalib_xxx.root -hist_file2 RunXXXHistCalib_xxx.root -s RunXXX_sampling_intervals_HG.fits.fits -ped RunXXX_pedestal.fits -g 0 -n 10000',
    description='This is example code to calculate charge with LocalPeakWindowSum class.',
    add_help=True,
    )
parser.add_argument('-i', '--input_file', help='input event file', action='store',
                    type=str, dest='path_to_data')
parser.add_argument('-hist1', '--hist_file1', help='output file name for charge histogram before timing calibration, recommended file format is .root', action='store',
                    type=str, dest='hist_file1')
parser.add_argument('-hist2', '--hist_file2', help='output file name for charge histogram after timing calibration, recommended file format is .root', action='store',
                    type=str, dest='hist_file2')
parser.add_argument('-s', '--sampling_interval', help='sampling interval file name for \
                    timing calibration. recommend file format is .fits (2019/12/19-)',
                    action='store', type=str, dest='interval_file')
parser.add_argument('-ped', '--pedestal_file', help='pedestal file name for the pedestal subtraction.',
                    action='store', type=str, dest='pedestal_file')
parser.add_argument('-g', '--gain_selection', help='gain selection (High-gain is 0, Low-gain is 1)',
                    action='store', type=int, choices=[0,1], dest='gain')
parser.add_argument('-n', '--num_events', help='Set the number of events to use. The default is None, it means you use all events',
                    action='store', default=None, type=str, dest='n_events')
args = parser.parse_args()

# pyROOT setting
num_fit_param =10 # each mu, sigma, chi-square, ndf and prob. with/without timing-calibration.
num_pixels = 1855
num_bins = 1200 # histogram's bin.
x_min = 0 # minimum about x-axis.
x_max = 12000 # maximum about x-axis.
hist_NoCalib = [] # Charge(ADC counts) histgram before timing calibration.
hist_Calib = [] # Charge(ADC counts) histgram after timing calibration.
for pix in range(num_pixels): # this loop create 1855 TCanvases. (every hist_array)
    hist_NoCalib.append(TH1D('pixel {} (NoCalib.)'.format(pix),
                        'Charge histogram (no calib.);Charge;Number of Events',
                        num_bins, x_min, x_max))
    hist_Calib.append(TH1D('pixel {} (Calib.)'.format(pix),
                      'Charge histogram (calib.);Charge;Number of Events',
                      num_bins, x_min, x_max))

# setting of event-reader with ctapipe (catpipe_io_lst).
if args.n_events != None:
    args.n_events = int(args.n_events)
if args.n_events == None:
    pass

reader = event_source(input_url=args.path_to_data, max_events=args.n_events)# you can set max_events of reading range
n_cell = 1024 # number of capacitors in a domino ring of DRS4.
n_gain = 2 # number of gains
n_pixel = 7 # number of pixels in a module.
roi = 40 # region of interest.
hg_id = 0 # High-gain's index id.
lg_id = 1 # Low-gain's index id.
fc = np.zeros((n_gain, num_pixels), dtype=np.int16)
fc_reorder = np.zeros((n_gain, num_pixels),dtype=np.int16)

# LSTR0Correction setting
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
integrator = LocalPeakWindowSum()
# reading sampling coefficients from a file.
with fits.open(args.interval_file) as f:
    tcal = f[0].data

# reading and analyzing events.
for event in reader:
    telID = event.lst.tels_with_data[0]
    fc_all = event.lst.tel[telID].evt.first_capacitor_id
    pid = event.lst.tel[telID].svc.pixel_ids
    n_module = event.lst.tel[telID].svc.num_modules
    # reshape and re-order the first capacitor id array to (2gains, 1855pixels) array.
    for nmod in range(n_module): # reshape the (2gains, 265modules*8drs4) array to (2gains, 1855pixels)
        for npix in range(n_pixel):
            fc[hg_id][nmod*7+npix] = fc_all[8*nmod+npix//2]
            fc[lg_id][nmod*7+npix] = fc_all[8*nmod+npix//2 + 4]

    for pix in range(num_pixels): # re-order the pid's order to the order such as 0, 1,..., 1853, 1854.
        fc_reorder[hg_id][pid[pix]] = fc[hg_id][pix]
        fc_reorder[lg_id][pid[pix]] = fc[lg_id][pix]

    # applying some calibrations without the sampling coeffcients.
    lst_r0.calibrate(event) # calibrate some DRS4 properties.
    ADC_all = event.r1.tel[telID].waveform
    ADC_all[:, :, :2] = 0 # ADC of first & last 2cells become strange.
    ADC_all[:, :, 38:] = 0 # So ADC of each 2cells are replaced to 0.
    charge_nc = integrator(ADC_all)[0][args.gain] # charge_nc is before applying sampling coeffcients only HG or LG.
    fc = fc_reorder%1024 # conversion the capacitor ids from 0-4095 to 0-1023.
    fc_to_last = 1024 - fc[:, :]

    # applying sampling coeffcients
    for pix in range(num_pixels):
        fc_i = fc[args.gain, pix] # first capacitor id of pixel id "pix".
        if 0 < fc_to_last[args.gain, pix] < roi: # treatments for some capacitor ids of ROI over capacitor id=1023.
            fl = fc_to_last[args.gain, pix] # buffer length(=1024) - fc of pixel id "pix".
            ADC_all[args.gain, pix, :fl] = ADC_all[args.gain, pix, :fl]*tcal[pix, fc_i:]
            ADC_all[args.gain, pix, fl:] = ADC_all[args.gain, pix, fl:]*tcal[pix, :roi - fl]

        else:
            ADC_all[args.gain, pix, :] = ADC_all[args.gain, pix, :]*tcal[pix, fc_i:fc_i + roi]

    charge_c = integrator(ADC_all)[0][args.gain] # charge_c is before applying the timing calibration only HG or LG.

    # filling the charge values to histograms.
    for pix in range(num_pixels):
        hist_NoCalib[pix].Fill(charge_nc[pix])
        hist_Calib[pix].Fill(charge_c[pix])

outfile = TFile(args.hist_file1, "recreate") # before timing-calibration
outfile2 = TFile(args.hist_file2, "recreate") # after timing-calibration
# add histograms to each root file.
for pix in range(num_pixels):
        # histogram before timing calibration
        outfile.Add(hist_NoCalib[pix])
        # histogram after timing calibration
        outfile2.Add(hist_Calib[pix])

outfile.Write()
outfile2.Write()
outfile.Close()
outfile2.Close()

print('the process is sucess!')
