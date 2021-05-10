"""
This script creates the sampling intervals for each DRS4 cell of all pixels
only high gain or low gain, and the result outputs as fits file.
Note:
1) The interval table creation is needed ~10^6 events (or more) of test pulse data.
2) This script subtracts pedestal and corrects the spikes & time-lapse before timing-calibration.
3) The sampling interval table should be re-created if the LST camera modules are changed.

This script was validated with ctapipe(v0.8.0), cta-lstchain(v0.6.3)
and ctapipe_io_lst(v0.5.3) in 2020/12/30.

Authors: Yuto Nogami (Ibaraki Univ.), Hideaki Katagri (Ibaraki Univ.)
Last update: January 18, 2021
"""
import os
import time
import h5py
import argparse
import numpy as np
import scipy.optimize as so
from ctapipe.io import event_source
from ctapipe.image import LocalPeakWindowSum
from ctapipe.version import get_version as ctapipe_ver
from lstchain import get_version as lstchain_ver
from lstchain.calib.camera.r0 import LSTR0Corrections
from traitlets.config.loader import Config

parser = argparse.ArgumentParser(
    usage='$> python sampling_interval_creation.py --input_file /.../LST.1.1.Run02064.0000.fits.fz --output_file sampling_interval_20201222.hdf5 --pedestal_file pedestal_Run2065.fits --coefficient_file sampling_interval_20201222.hdf5 --gain_selection 0 --num_events 1000000, --num_charges 1000',
    description='This script creates DRS4 sampling interval for each capacitor of all pixels. We need ~10^6 test pulse events to correct the sampling interval.',
    add_help=True,
    )
# required arguments
parser.add_argument('-i', '--input_file', help='Input event file', action='store',
                    type=str, dest='path_to_data', required=True)
parser.add_argument('-o', '--output_file', help='Output file, file format is .fits',
                    action='store', type=str, dest='output_file', required=True)
parser.add_argument('-ped', '--pedestal_file', help='Pedestal file for the subtraction', 
                    action='store', type=str, dest='pedestal_file', required=True)
parser.add_argument('-gain', '--gain_selection', help='Select gain to create. High-gain is 0, low-gain is 1)', 
                    action='store', type=int, choices=[0,1], dest='gain', required=True)
# optinal arguments
parser.add_argument('-coef', '--coefficient_file', help='please add if the coefficient file was created and error pixels were existed.', action='store',
                    type=str, dest='coefficient_file')
parser.add_argument('-ne', '--num_events', help='Set the number of events you want to use. The default is "None", i.e. all events are used.', 
                    action='store', type=str, default=None, dest='n_events')
parser.add_argument('-nc', '--num_charges', help='Set the number of events for charge caluculation. Default: 1000', 
                    action='store', type=int, default=1000, dest='n_charges')

args = parser.parse_args()

# set the number of events to calculate the sampling interval.
if args.n_events != None:
    args.n_events = int(args.n_events)
    print('{} events will be used to create new sampling interval.'.format(args.n_events))
else:
    print('All events will be used to create new sampling interval.')

# serach the event file for the sampling interval calculation.
# Example filename was as LST-1.1.Run00001.0000.fits.fz.
path, name = os.path.split(os.path.abspath(args.path_to_data))
if 'Run' in name:
    # ex) divide to "LST-1.1." and "0001.0000.fits.fz"
    stream, run = name.split('Run', 1)
    # ex) divide to "0001" and "0000.fits.fz"
    run_num, run_sub = run.split('.',1)
    run_name = 'Run' + run_num
    run_num = stream + run_name


# for log
command = 'python sampling_interval_creation.py'\
        + ' -i ' + args.path_to_data\
        + ' -o ' + args.output_file\
        + ' -ped ' + args.pedestal_file\
        + ' -gain {}'.format(args.gain)\
        + ' -ne ' + str(args.n_events)\
        + ' -nc {}'.format(args.n_charges)

if args.coefficient_file != None:
    command = command + ' -coef ' + args.coefficient_file

method_name = 'counting method'
date = time.strftime('%Y-%m-%d')

ls = os.listdir(path)
file_list = []
for file_name in ls:
    if run_num in file_name:
        full_name = os.path.join(path,file_name)
        file_list.append(full_name)

file_list = sorted(file_list)

# setting for reading and analyzing events
n_gain = 2  # number of gain channels
n_cell = 1024  # number of DRS4 capacitors per 1 domino ring.
n_pixels = 7  # number of pixels per 1 module.
n_module = 265  # number of modules.
roi = 40  # region of interesta
num_pixels = n_pixels*n_module  # number of all pixels for LST.
offset_adc = 400  #  offset adc counts of LST1.
# fill the counts of test pulse peaks for each drs4 cell, each pixel.
peak_counts = np.zeros((num_pixels, n_cell), dtype=np.float32)
sampling_interval = np.zeros((n_gain, num_pixels, n_cell), dtype=np.float32)
# high gain
if args.gain == 0:
    # for drs chip id
    chan = 0
# low gain
elif args.gain == 1:
    # for drs chip id
    chan = 4

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

# read events and counting test pulse peak for sampling interval creation.
print("-----sampling interval calculation is started-----")
# this flag controls the number of events to readout.
flag = False
# control the number of events to create the interval. 
num_used_events = 0
start_time = time.time()
for filename in file_list:
    print(filename)
    with event_source(filename) as source:
        for event in source:
            tel_id = event.lst.tels_with_data[0]
            # convert first capacitor ids from 0-4095 to 0-1023
            fc_all = event.lst.tel[tel_id].evt.first_capacitor_id % 1024
            pid = event.lst.tel[tel_id].svc.pixel_ids
            # DRS4 correction
            lst_r0.calibrate(event)
            # The counts of first 3 cells & last 1 cell become strange.
            event.r1.tel[tel_id].waveform[args.gain, :, :3] = 0
            event.r1.tel[tel_id].waveform[args.gain, :, 39] = 0
            for n_mod in range(n_module):
                for n_pix in range(n_pixels):
                    drs_chip_id = 8*n_mod+n_pix//2 + chan
                    fc = fc_all[drs_chip_id]
                    pix_id = pid[n_mod*7+n_pix]
                    adc = event.r1.tel[tel_id].waveform[args.gain, pix_id]
                    max_cell = adc.argmax() + fc
                    true_max = max_cell
                    # for the cell ids of pulse peak > 1023.
                    if true_max >= n_cell:
                        true_max = max_cell - n_cell
                    peak_counts[pix_id, true_max] += 1

            num_used_events += 1
            # when event_id reach to number of events that user set, the counting process finishes.
            if args.n_events != None:
                if num_used_events == args.n_events:
                    flag = True
                    break

        if flag == True:
            break

# convert from the counts to the sampling intervals
sampling_interval[args.gain] = (peak_counts / num_used_events) * n_cell

end_time_1 = time.time()
print('{} events are used to obtain the sampling interval.'.format(num_used_events))
print('-----sampling interval calculation was finished-----')
time1 = (end_time_1 - start_time) / 3600
print('this process took {} hours to calculate.'.format(time1))

# validate the sampling interval coefficients
print('-----Charge calculation is started-----')
num_loop = 0
# number of events we calculate the charge.
num_events = args.n_charges
print('{} events were used to obtain the charge resolution'.format(num_events))
# histogram setting
num_bins = 1000
pix_arr = np.arange(num_pixels)
x_min = 0
x_max = num_pixels
y_min = 0
y_max = 10000

# charge histogram without the coefficients
hist_without_coef = np.zeros((num_pixels, num_bins), dtype=np.int16)
# with previous coefficients
hist_old_coef = np.zeros((num_pixels, num_bins), dtype=np.int16)
# with new coefficients
hist_new_coef = np.zeros((num_pixels, num_bins), dtype=np.int16)

# read old sampling interval constant
if args.coefficient_file == None:
    old_coef = np.ones((num_pixels, n_cell), dtype=np.float32)
else:
    with h5py.File(args.coefficient_file, 'r') as f:
        old_coef = f['sampling_interval_coefficient'][args.gain]

# charge calculation process
flag = False
for filename in file_list:
    print(filename)
    with event_source(filename) as source:
        integrator = LocalPeakWindowSum(source.subarray)
        for event in source:
            tel_id = event.lst.tels_with_data[0]
            # convert ids 0-4095 to 0-1023
            fc_all = event.lst.tel[tel_id].evt.first_capacitor_id%1024
            pid = event.lst.tel[tel_id].svc.pixel_ids
            # DRS4's properties correction process
            lst_r0.calibrate(event)
            # The counts of first 3 cells & last 1 cell become strange.
            event.r1.tel[tel_id].waveform[args.gain, :, :3] = 0
            event.r1.tel[tel_id].waveform[args.gain, :, 39] = 0
            raw_r1 = event.r1.tel[tel_id].waveform[args.gain]
            # fill the charge to histogram
            raw_charge = integrator(raw_r1, tel_id, args.gain)[0]
            count, xedge, yedge = np.histogram2d(pix_arr, raw_charge, bins=(num_pixels, num_bins), range=((x_min, num_pixels), (y_min, y_max)))
            hist_without_coef += np.int16(count)
            old_coef_r1 = event.r1.tel[tel_id].waveform[args.gain]
            # generate the independent array
            new_coef_r1 = event.r1.tel[tel_id].waveform[args.gain].copy()
            for n_mod in range(n_module): 
                for n_pix in range(n_pixels):
                    drs_chip_id = 8*n_mod+n_pix//2 + chan
                    fc = fc_all[drs_chip_id]
                    pix_id = pid[n_mod*7+n_pix]
                    residual = n_cell - fc
                    # apply DRS4 sampling interval coefficients
                    # treatments for some capacitor ids of ROI over capacitor id=1023.
                    if residual < roi:
                        old_coef_r1[pix_id, :residual] = old_coef_r1[pix_id, :residual]*old_coef[pix_id, fc:]
                        old_coef_r1[pix_id, residual:] = old_coef_r1[pix_id, residual:]*old_coef[pix_id, :roi - residual]

                        new_coef_r1[pix_id, :residual] = new_coef_r1[pix_id, :residual]*sampling_interval[args.gain, pix_id, fc:]
                        new_coef_r1[pix_id, residual:] = new_coef_r1[pix_id, residual:]*sampling_interval[args.gain, pix_id, :roi - residual]
                    else:
                        old_coef_r1[pix_id, :] = old_coef_r1[pix_id, :]*old_coef[pix_id, fc:fc+roi]
                        new_coef_r1[pix_id, :] = new_coef_r1[pix_id, :]*sampling_interval[args.gain, pix_id, fc:fc+roi]
            
            # fill the charge to histogram
            old_coef_charge = integrator(old_coef_r1, tel_id, args.gain)[0]
            count, xedge, yedge = np.histogram2d(pix_arr, old_coef_charge, bins=(num_pixels, num_bins), range=((x_min, num_pixels), (y_min, y_max)))
            hist_old_coef += np.int16(count)

            new_coef_charge = integrator(new_coef_r1, tel_id, args.gain)[0]
            count, xedge, yedge = np.histogram2d(pix_arr, new_coef_charge, bins=(num_pixels, num_bins), range=((x_min, num_pixels), (y_min, y_max)))
            hist_new_coef += np.int16(count)

            num_loop += 1
            # when event_id reach to number of events that user set, the counting process finishes.
            if num_loop == num_events:
                flag = True
                break

        if flag == True:
            break

end_time_2 = time.time()
time2 = (end_time_2 - end_time_1) / 60
print(yedge)
print('{} events were used'.format(num_loop))
print('-----Charge calculation was finished-----')
print('this process took {} minutes'.format(time2))

print('-----sampling interval selection is started-----')

def gaus_fit(x, amp, mu, sigma):
    return amp * np.exp(- (x - mu)**2 / (2 * sigma**2))

# number of pixels considered to have the best coefficients of the three cases.
pixel_ratio = {'without_coef': 0, 'old_coef': 0, 'new_coef': 0}
# charge resolution
reso_without_coef = np.zeros(num_pixels)
reso_old_coef = np.zeros(num_pixels)
reso_new_coef = np.zeros(num_pixels)

y_axis = np.array([sum(yedge[i:i+2])/2 for i in range(num_bins)])
for pix in range(num_pixels):
    # estimate the max amplitude of histogram
    peak_index = np.argmax(hist_new_coef[pix])
    estimate_amp = hist_new_coef[pix, peak_index]
    # convert the amplitude of histogram to charge values array
    charge_arr = []
    for i, n_events in enumerate(hist_new_coef[pix]):
        for j in range(n_events):
            charge_arr.append(y_axis[i])

    estimate_mu = np.mean(charge_arr)
    estimate_sigma = np.std(charge_arr)
    # initial parameters for fitting
    param_init = np.array([estimate_amp, estimate_mu, estimate_sigma])
    # the case of pedestal event.
    if estimate_mu < offset_adc:
        print('pixel {} had pedestal events(lower charge than offset adc(={}))'.format(pix, offset_adc))
        reso_without_coef[pix] = 1
        reso_old_coef[pix] = 1
        reso_new_coef[pix] = 1
        pixel_ratio['old_coef'] += 1
        sampling_interval[args.gain, pix, :] = old_coef[pix]
    # the normal case.
    else:
        try:
            popt, pcov = so.curve_fit(gaus_fit, y_axis, hist_without_coef[pix], p0=param_init)
            reso_without_coef[pix] = popt[2] / popt[1]
            popt, pcov = so.curve_fit(gaus_fit, y_axis, hist_old_coef[pix], p0=param_init)
            reso_old_coef[pix] = popt[2] / popt[1]
            popt, pcov = so.curve_fit(gaus_fit, y_axis, hist_new_coef[pix], p0=param_init)
            reso_new_coef[pix] = popt[2] / popt[1]

            # for comparison of charge resolution 
            tmp_reso = {'without_coef': reso_without_coef[pix]}
            tmp_reso['old_coef'] = reso_old_coef[pix]
            tmp_reso['new_coef'] = reso_new_coef[pix]

        except (so.OptimizeWarning, RuntimeError, ZeroDivisionError) as err:
            print('pixel {}:'.format(pix), err)
            reso_without_coef[pix] = 1
            reso_old_coef[pix] = 1
            reso_new_coef[pix] = 1

            # for comparison of charge resolution 
            tmp_reso = {'without_coef': reso_without_coef[pix]}
            tmp_reso['old_coef'] = reso_old_coef[pix]
            tmp_reso['new_coef'] = reso_new_coef[pix]

        # the coefficient selection using charge resolutions
        min_key = min(tmp_reso, key=tmp_reso.get)
        if min_key == 'new_coef':
            pixel_ratio[min_key] += 1
        elif min_key == 'old_coef':
            sampling_interval[args.gain, pix] = old_coef[pix]
            pixel_ratio[min_key] += 1
        elif min_key == 'without_coef':
            sampling_interval[args.gain, pix, :] = 1
            pixel_ratio[min_key] += 1
            print('pixel id {} was bad!'.format(pix))


end_time_3 = time.time()
time3 = (end_time_3 - end_time_2) / 60
print('this process took {} min.'.format(time3))
print('-----sampling interval selection was finished-----')

# set the key, high gain or low gain
if args.gain == 0:
    gain_key = 'hg'
elif args.gain == 1:
    gain_key = 'lg'
# output as hdf5 format.
with h5py.File(args.output_file, 'a') as f:
    # the case if the constants file was already created.
    if 'sampling_interval_coefficient' in f:
        dataset = f['sampling_interval_coefficient']
        dataset[args.gain] = sampling_interval[args.gain]
        # meta data
        dataset.attrs['creation_date'] = np.append(dataset.attrs['creation_date'], date)
        # add to existing attribute
        if gain_key + '_dataset' in dataset.attrs:
            dataset.attrs[gain_key + '_dataset'] = np.append(dataset.attrs[gain_key + '_dataset'], run_name)
        # create the new attribute for dataset
        else:
            dataset.attrs[gain_key + '_dataset'] = np.array([run_name], dtype=object)

        dataset.attrs['shell'] = np.append(dataset.attrs['shell'], command)
        dataset.attrs['num_events'] = np.append(dataset.attrs['num_events'], num_used_events)
        dataset.attrs[gain_key + '_num_pixels_wo_coef'] = pixel_ratio['without_coef']
        dataset.attrs[gain_key + '_num_pixels_old_coef'] = pixel_ratio['old_coef']
        dataset.attrs[gain_key + '_num_pixels_new_coef'] = pixel_ratio['new_coef']
        dataset.attrs[gain_key + '_charge_resolution_wo_coef'] = reso_without_coef
        dataset.attrs[gain_key + '_charge_resolution_old_coef'] = reso_old_coef
        dataset.attrs[gain_key + '_charge_resolution_new_coef'] = reso_new_coef

    # for the first time
    else:
        dataset = f.create_dataset('sampling_interval_coefficient', data=sampling_interval)
        # meta data
        dataset.attrs['creation_date'] = np.array([date], dtype=object)
        dataset.attrs[gain_key + '_dataset'] = np.array([run_name], dtype=object)
        dataset.attrs['shell'] = np.array([command], dtype=object)
        dataset.attrs['method'] = method_name
        # number of events to create sampling interval coefficients.
        dataset.attrs['num_events'] = np.array([num_used_events], dtype=np.int)
        # number of pixels selected as best coefficients of the three.
        dataset.attrs[gain_key + '_num_pixels_wo_coef'] = pixel_ratio['without_coef']
        dataset.attrs[gain_key + '_num_pixels_old_coef'] = pixel_ratio['old_coef']
        dataset.attrs[gain_key + '_num_pixels_new_coef'] = pixel_ratio['new_coef']
        dataset.attrs[gain_key + '_charge_resolution_wo_coef'] = reso_without_coef
        dataset.attrs[gain_key + '_charge_resolution_old_coef'] = reso_old_coef
        dataset.attrs[gain_key + '_charge_resolution_new_coef'] = reso_new_coef
        dataset.attrs['ctapipe_version'] = ctapipe_ver()
        dataset.attrs['lstchain_version'] = lstchain_ver()

total_time = (end_time_3 - start_time) / 3600
print('Total time is {} hours'.format(total_time))
print('Finished!!')

