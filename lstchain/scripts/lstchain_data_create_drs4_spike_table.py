#!/usr/bin/env python
# coding: utf-8

# set the filename

import argparse

# Required arguments                                                                                                                                   
parser = argparse.ArgumentParser()
 
parser.add_argument("--input-file", '-f', type=str, action='store',
                    dest='input_file',
                    help="Path to fitz.fz file to create a spike table.",
                    default=None, required=True)

parser.add_argument("--output-file", '-o', type=str, action='store',
                    dest='output_file',
                    help="Path where script create a spike table",
                    default=None, required=False)

parser.add_argument("--pedestal-file", '-p', type=str, action='store',
                    help="Path to pedestal file.",
                    default=None, required=False)

args = parser.parse_args()


inputfile = args.input_file
pedfile = args.pedestal_file
spike_dat = args.output_file

print("")
print("Start!")
print("it takes about 1 minute (exlude a calculation of the pedestal).")

print("input data: {}".format(inputfile))
print("output file: {}".format(spike_dat))

import time
start_time = time.time()

# Make pedestal
if not pedfile: 
    pedfile = "./pedestal.fis"
    print ("you did not set the pedestal file name. {} is created in the current directry.".format(pedfile))
else:
    print("pedestal output: {}".format(pedfile))

import os

if os.path.exists(pedfile) is True:
    print("please check the pedestal filename: {}. it already exists.".format(pedfile))    
else:
    os.system('lstchain_data_create_drs4_pedestal_file --input-file {} --output-file {}'.format(inputfile, pedfile))
    print("finish making a pedestal file")


if not spike_dat:
    spike_dat = "./spike_table.dat"
    print ("you did not set the spike table name. {} is created in the current directry.".format(spike_dat))
else:
    print("spike adc output: {}".format(spike_dat))

# import
import numpy as np
from numba import jit
from numba import prange
# for the LST analysis
from ctapipe_io_lst import LSTEventSource
from traitlets.config.loader import Config
from ctapipe.io import EventSeeker
from ctapipe.image.extractor import LocalPeakWindowSum
from lstchain.calib.camera.r0 import LSTR0Corrections

# Read the data set
reader = LSTEventSource(input_url=inputfile, max_events=None)
print("---> Number of files", reader.multi_file.num_inputs())

config = Config({
    "LSTR0Corrections": 
    {
        "pedestal_path": pedfile,
        "offset":  400,
        "tel_id": 1,
        "r1_sample_start": 3,
        "r1_sample_end": 39,
    }
})

lst_r0 = LSTR0Corrections(config=config)
seeker = EventSeeker(reader)
ev0 = seeker[0]
tel_id = ev0.r0.tels_with_data[0]
nmod = ev0.lst.tel[tel_id].svc.num_modules
mod_id = ev0.lst.tel[tel_id].svc.module_ids
pixel_spiral_id = ev0.lst.tel[tel_id].svc.pixel_ids


def GetFC(ev, tel_id, mod_id, pixel_spiral_id):
    fc_evb = ev.lst.tel[tel_id].evt.first_capacitor_id 
    # reshape the array from (2120) to (265,8)
    fc_mod = np.reshape(fc_evb, (265,8))

    # re-order the array following the pixel_rank_id
    fc_hg = np.reshape(fc_mod[:,(0,0,1,1,2,2,3)] ,(1,1855))
    fc_lg = np.reshape(fc_mod[:,(4,4,5,5,6,6,7)] ,(1,1855))
    fc_pix = np.r_[fc_hg, fc_lg]
    
    # re-order the array following the pixel_spiral_id
    fc_array = np.zeros((2,1855), dtype = int)
    fc_array[:, pixel_spiral_id] = fc_pix 
    return fc_array


def ReOrder(data, mod_id, pixel_spiral_id):
    data_mod_reorder = np.zeros(265)
    data_mod_reorder[mod_id] = data
    data_mod_reshape = np.reshape(data_mod_reorder, (265,1))
    data_copy = data_mod_reshape[:, (0,0,0,0,0,0,0)]
    data_pix = np.reshape(data_copy, (1855))
    data = np.zeros(1855)
    data[pixel_spiral_id] = data_pix
    return data


@jit
def spike_func(cell_id, fc_old):
    
    lc_old=(fc_old+39)%4096
    
    if ((lc_old%2==0) & (lc_old%1024 <= 511)):
        spikeA1_flag = (cell_id%1024 == (lc_old)%1024) 
        spikeB2_flag = (cell_id%1024 == (1022 - lc_old%1024))    
        spikeA2_flag = (cell_id%1024 == (lc_old + 1)%1024)
        spikeB1_flag = (cell_id%1024 == (1023 - lc_old%1024))
    
    if ( spikeA1_flag | spikeB2_flag ):
        spike_flag = 1
    elif ( spikeA2_flag | spikeB1_flag ):
        spike_flag = 2
    else:
        spike_flag = 0
    
    return spike_flag


# initialize
ini_event_id=1000 # event id 1000 or later is fine
fin_event_id=ini_event_id+500

num_of_spike_1st = 0
num_of_spike_2nd = 0
spike_adc_1st = 0.0
spike_adc_2nd = 0.0

mpix=0
npix=1855


for ev in reader:   
    
    event_id = ev.index.event_id    
    if(event_id < ini_event_id): continue

    lst_r0.subtract_pedestal(ev, tel_id)
    lst_r0.time_lapse_corr(ev, tel_id)
    fc_array = GetFC(ev, tel_id, mod_id, pixel_spiral_id)
    
    if(event_id > ini_event_id):        
        for ipix in prange(mpix, npix):
            
            spike_flag = np.zeros(40)
            
            for icell in prange(3, 39):                
                for igain in prange(0, 2):
                
                    spike_flag[icell] = spike_func((fc_array[igain, ipix]+icell)%4096, fc_old_array[igain, ipix])

                    if(spike_flag[icell]==1):
                        spike_adc_1st += ev.r1.tel[tel_id].waveform[igain, ipix, icell]
                        num_of_spike_1st += 1                       
                    elif(spike_flag[icell]==2):
                        spike_adc_2nd += ev.r1.tel[tel_id].waveform[igain, ipix, icell]
                        num_of_spike_2nd += 1
        
    fc_old_array = fc_array
    
    #if(event_id>=fin_event_id): break;
    if(num_of_spike_1st >= 10000 and num_of_spike_2nd >= 10000):
        fin_event_id_true = event_id
        break;
        
        
print ("finish loop: event id: {}".format(fin_event_id_true))

# calculate mean of the spike adc
mean_adc_1st = spike_adc_1st/num_of_spike_1st
mean_adc_2nd = spike_adc_2nd/num_of_spike_2nd
num_of_event = fin_event_id_true - ini_event_id

# write into a dat file
with open(spike_dat, mode='w') as f: 
    f.write("# event_num, num, adc [ADC], flag (1st or 2nd) \n")
    f.write("{} {} {} {}\n".format(num_of_event, num_of_spike_1st, mean_adc_1st-400, 1))
    f.write("{} {} {} {}\n".format(num_of_event, num_of_spike_2nd, mean_adc_2nd-400, 2))

# Check
ev, num, mean, flag = np.loadtxt(spike_dat, comments="#", unpack=True)

if (  (abs(mean[0]-47) > 10) | (abs(mean[1]-53) > 10) ):
    print ('Warning! the spike ADC value is someting wrong!')
    print ('Plase check: 1st and 2nd spike should be ~47 and ~53, respectively.')
    print ('Current: 1st = {}, 2nd = {}'.format(mean[0], mean[1]))
elif ( (num[0] < 10000) | (num[1] < 10000) ):
    print ('Warning! The amount of spike events is small.')
else:
    print ('CalcSpikeRun was Done!')


print ("Elapsed time = {:.2f} s".format(time.time() - start_time))
