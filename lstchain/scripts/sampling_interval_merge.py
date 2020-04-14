"""
Authors: Yuto Nogami (Ibaraki Univ.), Hide Katagiri (Ibaraki Univ.)
last update: April 14, 2020
# This script was validated in 14 April.
# This code merges sevelar output files created by sampling_interval_creation_short.py.
# **The order of execution**
# 1. sampling_interval_creation_short.py
#  It creates the DRS4 sampling interval table for each DRS4 cell of all pixels only high gain or low gain. 
#  It needs to be done several times because it can read only ~53000 events in one execution. 
#  The format of the output is as a fits file.
# 2. sampling_interval_merge.py
#  It merges sevelar files of DRS4 sampling interval as a fits file. 
"""
import argparse
import numpy as np
from astropy.io import fits
import os

num_pix = 1855 # number of LST's pixels
num_cell = 1024 # number of DRS4 capcitors (cell).

parser = argparse.ArgumentParser(
    usage='$python sampling_interval_merge.py --input_file RunXXX_interval_HG_0.fits -output_file RunXXX_sampling_intervals_HG.fits',
    description='This script merge several files made by sampling_interval_creation_short.py.\
                 **Caution** If you change the pattern of input file name\
                 from the example in usage, please change line 42 ~ 54 in this code.', 
    add_help=True,
    )

parser.add_argument('-i', '--input_file', help='input the sampling interval file made by sampling_interval_creation_short.py.',
                    action='store', type=str, dest='input_file')
parser.add_argument('-o', '--output_file', help='output sampling intervals file as fits format.',
                    action='store', type=str, dest='output_file')

args = parser.parse_args()

# Searching process about several sampling interval files.
path, name = os.path.split(os.path.abspath(args.input_file))
if '_interval_' in name:
    run, stream = name.split('_interval_', 1)
else:
    run = name

ls = os.listdir(path)
file_list = []
for file_name in ls:
    if run in file_name:
        full_name = os.path.join(path,file_name)
        file_list.append(full_name)

# The process of reading and merging several files
full_events = 0 # number of events to obtain sampling interval.
full_counts = np.zeros((num_pix, num_cell)) # number of pulse-peak counts for each pixcel, each DRS4 cell.
for filename in file_list:
    with fits.open(filename) as f:
        num_events = f[0].header['EVENTS']
        interval = f[0].data
        count = (interval*num_events)/num_cell # convert the interval to counts
        full_events += num_events
        full_counts += count

# convert the counts to the sampling interval
sampling_interval = (full_counts/full_events)*num_cell
print('{} events were used to obtain DRS4 sampling intervals.'.format(full_events))

# output sampling intervals as fits file.
primaryhdu = fits.PrimaryHDU(sampling_interval)
primaryhdu.header['EVENTS'] = (full_events, 'Number of events to obtain sampling intervals.')
hdulist = fits.HDUList([primaryhdu])
hdulist.writeto(args.output_file)
print('The process compleated!!')
