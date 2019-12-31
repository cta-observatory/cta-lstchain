"""
Authors: Yuto Nogami (Ibaraki Univ.), Hide Katagiri (Ibaraki Univ.)
Date: Dec. 20, 2019
# This script was validated in Dec 20.
# This code merges sevelar output files created by create_sampling_interval.py and
#converts the counts to sampling intervals for all pixels, only high-gain or low-gain.

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
import numpy as np
from astropy.io import fits
import os

num_pix = 1855
num_capa_mod = 1024
sampling_interval = np.zeros((num_pix,num_capa_mod))

parser = argparse.ArgumentParser(
    usage='$python ConvertCountsToInterval.py --input_file /.../RunXXX_PeakCount_HG_xxx.fits -output_file RunXXX_sampling_intervals_HG.fits',
    description='This script merge several files made by create_sampling_intervals.py\ 
                 and converts the counts of test pulse peak for each capacitor of all pixels\
                 to sampling intervals. **Caution** If you change the pattern of input file name\
                 from the example in usage, please change line 31 & 32 of this code.', 
    add_help=True,
    )

parser.add_argument('-i', '--input_file', help='input the counts file made by create_sampling_interval.py',
                    action='store', type=str, dest='input_file')
parser.add_argument('-o', '--output_file', help='output sampling intervals file as fits format.',
                    action='store', type=str, dest='output_file')

args = parser.parse_args()

# Searching process about several pulse-peak-counts files.
path, name = os.path.split(os.path.abspath(args.input_file))
if '_PeakCount_' in name:
    run, stream = name.split('_PeakCount_', 1)
else:
    run = name

ls = os.listdir(path)
file_list = []
for file_name in ls:
    if run in file_name:
        full_name = os.path.join(path,file_name)
        file_list.append(full_name)

# To read several files and merge the pulse-peak-counts for each pixel.
total = np.zeros((num_pix,num_capa_mod), dtype=np.int16)
for peak_counts_file in file_list:
#    with open(peak_counts_file, 'rb') as f:
#        count = pickle.load(f)
    with fits.open(peak_counts_file) as f:
        count = np.int16(f[0].data)
        total += count

# Convert peak_count to calib_time
num_events = sum(total[0])
for cellID in range(1024):
    sampling_interval[:,cellID] = (total[:,cellID]/num_events)*num_capa_mod

print('{} events were used to create DRS4 sampling intervals.'.format(num_events))
# output sampling intervals as fits file.
primaryhdu = fits.PrimaryHDU(sampling_interval)
hdulist = fits.HDUList([primaryhdu])
hdulist.writeto(args.output_file)
print('The process compleated!!')
