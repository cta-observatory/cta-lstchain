"""
Author: Yuto Nogami(Ibaraki Univ.), Hide Katagiri(Ibaraki Univ.)
Date: Dec. 23, 2019
# This example script was validated in Dec 23.
# This script can merge several root files made by charge_calculation.py.
# This script uses ROOT.

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
import subprocess
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from ROOT import PyConfig
PyConfig.IgnoreCommandLineOptions=True # ignore cmd options for ROOT if it sets "True".
from ROOT import gROOT, gStyle
gROOT.SetBatch() # this is not to show the graph in GUI.
from ROOT import TCanvas, TH1D, TF1, TMath, TFile

parser = argparse.ArgumentParser(
    usage='$python merge_hist.py -root_file1 RunXXXHistNoCalib_xxx.root -root_file2 RunXXXHistCalib_xxx.root -output_file1 RunXXXHistNoCalibAllEvents.root -outout_file2 RunXXXHistCalibAllEvents.root',
    description='This script merges histograms of each root-file and output merged new root-file.\
                 **Caution** If you change the pattern of input file name from the example\
                 in usage, please change line 51-58 of this code.',
    add_help=True,
    )
parser.add_argument('-r1', '--root_file1', help='input root-file name for charge histogram before\
                    timing-calibration', action='store',type=str, dest='root_file1')
parser.add_argument('-r2', '--root_file2', help='input root-file name for charge histogram after\
                    timing-calibration', action='store', type=str, dest='root_file2')
parser.add_argument('-o1', '--output_file1', help='output file name for merged root-file before\
                    timing-calibration, file format is root', action='store',
                    type=str, dest='outfile1')
parser.add_argument('-o2', '--output_file2', help='output file name for merged root-file after\
                    timing-calibration, file format is root', action='store',
                    type=str, dest='outfile2')

args = parser.parse_args()

# pyROOT settings
num_pixels = 1855
num_page = num_pixels//4 + 1 # number of pages to output file about histograms.
canvas_NoCalib = [] # TCanvas array for hist_NoCalib.
canvas_Calib = [] # TCanvas array for hist_Calib.
for page_i in range(num_page): # create number of "num_page" TCanvases.
    canvas_NoCalib.append(TCanvas())
    canvas_Calib.append(TCanvas())

# merge the several root file.
command1 = ["hadd", args.outfile1] # the command to merge histograms in root files (NoCal).
command2 = ["hadd", args.outfile2] # the command to merge histograms in root files (Cal).
path1, name1 = os.path.split(os.path.abspath(args.root_file1))
path2, name2 = os.path.split(os.path.abspath(args.root_file2))
if 'HistNoCalib' in name1:
    run1, stream1 = name1.split('_', 1)
else:
    run1 = name1
if 'HistCalib' in name2:
    run2, stream2 = name2.split('_', 1)
else:
    run2 = name2

ls1 = os.listdir(path1)
ls2 = os.listdir(path2)
for file_name1 in ls1:
    if run1 in file_name1:
        full_name1 = os.path.join(path1,file_name1)
        command1.append(full_name1)

for file_name2 in ls2:
    if run2 in file_name2:
        full_name2 = os.path.join(path2,file_name2)
        command2.append(full_name2)

subprocess.call(command1) # execute shell command about "commnad1"
subprocess.call(command2) # execute shell command about "commnad2"

print('The process compleated!!')
