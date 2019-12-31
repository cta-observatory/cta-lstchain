"""
Author: Yuto Nogami(Ibaraki Univ.), Hide Katagiri(Ibaraki Univ.)
Date: Dec 26, 2019.
# This scrept was validated in Dec 26, 2019.
# This script creates the plots to check the charge distribution and charge resolution.

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
import matplotlib.pyplot as plt
from ROOT import PyConfig
PyConfig.IgnoreCommandLineOptions=True # ignore cmd options for ROOT if it sets "True".
from ROOT import gROOT, gStyle
gROOT.SetBatch() # this is not to show the graph in GUI.
from ROOT import TCanvas, TH1D, TF1, TMath, TFile

parser = argparse.ArgumentParser(
    usage='$python sampling_interval_check_calib.py --root_file1 RunXXMergedNoCalib.root --root_file2 RunXXMergedCalib.root --outfile1 RunXXHistNoCalib.pdf --outfile2 RunXXHistCalib.pdf --resolution_file RunXXcharge_resolution.png --parameters_file RunXXPramFile.txt',
    description='This script can check the charge distribution and the cahrge resolution.\
                The charge distribution outputs as a pdf file, the charge resolution outputs\
                as a png file, the fitting parameters outputs as a text file.',
    add_help=True)

parser.add_argument('-file1', '--root_file1', help='input merged root-file name before\
                    timing-calibration', action='store',type=str, dest='root_file1')
parser.add_argument('-file2', '--root_file2', help='input merged root-file name after\
                    timing-calibration', action='store', type=str, dest='root_file2')
parser.add_argument('-out1', '--outfile1', help='output charge distributions before\
                    timing-calibration as a pdf file.', action='store',
                    type=str, dest='outfile1')
parser.add_argument('-out2', '--outfile2', help='output charge distributions after\
                    timing-calibration as a pdf file.', action='store',
                    type=str, dest='outfile2')
parser.add_argument('-reso', '--resolution_file', help='output 2D plot of charge resolution\
                    as a png file.', action='store', type=str, dest='reso_file')
parser.add_argument('-param', '--parameters_file', help='output file name for fitting parameters\
                    before/after timing-calibration, recommended file format is pdf', 
                    action='store', type=str, dest='param_file')

args = parser.parse_args()

# charge resolution plot setting
num_pixels = 1855
reso_a = np.zeros(num_pixels) # charge resolution after timing calibration
reso_b = np.zeros(num_pixels) # charge resolution before timing calibration

# pyROOT settings
num_fit_param =10 # mu, sigma, chi-square, ndf and prob. before/after timing-calibration.
num_page = num_pixels//4 + 1 # number of pages to output file about histograms.
canvas_NoCalib = [] # TCanvas array for hist_NoCalib.
canvas_Calib = [] # TCanvas array for hist_Calib.
fit_param = np.zeros((num_pixels, num_fit_param))
gauss_fit = TF1("fitting",'gaus') # set the fitting function.
gStyle.SetOptFit(1111) # show the box of some parameter.
for page_i in range(num_page): # create number of "num_page" TCanvases.
    canvas_NoCalib.append(TCanvas())
    canvas_Calib.append(TCanvas())

#open the merged files
merged_file1 = TFile(args.root_file1, "read")
merged_file2 = TFile(args.root_file2, "read")
# plot the charge distribution with fitting curve to TCanvases and get fitting parameters.
k = 0 # counting until num_pixels(=> pixel id 0-1854).
for i_page in range(num_page):
    canvas_NoCalib[i_page].Divide(2, 2, 1e-10, 1e-10)
    canvas_Calib[i_page].Divide(2, 2, 1e-10, 1e-10)
    for j_pad in range(4):
        # histogram before timing calibration
        canvas_NoCalib[i_page].cd(j_pad+1)
        hist1 = merged_file1.Get("pixel {} (NoCalib.)".format(k))
        hist1.Fit("fitting")
        hist1.Draw()
        fit_param[k, 0] = gauss_fit.GetParameter(1) # mu
        fit_param[k, 1] = gauss_fit.GetParameter(2) # sigma
        fit_param[k, 2] = gauss_fit.GetChisquare() # chi^2
        fit_param[k, 3] = gauss_fit.GetNDF() # ndf
        fit_param[k, 4] = gauss_fit.GetProb() # probability of chi^2 distribution.

        # histogram after timing calibration
        canvas_Calib[i_page].cd(j_pad+1)
        hist2 = merged_file2.Get("pixel {} (Calib.)".format(k))
        hist2.Fit("fitting")
        hist2.Draw()
        fit_param[k, 5] = gauss_fit.GetParameter(1) # mu
        fit_param[k, 6] = gauss_fit.GetParameter(2) # sigma
        fit_param[k, 7] = gauss_fit.GetChisquare() # chi^2
        fit_param[k, 8] = gauss_fit.GetNDF() # ndf
        fit_param[k, 9] = gauss_fit.GetProb() # probability of chi^2 distribution.
        if k == num_pixels - 1:
            break
        else:
            k += 1

    if i_page == 0:
        canvas_NoCalib[i_page].Print(args.outfile1 + "(")
        canvas_Calib[i_page].Print(args.outfile2 + "(")
    elif i_page == num_page - 1:
        canvas_NoCalib[i_page].Print(args.outfile1 + ")")
        canvas_Calib[i_page].Print(args.outfile2 + ")")
    else:
        canvas_NoCalib[i_page].Print(args.outfile1)
        canvas_Calib[i_page].Print(args.outfile2)

# calculate and plot cahrge resolution for all pixels
for pix in range(num_pixels):
        reso_b[pix] = fit_param[pix, 1]/fit_param[pix, 0] # i.e fit_sigma / fit_mu
        reso_a[pix] = fit_param[pix, 6]/fit_param[pix, 5]
        if reso_b[pix] < reso_a[pix]:
                print('pixel id {} is worse resolution!!'.format(pix))

base = np.arange(0, max(reso_b), 0.0001)
plt.title('charge resolution before & after timing calibration')
plt.xlabel('sigam / mu (before calib.)')
plt.ylabel('sigam / mu (after calib.)')
plt.scatter(reso_b, reso_a)
plt.plot(base, base, color='red')
plt.savefig(args.reso_file, dpi=200)
plt.close()

# output fitting parameters as a text format file.
np.savetxt(args.param_file, fit_param, fmt='%8e', delimiter=', ',
           header='These values are fitting parameters with gaussian.\n mu(NoCal.), sigma(NoCal.), chi squre(NoCal.), ndf(NoCal.), prob.(NoCal.), mu(Cal.), sigma(Cal.), chi squre(Cal.), ndf(Cal.), prob.(Cal.)')

print('The process compleated!!')
