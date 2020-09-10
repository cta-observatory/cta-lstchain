#!/usr/bin/env python3

"""
Script to create event list of single run along with IRF files.
The IRF files are not stored in a separate fits file

The gammaness cut is the same for both IRFs and data
The selection cuts are applied using a separate yml file

Problems to solve:

 - Input: Path where the merged DL2 data HDF5 file is present
          MC gamma file path #other MC files are optional currently
          Source name
          Run number
          pyirf config file, if using a different than default one
 - Output: Event list of the input data file in fits format.


Usage:
$> python lstchain_create_event_list.py
--input-data-dir ./DL2/
--output-fits-dir ./DL3/
--input-file-gamma ./gamma/dl2_gamma_*.h5
--input-file-gamma-diff ./gamma-diff/dl2_gamma-diff*.h5 #optional for now
--input-file-proton ./proton/dl2_proton_*.h5 #optional for now
--input-file-electron ./electron/dl2_electron_*.h5 #optional for now
--source-name Crab
--run-number 0000
--pyirf-config ./config.yml
"""

import pandas as pd
import os
import yaml
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import argparse

from lstchain.reco.utils import camera_to_altaz
from lstchain.io.io import dl2_params_lstcam_key
from lstchain.clean import read_and_update_dl2, mc_filter, data_filter
from lstchain.hdu import create_event_list, create_obs_hdu_index

from pyirf.perf.irf_maker import IrfMaker
from astropy.io import fits

parser = argparse.ArgumentParser(description="DL2 to event_list")

# Required arguments
parser.add_argument('--input-data-dir', '-d', type=str,
                    dest='input_data_dir',
                    help='path to merged DL2 data HDF5 file',
                    default=None, required=True
                    )

parser.add_argument('--output-fits-dir', '-o', type=str,
                    dest='output_fits_dir',
                    help='path to output fits files',
                    default=None, required=True
                    )
parser.add_argument('--input-file-gamma', '-fg', type=str, dest='gamma_file',
                    help='Path to the dl2 file of gamma events for building IRF',
                    default=None, required=True
                    )
parser.add_argument('--source-name', '-s', type=str,
                    dest='source_name',
                    help='Name of the source',
                    default='Crab', required=True
                    )
parser.add_argument('--run-number', '-r', type=int,
                    dest='run_number',
                    help='If number of files is 1, provide run number',
                    default=None, required=True
                    )
#parser.add_argument('--num-files', '-n', type=int,
#                    dest='number_of_files',
#                    help='Number of files in sorted order',
#                    default=1, required=True
#                    )

# Optional arguments
parser.add_argument('--pyirf-config', '-irf-conf', type=str,
                    dest='pyirf_config',
                    help='Config file for creating IRF files by using pyIRF',
                    default=None, required=False
                    )
#For now as we are not produccing background IRFs, the other MC files can be optional
parser.add_argument('--input-file-gamma-diff', '-fg-diff', type=str, dest='gamma_diff_file',
                    help='Path to the dl2 file of gamma diffuse events for building IRF',
                    default=None, required=False
                    )
parser.add_argument('--input-file-proton', '-fp', type=str, dest='proton_file',
                    help='Path to the dl2 file of proton events for building IRF',
                    default=None, required=False
                    )
parser.add_argument('--input-file-electron', '-fe', type=str, dest='electron_file',
                    help='Path to the dl2 file of electron events for building IRF',
                    default=None, required=False
                    )

args = parser.parse_args()

def main():

    # read, update and filter files
    directory_fits_file=args.output_fits_dir

    # Assuming the nomenclature of merged DL2 file to be
    #dl2_Run_{#run_number}_merged_{#version}.h5
    start_name = 'dl2_Run'
    file = glob(args.input_data_dir+'dl2*'+str(args.run_number)+'*.h5')

    data = read_and_update_dl2(file[0])
    data = data_filter(data)


    gamma = read_and_update_dl2(args.gamma_file)
    gamma = mc_filter(gamma)
    #gamma_diff = read_and_update_dl2(args.gamma_diff_file)
    #gamma_diff = mc_filte(gamma_diff)
    #proton = read_and_update_dl2(args.proton_file)
    #proton = mc_filter(proton)
    #electron = read_and_update_dl2(args.electron_file)
    #electron = mc_filter(electron)

    # This is used just to run the irf_maker script and not the lst_performance
    # for getting the best optimised IRFs
    # When it is to be used, the config file's indir and outdir values have to
    # be edited accordingly
    if args.pyirf_config is not None:
        with open(args.pyirf_config, 'r') as check:
            config = yaml.loaf(check, Loader=yaml.FullLoader)
    else:
        cfg = os.path.join(os.path.dirname(__file__),'../data/pyirf_config.yml')
        with open(cfg, 'r') as check:
            config = yaml.load(check, Loader=yaml.FullLoader)

    evt_dict = dict(gamma=gamma) # proton=proton, electron=electron)

    # We are not saving a separate IRF file, but just using the HDU generated
    # to directly include in the DL3 file
    im = IrfMaker(config = config, evt_dict = evt_dict, outdir='.')

    #Returns HDUs
    aeff = im.make_effective_area()
    # Divide the Effective Area by 4 to normalize it for 1 stram of data and not the assumed 4
    aeff[1].data[0][4]/=4
    edisp = im.make_energy_dispersion()

    # Checking if after the filter cut, the run does not have significant number of events
    # Randomly selected 100 as minimum events
    #if n_event < 100: # check the time for the shorter events post cuts
    #    print('Number of events after cuts: ',n_event, 'for run', Run[i])
    #    continue

    #Create primary HDU
    n = np.arange(100) # a simple sequence of floats from 0.0 to 99.9
    primary_hdu = fits.PrimaryHDU(n)

    events, gti, pointing = create_event_list(data=data, Run=args.run_number,
                    Source_name=args.source_name)

    name_dl3_file = f"dl3_LST-1_0{args.run_number}_merged.fits"

    hdulist = fits.HDUList([primary_hdu, events, gti, pointing, aeff[1], edisp[1]])
    hdulist.writeto(f"{directory_fits_file}{name_dl3_file}",overwrite=True)

if __name__ == '__main__':
    main()
