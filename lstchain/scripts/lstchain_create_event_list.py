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
--input-data ./DL2/dl2_LST-1.Run*.h5
--output-fits-dir ./DL3/
--input-file-gamma ./gamma/dl2_gamma_*.h5
--input-file-gamma-diff ./gamma-diff/dl2_gamma-diff*.h5 #optional for now
--input-file-proton ./proton/dl2_proton_*.h5 #optional for now
--input-file-electron ./electron/dl2_electron_*.h5 #optional for now
--source-name Crab
--add-irf True
--pyirf-config ./config.yml
"""

import pandas as pd
import os
import yaml
import re
from glob import glob
from distutils.util import strtobool
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
import logging
import sys

from lstchain.reco.utils import camera_to_altaz
from lstchain.io.io import dl2_params_lstcam_key
from lstchain.clean import read_and_update_dl2, mc_filter, data_filter
from lstchain.hdu import create_event_list, create_obs_hdu_index

from pyirf.perf.irf_maker import IrfMaker
from astropy.io import fits

log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="DL2 to event_list")

# Required arguments
parser.add_argument('--input-data', '-d', type=Path,
                    dest='input_data',
                    help='path to merged DL2 data HDF5 file',
                    default=None, required=True
                    )

parser.add_argument('--output-fits-dir', '-o', type=Path,
                    dest='output_fits_dir',
                    help='path to output fits files',
                    default=None, required=True
                    )

parser.add_argument('--input-file-gamma', '-fg', type=Path, dest='gamma_file',
                    help='Path to the dl2 file of gamma events for building IRF',
                    default=None, required=True
                    )

parser.add_argument('--source-name', '-s', type=str,
                    dest='source_name',
                    help='Name of the source',
                    default='Crab', required=True
                    )

parser.add_argument('--add-irf', '-irf', action='store', type=lambda x: bool(strtobool(x)),
                    dest='add_irf',
                    help='Boolean: True to add IRF to DL3',
                    default=False, required=True
                    )

# Optional arguments
parser.add_argument('--pyirf-config', '-irf-conf', action='store', type=str,
                    dest='pyirf_config',
                    help='Config file for creating IRF files by using pyIRF',
                    default=None, required=False
                    )

#For now as we are not produccing background IRFs, the other MC files can be optional
parser.add_argument('--input-file-gamma-diff', '-fg-diff', type=Path, dest='gamma_diff_file',
                    help='Path to the dl2 file of gamma diffuse events for building IRF',
                    default=None, required=False
                    )

parser.add_argument('--input-file-proton', '-fp', type=Path, dest='proton_file',
                    help='Path to the dl2 file of proton events for building IRF',
                    default=None, required=False
                    )

parser.add_argument('--input-file-electron', '-fe', type=Path, dest='electron_file',
                    help='Path to the dl2 file of electron events for building IRF',
                    default=None, required=False
                    )

args = parser.parse_args()

def main():

    # read, update and filter files
    if not args.input_data.is_file():
        log.error('Input Path does not exist or is not a file')
        sys.exit(1)

    file = str(args.input_data).split('/')[-1]

    output_dir = args.output_fits_dir.absolute()
    output_dir.mkdir(exist_ok=True)

    data = read_and_update_dl2(args.input_data)

    # Get the obs_id from the filename if it is -1 in the column
    # Assuming the filename to be 'dl2_LST-1.Run#####_*.h5'
    # If the nomenclature is different, change the final index position to get the run number
    if data.obs_id[0] <= 0:
        run_number = int(re.findall('\d+', file)[2])
    else:
        run_number= data.obs_id[0]

    log.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    logging.getLogger().addHandler(handler)

    data = data_filter(data)
    #Temporary filter for non/fewer events file
    if len(data)<100:
        log.error('Less than 100 events, please check the selection cuts')
        sys.exit(1)

    if args.add_irf:
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
                config = yaml.load(check, Loader=yaml.FullLoader)
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

    #Create primary HDU
    n = np.arange(100) # a simple sequence of floats from 0.0 to 99.9
    primary_hdu = fits.PrimaryHDU(n)

    events, gti, pointing = create_event_list(data=data, run_number=run_number,
                    Source_name=args.source_name)

    name_dl3_file = file.replace('dl2', 'dl3')#f"dl3_LST-1_{run_number:05d}_merged.fits"

    if args.add_irf:
        hdulist = fits.HDUList([primary_hdu, events, gti, pointing, aeff[1], edisp[1]])
    else:
        hdulist = fits.HDUList([primary_hdu, events, gti, pointing])
    hdulist.writeto(str(args.output_fits_dir)+'/'+name_dl3_file,overwrite=True)

if __name__ == '__main__':
    main()
