#!/usr/bin/env python3

"""
Script to create DL3 of single (merged) DL2 run along with IRF files.
The IRF files are not stored in a separate fits file
The selection cuts are applied using a separate yml file

TODO:
Adding filter for size of files after cuts, to have significant number of events
Include angular cuts in filter_events
Generalize source_fov_offset binning

 - Input: Path where the merged DL2 data HDF5 file is present
          MC gamma file path #other MC files are optional currently
          Source name
          Adding IRF or not
 - Output: DL3 of the input DL2 data file in fits format.

Usage:
$> python lstchain_dl2_to_dl3.py
--input-data ./DL2/dl2_LST-1.Run*.h5
--output-fits-dir ./DL3/
--input-file-gamma ./gamma/dl2_gamma_*.h5
--input-file-gamma-diff ./gamma-diff/dl2_gamma-diff*.h5 #optional for now
--input-file-proton ./proton/dl2_proton_*.h5 #optional for now
--input-file-electron ./electron/dl2_electron_*.h5 #optional for now
--source-name Crab
--add-irf True

"""

import os
import json
import re
from distutils.util import strtobool
import numpy as np
import argparse
from pathlib import Path
import logging
import sys

from lstchain.hdu import create_event_list#, make_aeff2d_hdu, make_edisp2d_hdu
from lstchain.io import read_mc_dl2_to_pyirf, read_data_dl2_to_QTable, read_configuration_file, get_standard_config
from lstchain.reco.utils import filter_events

from astropy.io import fits
import astropy.units as u
from astropy.coordinates.angle_utilities import angular_separation

# pyIRF packages
from pyirf.io.gadf import create_aeff2d_hdu, create_energy_dispersion_hdu
from pyirf.irf import effective_area_per_energy, energy_dispersion#, effective_area_per_energy_and_fov
from pyirf.utils import calculate_source_fov_offset, calculate_theta

log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="DL2 to DL3")

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
#For now as we are not producing background IRFs, the other MC files can be optional
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

    if not args.input_data.is_file():
        log.error('Input Path does not exist or is not a file')
        sys.exit(1)
    file = str(args.input_data).split('/')[-1]

    output_dir = args.output_fits_dir.absolute()
    output_dir.mkdir(exist_ok=True)

    data = read_data_dl2_to_QTable(args.input_data)
    # Add angular separation column, different from pyirf function, because
    # it is used only for MC files.
    data['source_fov_offset'] = angular_separation(data['reco_az'] * u.rad,
                                              data['reco_alt'] * u.rad,
                                              data['az_tel'] * u.rad,
                                              data['alt_tel'] * u.rad,
                                              ).to(u.deg)

    # Get the obs_id from the filename if it is -1 in the column
    # Assuming the filename to be 'dl2_LST-1.Run#####_*.h5'
    # If the nomenclature is different, change the final index position to get the run number
    if data['obs_id'][0] <= 0:
        run_number = int(re.findall('\d+', file)[2])
    else:
        run_number= data['obs_id'][0]

    log.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    logging.getLogger().addHandler(handler)

    selection_config = os.path.join(os.path.dirname(__file__),"../data/data_selection_cuts.json")
    cuts = read_configuration_file(selection_config)

    data = filter_events(data, cuts["events_filters"])
    # Separate cuts for angular separations, for now. Will be included later in filter_events
    data = data[data["source_fov_offset"].to_value() < cuts["angular_filters"]["source_fov_offset"][1]]
    #Temporary filter for non/fewer events file
    #if len(data)<100:
    #    log.error(len(data), 'events only')
    #    log.error('Less than 100 events, please check the selection cuts')
    #    sys.exit(1)
    if args.add_irf:

        gamma, gamma_info = read_mc_dl2_to_pyirf(args.gamma_file)
        # Select the MC only for the same TEL as of real data
        gamma = gamma[gamma["tel_id"]==data["tel_id"][0]]
        # Add angular separation columns using pyirf's functions
        gamma["source_fov_offset"] = calculate_source_fov_offset(gamma)
        gamma["theta"] = calculate_theta(gamma, gamma["true_az"], gamma["true_alt"])

        gamma = filter_events(gamma, cuts["events_filters"])
        gamma = gamma[gamma["source_fov_offset"].to_value() < cuts["angular_filters"]["source_fov_offset"][1]]
        gamma = gamma[gamma["theta"].to_value() < cuts["angular_filters"]["theta_cut"][1]]
        #gamma_diff, gamma_diff_info = read_mc_dl2_to_pyirf(args.gamma_diff_file)
        #gamma_diff = mc_filter(gamma_diff)
        #proton, proton_info = read_mc_dl2_to_pyirf(args.proton_file)
        #proton = mc_filter(proton)
        #electron, electron_info = read_mc_dl2_to_pyirf(args.electron_file)
        #electron = mc_filter(electron)

        # Binning of parameters used in IRFs
        # This can be included in the config files
        #Number of bins is based on crude check for smooth distribution
        true_energy_bins = np.logspace(np.log10(0.05),np.log10(50), 10) *u.TeV
        ### TODO: The FoV offset angle is 0.4 deg for LST1 and it is used in
        # this manner because of an issue with astropy, which will be rectified
        # in a recent PR and updated in atropy v4.0.2
        # Later, we can just find the mean of FoV offset values, and use it
        # for the binning
        fov_offset_bins = [0.2, 0.6, 1.0] * u.deg
        migration_bins = np.linspace(0.2, 5, 31)

        area = effective_area_per_energy(gamma, gamma_info, true_energy_bins)
        effective_area = np.column_stack([area, np.zeros_like(area)])
        # use effective_area_per_energy_and_fov for diffuse MC
        #area = effective_area_energy_fov(gamma_diff, gamma_diff_info, true_energy_bins, fov_offset_bins)
        edisp = energy_dispersion(gamma, true_energy_bins, fov_offset_bins, migration_bins)

        # BinTableHDU is returned here
        aeff2d = create_aeff2d_hdu(effective_area, true_energy_bins, fov_offset_bins)
        edisp_2d = create_energy_dispersion_hdu(edisp,true_energy_bins, migration_bins, fov_offset_bins)
        # Using the HDU name as required by gammapy
        edisp_2d.header["EXTNAME"] = "ENERGY DISPERSION"

    #Create primary HDU
    n = np.arange(100) # a simple sequence of floats from 0.0 to 99.9
    primary_hdu = fits.PrimaryHDU(n)
    events, gti, pointing = create_event_list(data=data, run_number=run_number,
                    Source_name=args.source_name)

    name_dl3_file = file.replace('dl2', 'dl3')
    name_dl3_file = name_dl3_file.replace('h5', 'fits')

    if args.add_irf:
        hdulist = fits.HDUList([primary_hdu, events, gti, pointing, aeff2d, edisp_2d])
    else:
        hdulist = fits.HDUList([primary_hdu, events, gti, pointing])
    hdulist.writeto(str(args.output_fits_dir)+'/'+name_dl3_file,overwrite=True)

if __name__ == '__main__':
    main()
