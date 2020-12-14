#!/usr/bin/env python3

"""
Script to generate IRFs from MC DL2 files, using pyirf functions.
This will save the IRFs, and either use fixed cuts or optimise them.
and later it can be used by lstchain_dl2_to_dl3 to generate DL3 files.

TODO: Add other IRFs
Use optimised cuts and binning

- Input: Path where the merged DL2 data HDF5 files are present
- Output: IRFs compiled in fits.gz format

Usage:
$> python lstchain_mc_dl2_to_irf.py
--input-file-gamma ./gamma/dl2_gamma_*.h5
--input-file-gamma-diff ./gamma-diff/dl2_gamma-diff*.h5 #optional for now
--output-dir ./IRFs/
--pnt-like True
--config ../../data/data_selection_cuts.json
"""

import os
import json
import numpy as np
import argparse
from pathlib import Path
from distutils.util import strtobool
import logging

from lstchain.io import read_mc_dl2_to_pyirf, read_configuration_file, get_standard_config
from lstchain.reco.utils import filter_events

from astropy.io import fits
import astropy.units as u
from astropy import table

# pyIRF packages
from pyirf.io.gadf import (create_aeff2d_hdu, create_energy_dispersion_hdu)
from pyirf.irf import (effective_area_per_energy, energy_dispersion, effective_area_per_energy_and_fov)
from pyirf.utils import calculate_source_fov_offset, calculate_theta

from pyirf.binning import (
    create_bins_per_decade,
    add_overflow_bins,
)

log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="MC DL2 to IRF")

# Required arguments
parser.add_argument('--input-file-gamma', '-fg', type=Path, dest='gamma_file',
                    help='Path to the DL file of point like gamma events for building IRF',
                    default=None, required=False
                    )

parser.add_argument('--output-irf-dir', '-o', type=Path, dest='output_irf_dir',
                    help='Path to output IRF files',
                    default=None, required=True
                    )

#Optional arguments
parser.add_argument('--input-file-gamma-diff', '-fg-diff', type=Path, dest='gamma_diff_file',
                    help='Path to the dl2 file of gamma diffuse events for building IRF',
                    default=None, required=False
                    )

parser.add_argument('--point-like', '-pnt', action='store',
                    type=lambda x: bool(strtobool(x)), dest='point_like',
                    help='True for point-like IRF, False for Full Enclosure',
                    required=True
                    )

parser.add_argument('--config', '-conf', type=Path,
                    dest='config',
                    help='Config file for selection cuts',
                    default=None, required=False
                    )

args = parser.parse_args()

def main():

    output_dir = args.output_irf_dir.absolute()
    output_dir.mkdir(exist_ok=True)

    log.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    logging.getLogger().addHandler(handler)

    if args.config is None:
        cuts = read_configuration_file(os.path.join(os.path.dirname(__file__), '../data/data_selection_cuts.json'))
    else:
        cuts = read_configuration_file(args.config)

    if args.gamma_diff_file is None:
        mc_gamma = {"file": args.gamma_file,
                 "type": "point-like",}
    else:
        mc_gamma = {"file": args.gamma_diff_file,
                 "type": "diffuse",}

    #Read and update MC information
    log.info(f'Simulated {mc_gamma["type"]} Gamma Events:')
    mc_gamma["events"], mc_gamma["simulation_info"] = read_mc_dl2_to_pyirf(mc_gamma["file"])
    mc_gamma["events"]["true_source_fov_offset"] = calculate_source_fov_offset(mc_gamma["events"], prefix='true')
    # calculate theta / distance between reco and assumed source position
    mc_gamma["events"]["theta"] = calculate_theta(
                    mc_gamma["events"],
                    assumed_source_az=mc_gamma["events"]["true_az"],
                    assumed_source_alt=mc_gamma["events"]["true_alt"],
                    )
    log.info(mc_gamma["simulation_info"])

    #Apply selection cuts
    gammas = mc_gamma["events"]

    gh_cut = cuts["fixed_cuts"]["gh_score"][0]
    log.info(f"Using fixed G/H cut of {gh_cut} to calculate theta cuts")

    gammas = filter_events(gammas, cuts["events_filters"])

    #Filtering the tels needed to use with the real data
    #Add MAGIC tels when need be
    tel_ids = cuts["LST_tels"]["tel_list"]
    for i in tel_ids:
        gammas["selected_tels"] = gammas["tel_id"] == i

    gammas["selected_gh"] = gammas["gh_score"] > gh_cut
    #irf_type = True for point like IRFs, False for Full Enclosure IRFs
    if args.point_like:
        gammas["selected_theta"] = gammas["theta"] < u.Quantity(**cuts["fixed_cuts"]["theta_cut"])
        gammas["selected_fov"] = gammas["true_source_fov_offset"] < u.Quantity(**cuts["fixed_cuts"]["source_fov_offset"])
        #Combining selection cuts
        gammas["selected"] = gammas["selected_theta"] & \
                            gammas["selected_gh"] & \
                            gammas["selected_fov"] & \
                            gammas["selected_tels"]
    else:
        gammas["selected"] = gammas["selected_gh"] & \
                            gammas["selected_tels"]

    # Binning of parameters used in IRFs
    #12.5 GeV - 51.28 TeV
    true_energy_bins =  create_bins_per_decade(0.01 * u.TeV, 100 * u.TeV, 5.5)
    #add_overflow_bins(***)[1:-1]
    # The overflow binning added is not needed in the current script
    reco_energy_bins = create_bins_per_decade(0.01 * u.TeV, 100 * u.TeV, 5.5)

    #TODO: Generalize FoV offset binning
    if args.gamma_diff_file is None:
        fov_offset_bins = [0.2, 0.6] * u.deg
    else:
        # temporary usage of bins as used in MAGIC
        fov_offset_bins = [0,0.3,0.5,0.7,0.9,1.1] * u.deg
    migration_bins = np.linspace(0.2, 5, 31)

    # Write HDUs
    hdus = [fits.PrimaryHDU(),]
    with np.errstate(invalid='ignore', divide='ignore'):
        if args.gamma_diff_file is None:
            effective_area = effective_area_per_energy(gammas[gammas["selected"]], mc_gamma["simulation_info"], true_energy_bins)
        else:
            effective_area = effective_area_per_energy_and_fov(gammas[gammas["selected"]], mc_gamma["simulation_info"], true_energy_bins, fov_offset_bins)
    #Adding a dimension for FoV offset for effective area
    hdus.append(create_aeff2d_hdu(effective_area[..., np.newaxis],true_energy_bins, fov_offset_bins, extname = "EFFECTIVE AREA"))

    edisp = energy_dispersion(gammas[gammas["selected"]], true_energy_bins, fov_offset_bins, migration_bins)
    hdus.append(create_energy_dispersion_hdu(edisp,true_energy_bins, migration_bins, fov_offset_bins, extname = "ENERGY DISPERSION"))

    output_file = output_dir/"irf.fits.gz"
    fits.HDUList(hdus).writeto(output_file, overwrite=True)

if __name__ == "__main__":
    main()
