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
--input-file-proton ./proton/dl2_proton_*.h5 #optional for now
--input-file-electron ./electron/dl2_electron_*.h5 #optional for now
--output-dir ./IRFs/
--pnt-like True
--config ../../data/data_selection_cuts.json
"""

import os
import json
import numpy as np
import argparse
from pathlib import Path
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
from pyirf.spectral import (
    calculate_event_weights,
    PowerLaw,
    CRAB_HEGRA,
    IRFDOC_PROTON_SPECTRUM,
    IRFDOC_ELECTRON_SPECTRUM,
)
from pyirf.binning import (
    create_bins_per_decade,
    add_overflow_bins,
    create_histogram_table,
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

parser.add_argument('--input-file-proton', '-fp', type=Path, dest='proton_file',
                    help='Path to the dl2 file of proton events for building IRF',
                    default=None, required=False
                    )

parser.add_argument('--input-file-electron', '-fe', type=Path, dest='electron_file',
                    help='Path to the dl2 file of electron events for building IRF',
                    default=None, required=False
                    )

parser.add_argument('--pnt-like', '-pnt', action='store',
                    type=lambda x: bool(strtobool(x)), dest='irf_type',
                    help='True for point-like IRF, False for Full Enclosure',
                    default=True, required=False
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
        particles = {
	       "gamma": {
	          "file": args.gamma_file,
	          "target_spectrum": CRAB_HEGRA,
	          },
        #	"proton": {
        #             "file": args.proton_file,
        #	      "target_spectrum": IRFDOC_PROTON_SPECTRUM,
        #             },
        #	"electron": {
        #             "file": args.electron_file,
        #	      "target_spectrum": IRFDOC_ELECTRON_SPECTRUM,
        #             },
    	}
    else:
        #Using proton spectrum at the moment for unit tests
        particles = {
	       "gamma": {
	          "file": args.gamma_diff_file,
	          "target_spectrum": IRFDOC_PROTON_SPECTRUM,
	          },
        #	"proton": {
        #             "file": args.proton_file,
        #	      "target_spectrum": IRFDOC_PROTON_SPECTRUM,
        #             },
        #	"electron": {
        #             "file": args.electron_file,
        #	      "target_spectrum": IRFDOC_ELECTRON_SPECTRUM,
        #             },
    	}

    for k, p in particles.items():
        log.info(f"Simulated {k.title()} Events:")
        p["events"], p["simulation_info"] = read_mc_dl2_to_pyirf(p["file"])
        p["simulated_spectrum"] = PowerLaw.from_simulation(p["simulation_info"], 50 * u.hour)
        p["events"]["weight"] = calculate_event_weights(
            p["events"]["true_energy"], p["target_spectrum"], p["simulated_spectrum"]
        )
        p["events"]["source_fov_offset"] = calculate_source_fov_offset(p["events"])
        # calculate theta / distance between reco and assumed source position
        p["events"]["theta"] = calculate_theta(
            p["events"],
            assumed_source_az=p["events"]["true_az"],
            assumed_source_alt=p["events"]["true_alt"],
        )
        log.info(p["simulation_info"])

    gammas = particles["gamma"]["events"]
    # selecting 1 tel_id as the data is collected from only 1 for now
    gammas = gammas[gammas["tel_id"]==1]
    #background = table.vstack(
    #    [particles["proton"]["events"], particles["electron"]["events"]]
    #)

    gh_cut = cuts["fixed_cuts"]["gh_score"][0]
    log.info(f"Using fixed G/H cut of {gh_cut} to calculate theta cuts")

    gammas = filter_events(gammas, cuts["events_filters"])

    gammas["selected_gh"] = gammas["gh_score"] > cuts["fixed_cuts"]["gh_score"][0]
    #For point like gammas
    gammas["selected_theta"] = gammas["theta"] < u.Quantity(**cuts["fixed_cuts"]["theta_cut"])
    gammas["selected_fov"] = gammas["source_fov_offset"] < u.Quantity(**cuts["fixed_cuts"]["source_fov_offset"])
    gammas["selected"] = gammas["selected_theta"] & gammas["selected_gh"] & gammas["selected_fov"]

    # Binning of parameters used in IRFs
    true_energy_bins =  add_overflow_bins(
        create_bins_per_decade(10 ** -1.9 * u.TeV, 10 ** 1.71 * u.TeV, 10)
    )[:-1]
    # The overflow binning added is not needed in the current script

    reco_energy_bins = add_overflow_bins(
        create_bins_per_decade(10 ** -1.9 * u.TeV, 10 ** 1.71 * u.TeV, 5)
    )[:-1]

    ### TODO: The FoV offset angle is 0.4 deg for LST1 and it is used in
    # this manner because of an issue with astropy which will be updated
    # in atropy v4.0.2, and later in pyirf and gammapy as well
    # Later, we can just find the mean of FoV offset values, and use it
    # for the binning
    fov_offset_bins = [0.2, 0.6, 1.0] * u.deg
    migration_bins = np.linspace(0.2, 5, 31)

    # Write HDUs
    hdus = [fits.PrimaryHDU(),]
    with np.errstate(invalid='ignore', divide='ignore'):
        effective_area = effective_area_per_energy(gammas[gammas["selected"]], particles["gamma"]["simulation_info"], true_energy_bins)
        effective_area = np.column_stack([effective_area, np.zeros_like(effective_area)])
    hdus.append(create_aeff2d_hdu(effective_area,true_energy_bins, fov_offset_bins,extname = "EFFECTIVE AREA")) #[..., np.newaxis]
    # use effective_area_per_energy_and_fov for diffuse MC

    edisp = energy_dispersion(gammas[gammas["selected"]], true_energy_bins, fov_offset_bins, migration_bins)
    hdus.append(create_energy_dispersion_hdu(edisp,true_energy_bins, migration_bins, fov_offset_bins, extname = "ENERGY DISPERSION"))

    output_file = (output_dir/"irf.fits.gz")
    fits.HDUList(hdus).writeto(output_file, overwrite=True)

if __name__ == "__main__":
    main()
