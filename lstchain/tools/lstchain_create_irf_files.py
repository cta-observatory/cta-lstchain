"""
Create FITS file for IRFs from given MC DL2 files and selection cuts
"""

import os
from pathlib import Path
import numpy as np

from ctapipe.core import Tool, traits, Provenance
from lstchain.io import read_mc_dl2_to_pyirf, read_configuration_file, get_standard_config
from lstchain.reco.utils import filter_events

from astropy.io import fits
import astropy.units as u

from pyirf.io.gadf import create_aeff2d_hdu, create_energy_dispersion_hdu
from pyirf.irf import effective_area_per_energy, energy_dispersion, effective_area_per_energy_and_fov
from pyirf.utils import calculate_source_fov_offset, calculate_theta
from pyirf.binning import create_bins_per_decade, add_overflow_bins

__all__ = [
    'IRFFITSWriter'
]

class IRFFITSWriter(Tool):
    name = "IRFFITSWriter"
    description = "Create IRF FITS file from given MC DL2 files and selection cuts"

    input_gamma_dl2 = traits.Path(
        help="Input MC gamma DL2 file", exists=True, directory_ok=False, file_ok=True
    ).tag(config=True)
    output_irf_path = traits.Path(
        help="IRF output filedir",
        directory_ok=True, file_ok=False
    ).tag(config=True)
    pnt_like = traits.Bool(
        help="True for point-like IRF, False for Full Enclosure",
        default_value=False,
    ).tag(config=True)
    config_file = traits.Path(
        help="Config file for selection cuts", directory_ok=False, file_ok=True,
    ).tag(config=True)
    input_gamma_diff_dl2 = traits.Path(
        help="Input MC diffuse gamma DL2 file", exists=True, directory_ok=False,
        file_ok=True
    ).tag(config=True)

    aliases = {
        "input_gamma_dl2" : "IRFFITSWriter.input_gamma_dl2",
        "fg" : "IRFFITSWriter.input_gamma_dl2",
        "output_irf_path" : "IRFFITSWriter.output_irf_path",
        "o" : "IRFFITSWriter.output_irf_path",
        "pnt_like" : "IRFFITSWriter.pnt_like",
        "pnt" : "IRFFITSWriter.pnt_like",
        "config_file" : "IRFFITSWriter.config_file",
        "conf" : "IRFFITSWriter.config_file",
        "input_gamma_diff_dl2" : "IRFFITSWriter.input_gamma_diff_dl2",
        "fg_diff" : "IRFFITSWriter.input_gamma_diff_dl2",
    }

    flag = {
        "pnt_like": (
        {"IRFFITSWriter": {"pnt_like":False}},
        "Full Enclosure IRFs will be produced"
        ),
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        """
        Tool that creates a compressed FITS file of IRFs for a given MC DL2 file
        and with the given selection cuts.
        For getting help run:
        lstchain_create_irf_files --help
        """
        self.mc_gamma = None
        self.cuts = None
        self.hdus = None
        self.effective_area = None
        self.edisp = None
        self.output_file = None

    def setup(self):

        if self.config_file is None:
            self.cuts = read_configuration_file(os.path.join(os.path.dirname(__file__), '../data/data_selection_cuts.json'))
        else:
            self.cuts = read_configuration_file(self.config_file)

        if str(self.input_gamma_dl2) is not None:
            self.mc_gamma = {"file": str(self.input_gamma_dl2),
                     "type": "point-like",}
        else:
            self.mc_gamma = {"file": str(self.input_gamma_diff_dl2),
                     "type": "diffuse",}

        #Read and update MC information
        self.log.info(f'Simulated {self.mc_gamma["type"]} Gamma Events:')
        self.mc_gamma["events"], self.mc_gamma["simulation_info"] = read_mc_dl2_to_pyirf(self.mc_gamma["file"])
        self.mc_gamma["events"]["true_source_fov_offset"] = calculate_source_fov_offset(self.mc_gamma["events"], prefix='true')
        # calculate theta / distance between reco and assumed source position
        self.mc_gamma["events"]["theta"] = calculate_theta(
                        self.mc_gamma["events"],
                        assumed_source_az=self.mc_gamma["events"]["true_az"],
                        assumed_source_alt=self.mc_gamma["events"]["true_alt"],
                        )
        self.log.info(self.mc_gamma["simulation_info"])

    def start(self):

        gammas = self.mc_gamma["events"]

        gh_cut = self.cuts["fixed_cuts"]["gh_score"][0]
        self.log.info(f"Using fixed G/H cut of {gh_cut} to calculate theta cuts")

        gammas = filter_events(gammas, self.cuts["events_filters"])

        #Filtering the tels needed to use with the real data
        #Add MAGIC tels when need be
        tel_ids = self.cuts["LST_tels"]["tel_list"]
        for i in tel_ids:
            gammas["selected_tels"] = gammas["tel_id"] == i

        gammas["selected_gh"] = gammas["gh_score"] > gh_cut
        #irf_type = True for point like IRFs, False for Full Enclosure IRFs
        if self.pnt_like:
            gammas["selected_theta"] = gammas["theta"] < u.Quantity(
                                        **self.cuts["fixed_cuts"]["theta_cut"])
            gammas["selected_fov"] = gammas["true_source_fov_offset"] < u.Quantity(
                                    **self.cuts["fixed_cuts"]["source_fov_offset"])
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
        if str(self.input_gamma_dl2) is None:
            fov_offset_bins = [0.2, 0.6] * u.deg
        else:
            # temporary usage of bins as used in MAGIC
            fov_offset_bins = [0,0.3,0.5,0.7,0.9,1.1] * u.deg
        migration_bins = np.linspace(0.2, 5, 31)

        self.log.info("Generating IRF HDUs")
        # Write HDUs
        self.hdus = [fits.PrimaryHDU(),]
        with np.errstate(invalid='ignore', divide='ignore'):
            if self.input_gamma_diff_dl2 is None:
                self.effective_area = effective_area_per_energy(gammas[gammas["selected"]],
                                        self.mc_gamma["simulation_info"], true_energy_bins)
            else:
                self.effective_area = effective_area_per_energy_and_fov(gammas[gammas["selected"]],
                                                self.mc_gamma["simulation_info"],
                                                true_energy_bins, fov_offset_bins)
        #Adding a dimension for FoV offset for effective area
        self.hdus.append(create_aeff2d_hdu(self.effective_area[..., np.newaxis],
                        true_energy_bins, fov_offset_bins, extname = "EFFECTIVE AREA"))
        self.log.info("Effective Area HDU created")
        self.edisp = energy_dispersion(gammas[gammas["selected"]],
                    true_energy_bins, fov_offset_bins, migration_bins)
        self.hdus.append(create_energy_dispersion_hdu(self.edisp,true_energy_bins,
                        migration_bins, fov_offset_bins, extname = "ENERGY DISPERSION"))
        self.log.info("Energy Dispersion HDU created")

    def finish(self):
        self.output_file = self.output_irf_path/"irf.fits.gz"
        if self.output_file.exists():
            self.log.info(f"{self.output_file} exists, will be overwritten")

        fits.HDUList(self.hdus).writeto(self.output_file, overwrite=True)
        Provenance().add_output_file(self.output_file)

def main():
    tool = IRFFITSWriter()
    tool.run()

if __name__ == "__main__":
    main()
