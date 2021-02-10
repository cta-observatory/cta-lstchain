"""
Create FITS file for IRFs from given MC DL2 files and selection cuts
MC gamma files can be point-like or diffuse
IRFs can be point-like or Full Enclosure
Currently using spectral weighting with the spectra given in pyirf.
It has to be updated with the ones in lstchain.spectra
And the Background HDU has to be checked with recent MC data

Simple usage with MC gamma, argument aliases and default config file for selection cuts:

lstchain_create_irf_files
    --fg /path/to/DL2_MC_gamma_file.h5
    --o /path/to/irf.fits.gz
    --pnt True
"""

import os
import numpy as np

from ctapipe.core import Tool, traits, Provenance
from lstchain.io import read_mc_dl2_to_pyirf, read_configuration_file
from lstchain.reco.utils import filter_events

from astropy.io import fits
import astropy.units as u
from astropy import table

from pyirf.io.gadf import (create_aeff2d_hdu,
                        create_energy_dispersion_hdu,
                        create_background_2d_hdu)
from pyirf.irf import (effective_area_per_energy,
                        energy_dispersion,
                        effective_area_per_energy_and_fov,
                        background_2d)
from pyirf.spectral import (
    calculate_event_weights,
    PowerLaw,
    CRAB_HEGRA,
    IRFDOC_PROTON_SPECTRUM,
    IRFDOC_ELECTRON_SPECTRUM,
)
from pyirf.utils import calculate_source_fov_offset, calculate_theta
from pyirf.binning import create_bins_per_decade #, add_overflow_bins

__all__ = [
    'IRFFITSWriter'
    ]

class IRFFITSWriter(Tool):
    name = "IRFFITSWriter"
    description = __doc__

    input_gamma_dl2 = traits.Path(
        help="Input MC gamma DL2 file",
        exists=True,
        directory_ok=False,
        file_ok=True
        ).tag(config=True)

    input_proton_dl2 = traits.Path(
        help="Input MC proton DL2 file",
        exists=True,
        directory_ok=False,
        file_ok=True
        ).tag(config=True)

    input_electron_dl2 = traits.Path(
        help="Input MC electron DL2 file",
        exists=True,
        directory_ok=False,
        file_ok=True
        ).tag(config=True)

    output_irf_file = traits.Path(
        help="IRF output file",
        directory_ok=False,
        file_ok=True,
        default_value="./irf.fits.gz"
        ).tag(config=True)

    point_like = traits.Bool(
        help="True for point-like IRF, False for Full Enclosure",
        default_value=False,
        ).tag(config=True)

    config_file = traits.Path(
        help="Config file for selection cuts",
        directory_ok=False,
        file_ok=True,
        ).tag(config=True)

    aliases = {
        "input_gamma_dl2" : "IRFFITSWriter.input_gamma_dl2",
        "fg" : "IRFFITSWriter.input_gamma_dl2",
        "input_proton_dl2" : "IRFFITSWriter.input_proton_dl2",
        "fp" : "IRFFITSWriter.input_proton_dl2",
        "input_electron_dl2" : "IRFFITSWriter.input_electron_dl2",
        "fe" : "IRFFITSWriter.input_electron_dl2",
        "output_irf_file" : "IRFFITSWriter.output_irf_file",
        "o" : "IRFFITSWriter.output_irf_file",
        "point_like" : "IRFFITSWriter.point_like",
        "pnt" : "IRFFITSWriter.point_like",
        "config_file" : "IRFFITSWriter.config_file",
        "conf" : "IRFFITSWriter.config_file",
        }

    flag = {
        "point_like": (
            {"IRFFITSWriter": {
                        "point_like": False}
                        },
            "Full Enclosure IRFs will be produced"
            ),
        }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mc_particle = None
        self.cuts = None
        self.hdus = None
        self.effective_area = None
        self.edisp = None
        self.backgroud = None
        # self.psf = None

    def setup(self):
        filename = os.path.basename(self.output_irf_file)
        if filename.split('.')[1:] != ['fits', 'gz']:
            self.log.error(f"{filename} is not a correct "
                        "compressed FITS file name. It will be corrected.")
            filename = filename.split('.')[0]+'.fits.gz'
            self.output_irf_file = os.path.join(
                                        os.path.dirname(self.output_irf_file),
                                        filename
                                        )

        if self.config_file is None:
            self.cuts = read_configuration_file(os.path.join(
                                            os.path.dirname(__file__),
                                            '../data/data_selection_cuts.json'))
        else:
            self.cuts = read_configuration_file(self.config_file)

        # Read and update MC information
        # Temporary if-else condition to either just use MC gamma or
        # to also use MC proton and electron to generate Background IRF
        if self.input_proton_dl2 is None:
            self.mc_particle = {"gamma":
                                    {
                                    "file": str(self.input_gamma_dl2),
                                    "target_spectrum": CRAB_HEGRA,
                                    }
                                }
        else:
            self.mc_particle = {"gamma":
                                    {
                                    "file": str(self.input_gamma_dl2),
                                    "target_spectrum": CRAB_HEGRA,
                                    },
                                "proton":
                                    {
                                    "file": str(self.input_proton_dl2),
                                    "target_spectrum": IRFDOC_PROTON_SPECTRUM,
                                    },
                                "electron":
                                    {
                                    "file": str(self.input_electron_dl2),
                                    "target_spectrum": IRFDOC_ELECTRON_SPECTRUM,
                                    },
                                }

        for particle_type, p in self.mc_particle.items():
            self.log.info(f"Simulated {particle_type.title()} Events:")
            p["events"], p["simulation_info"] = read_mc_dl2_to_pyirf(p["file"])

            if p["simulation_info"].viewcone.value == 0.:
                p["mc_type"] = "point-like"
            else:
                p["mc_type"] = "diffuse"
                # For diffuse gamma using Proton Spectra for calculating event weights
                if particle_type == 'gamma':
                    p["target_spectrum"] = IRFDOC_PROTON_SPECTRUM
                    self.log.info('Proton spectrum used as target spectrum'
                                    ' for MC diffuse gamma')

            self.log.info(f"Simulated {p['mc_type']} {particle_type.title()} Events:")

            p["simulated_spectrum"] = PowerLaw.from_simulation(
                                                    p["simulation_info"],
                                                    50*u.hour)
            p["events"]["weight"] = calculate_event_weights(
                                            p["events"]["true_energy"],
                                            p["target_spectrum"],
                                            p["simulated_spectrum"])

            for prefix in ('true', 'reco'):
                k = f"{prefix}_source_fov_offset"
                p["events"][k] = calculate_source_fov_offset(
                                                        p["events"],
                                                        prefix=prefix
                                                        )
            # calculate theta / distance between reco and assumed source position
            p["events"]["theta"] = calculate_theta(
                                p["events"],
                                assumed_source_az=p["events"]["true_az"],
                                assumed_source_alt=p["events"]["true_alt"],
                                )
            self.log.info(p["simulation_info"])

    def start(self):

        gammas = self.mc_particle["gamma"]["events"]

        gh_cut = self.cuts["fixed_cuts"]["gh_score"][0]
        self.log.info(f"Using fixed G/H cut of {gh_cut} to calculate theta cuts")

        gammas = filter_events(gammas, self.cuts["events_filters"])

        # Filtering the tels needed to use with the real data
        # Add MAGIC tels when need be
        tel_ids = self.cuts["LST_tels"]["tel_list"]
        for i in tel_ids:
            gammas["selected_tels"] = gammas["tel_id"] == i

        gammas["selected_gh"] = gammas["gh_score"] > gh_cut

        # point_like = True for point like IRFs, False for Full Enclosure IRFs
        if self.point_like:
            gammas["selected_theta"] = gammas["theta"] < u.Quantity(
                                        **self.cuts["fixed_cuts"]["theta_cut"])
            gammas["selected_fov"] = gammas["true_source_fov_offset"] < u.Quantity(
                                    **self.cuts["fixed_cuts"]["source_fov_offset"])
            # Combining selection cuts
            gammas["selected"] = gammas["selected_theta"] & \
                                gammas["selected_gh"] & \
                                gammas["selected_fov"] & \
                                gammas["selected_tels"]
        else:
            gammas["selected"] = gammas["selected_gh"] & \
                                gammas["selected_tels"]

        # For Background IRF
        if self.input_proton_dl2:
            background = table.vstack(
                                [self.mc_particle["proton"]["events"],
                                self.mc_particle["electron"]["events"]]
            )

            background = filter_events(background, self.cuts["events_filters"])

            for i in tel_ids:
                background["selected_tels"] = background["tel_id"] == i
            background["selected_gh"] = background["gh_score"] > gh_cut
            background["selected"] = background["selected_gh"] & \
                                    background["selected_tels"]


        # Binning of parameters used in IRFs
        # 12.5 GeV - 51.28 TeV
        true_energy_bins =  create_bins_per_decade(0.01 * u.TeV, 100 * u.TeV, 5.5)
        # add_overflow_bins(***)[1:-1]
        # The overflow binning added is not needed in the current script
        reco_energy_bins = create_bins_per_decade(0.01 * u.TeV, 100 * u.TeV, 5.5)

        # TODO: Generalize FoV offset binning
        if self.mc_particle["gamma"]["mc_type"] == "point-like":
            fov_offset_bins = [0.2, 0.6] * u.deg
        else:
            # temporary usage of bins as used in MAGIC
            fov_offset_bins = [0,0.3,0.5,0.7,0.9,1.1] * u.deg
        migration_bins = np.linspace(0.2, 5, 31)

        if self.point_like:
            self.log.info("Generating Point-Like IRF HDUs")
        else:
            self.log.info("Generating Full-Enclosure IRF HDUs")

        # Write HDUs
        self.hdus = [fits.PrimaryHDU(),]
        extra_headers = {"TELESCOP":"CTA", "INSTRUME":"LST-1"}

        with np.errstate(invalid='ignore', divide='ignore'):
            if self.mc_particle["gamma"]["mc_type"] == "point-like":
                self.effective_area = effective_area_per_energy(
                                                gammas[gammas["selected"]],
                                                self.mc_particle["gamma"]["simulation_info"],
                                                true_energy_bins)
                self.hdus.append(create_aeff2d_hdu(
                                    self.effective_area[..., np.newaxis],
                                    true_energy_bins,
                                    fov_offset_bins,
                                    point_like=self.point_like,
                                    extname = "EFFECTIVE AREA",
                                    **extra_headers)
                                    )
            else:
                self.effective_area = effective_area_per_energy_and_fov(
                                                gammas[gammas["selected"]],
                                                self.mc_particle["gamma"]["simulation_info"],
                                                true_energy_bins,
                                                fov_offset_bins)
                self.hdus.append(create_aeff2d_hdu(
                                    self.effective_area,
                                    true_energy_bins,
                                    fov_offset_bins,
                                    point_like=self.point_like,
                                    extname = "EFFECTIVE AREA",
                                    **extra_headers)
                                    )
        # Adding a dimension for FoV offset for effective area

        self.log.info("Effective Area HDU created")
        self.edisp = energy_dispersion(
                            gammas[gammas["selected"]],
                            true_energy_bins,
                            fov_offset_bins,
                            migration_bins
                            )
        self.hdus.append(create_energy_dispersion_hdu(
                            self.edisp,
                            true_energy_bins,
                            migration_bins,
                            fov_offset_bins,
                            point_like=self.point_like,
                            extname = "ENERGY DISPERSION",
                            **extra_headers)
                            )
        self.log.info("Energy Dispersion HDU created")

        # Using the same FOV offset binning as pyirf for now.
        # Needs to be checked with standard MC files
        if self.input_proton_dl2:
            self.background = background_2d(
                            background[background["selected"]],
                            reco_energy_bins=reco_energy_bins,
                            fov_offset_bins=np.arange(0,11)*u.deg,
                            t_obs=50*u.hour
                            )
            self.hdus.append(create_background_2d_hdu(
                                self.background.T,
                                reco_energy_bins,
                                np.arange(0,11)*u.deg,
                                extname = "BACKGROUND",
                                **extra_headers)
                                )
            self.log.info("Background HDU created")
    def finish(self):
        if self.output_irf_file.exists():
            self.log.info(f"{self.output_irf_file} exists, will be overwritten")

        fits.HDUList(self.hdus).writeto(self.output_irf_file, overwrite=True)
        Provenance().add_output_file(self.output_irf_file)

def main():
    tool = IRFFITSWriter()
    tool.run()

if __name__ == "__main__":
    main()
