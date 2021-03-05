"""
Create FITS file for IRFs from given MC DL2 files and selection cuts

MC gamma files can be point-like or diffuse
IRFs can be point-like or Full Enclosure
Background HDU maybe added if proton and electron MC are provided

Currently using spectral weighting with the spectra given in pyirf.
It has to be updated with the ones in lstchain.spectra

Usage for all 4 IRFs, argument aliases and standard config file for selection cuts:

lstchain_create_irf_files
    --fg /path/to/DL2_MC_gamma_file.h5
    --fp /path/to/DL2_MC_proton_file.h5
    --fe /path/to/DL2_MC_electron_file.h5
    --o /path/to/irf.fits.gz
    --pnt False
    --config /path/to/cta-lstchain/lstchain/data/data_selection_cuts.json
"""

import numpy as np

from ctapipe.core import Tool, traits, Provenance, ToolConfigurationError
from lstchain.io import read_mc_dl2_to_pyirf, read_configuration_file
from lstchain.reco.utils import filter_events

from astropy.io import fits
import astropy.units as u
from astropy import table

from pyirf.io.gadf import (
    create_aeff2d_hdu,
    create_energy_dispersion_hdu,
    create_background_2d_hdu,
    create_psf_table_hdu,
)
from pyirf.irf import (
    effective_area_per_energy,
    energy_dispersion,
    effective_area_per_energy_and_fov,
    background_2d,
    psf_table,
)
from pyirf.spectral import (
    calculate_event_weights,
    PowerLaw,
    CRAB_HEGRA,
    IRFDOC_PROTON_SPECTRUM,
    IRFDOC_ELECTRON_SPECTRUM,
)
from pyirf.utils import calculate_source_fov_offset, calculate_theta
from pyirf.binning import create_bins_per_decade  # , add_overflow_bins

__all__ = ["IRFFITSWriter"]


class IRFFITSWriter(Tool):
    name = "IRFFITSWriter"
    description = __doc__

    input_gamma_dl2 = traits.Path(
        help="Input MC gamma DL2 file", exists=True, directory_ok=False, file_ok=True
    ).tag(config=True)

    input_proton_dl2 = traits.Path(
        help="Input MC proton DL2 file", exists=True, directory_ok=False, file_ok=True
    ).tag(config=True)

    input_electron_dl2 = traits.Path(
        help="Input MC electron DL2 file", exists=True, directory_ok=False, file_ok=True
    ).tag(config=True)

    output_irf_file = traits.Path(
        help="IRF output file",
        directory_ok=False,
        file_ok=True,
        default_value="./irf.fits.gz",
    ).tag(config=True)

    point_like = traits.Bool(
        help="True for point-like IRF, False for Full Enclosure",
        default_value=False,
    ).tag(config=True)

    overwrite = traits.Bool(
        help="If True, overwrites existing output file without asking",
        default_value=True,
    ).tag(config=True)


    aliases = {
        ("fg", "input_gamma_dl2"): "IRFFITSWriter.input_gamma_dl2",
        ("fp", "input_proton_dl2"): "IRFFITSWriter.input_proton_dl2",
        ("fe", "input_electron_dl2"): "IRFFITSWriter.input_electron_dl2",
        ("o", "output_irf_file"): "IRFFITSWriter.output_irf_file",
        ("pnt", "point_like"): "IRFFITSWriter.point_like",
    }

    flag = {
        "point_like": (
            {"IRFFITSWriter": {"point_like": False}},
            "Full Enclosure IRFs will be produced",
        ),
        "overwrite": ({"IRFFITSWriter": {"overwrite": True}}, "overwrite output file"),
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setup(self):
        filename = self.output_irf_file.name
        if filename.split(".")[1:] != ["fits", "gz"]:
            self.log.debug(
                f"{filename} is not a correct "
                "compressed FITS file name. It will be corrected."
            )
            filename = filename.split(".")[0] + ".fits.gz"
            self.output_irf_file = self.output_irf_file.parent / filename

        self.cuts = read_configuration_file(self.config_file)

        if self.output_irf_file.exists() and not self.overwrite:
            raise ToolConfigurationError(
                f"Output file {self.output_irf_file} already exists,"
                " use --overwrite to overwrite"
            )

        if self.input_proton_dl2 and self.input_electron_dl2 is not None:
            self.only_gamma_irf = False
        else:
            self.only_gamma_irf = True

        # Read and update MC information
        if self.only_gamma_irf:
            self.mc_particle = {
                "gamma": {
                    "file": str(self.input_gamma_dl2),
                    "target_spectrum": CRAB_HEGRA,
                },
            }
            Provenance().add_input_file(self.input_gamma_dl2)
            self.log.info("Only gamma MC used to generate IRFs")

        else:
            self.mc_particle = {
                "gamma": {
                    "file": str(self.input_gamma_dl2),
                    "target_spectrum": CRAB_HEGRA,
                },
                "proton": {
                    "file": str(self.input_proton_dl2),
                    "target_spectrum": IRFDOC_PROTON_SPECTRUM,
                },
                "electron": {
                    "file": str(self.input_electron_dl2),
                    "target_spectrum": IRFDOC_ELECTRON_SPECTRUM,
                },
            }
            Provenance().add_input_file(self.input_gamma_dl2)
            Provenance().add_input_file(self.input_proton_dl2)
            Provenance().add_input_file(self.input_electron_dl2)
            self.log.info("All particles MC used to produce IRFs")

        self.provenance_log = self.output_irf_file.parent / (
                self.name + ".provenance.log"
            )

    def start(self):

        for particle_type, p in self.mc_particle.items():
            self.log.info(f"Simulated {particle_type.title()} Events:")
            p["events"], p["simulation_info"] = read_mc_dl2_to_pyirf(p["file"])

            if p["simulation_info"].viewcone.value == 0.0:
                p["mc_type"] = "point-like"
            else:
                p["mc_type"] = "diffuse"
                # For diffuse gamma using Proton Spectra for calculating event weights
                if particle_type == "gamma":
                    p["target_spectrum"] = IRFDOC_PROTON_SPECTRUM
                    self.log.debug(
                        "Proton spectrum used as target spectrum"
                        " for MC diffuse gamma"
                    )

            self.log.debug(f"Simulated {p['mc_type']} {particle_type.title()} Events:")

            p["simulated_spectrum"] = PowerLaw.from_simulation(
                p["simulation_info"], 50 * u.hour
            )
            p["events"]["weight"] = calculate_event_weights(
                p["events"]["true_energy"],
                p["target_spectrum"],
                p["simulated_spectrum"],
            )

            for prefix in ("true", "reco"):
                k = f"{prefix}_source_fov_offset"
                p["events"][k] = calculate_source_fov_offset(p["events"], prefix=prefix)
            # calculate theta / distance between reco and assumed source position
            p["events"]["theta"] = calculate_theta(
                p["events"],
                assumed_source_az=p["events"]["true_az"],
                assumed_source_alt=p["events"]["true_alt"],
            )
            self.log.debug(p["simulation_info"])

        gammas = self.mc_particle["gamma"]["events"]

        gh_cut = self.cuts["fixed_cuts"]["gh_score"][0]
        self.log.debug(f"Using fixed G/H cut of {gh_cut} to calculate theta cuts")

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
                **self.cuts["fixed_cuts"]["theta_cut"]
            )
            gammas["selected_fov"] = gammas["true_source_fov_offset"] < u.Quantity(
                **self.cuts["fixed_cuts"]["source_fov_offset"]
            )
            # Combining selection cuts
            gammas["selected"] = (
                gammas["selected_theta"]
                & gammas["selected_gh"]
                & gammas["selected_fov"]
                & gammas["selected_tels"]
            )
        else:
            gammas["selected"] = gammas["selected_gh"] & gammas["selected_tels"]

        # Binning of parameters used in IRFs
        # Energy bins
        en_bins = self.cuts["energy_bins"]
        true_energy_bins = create_bins_per_decade(
            en_bins["true"][0] * u.TeV, en_bins["true"][1] * u.TeV, en_bins["true"][2]
        )
        # add_overflow_bins(***)[1:-1]
        # The overflow binning added is not needed in the current script
        reco_energy_bins = create_bins_per_decade(
            en_bins["reco"][0] * u.TeV, en_bins["reco"][1] * u.TeV, en_bins["reco"][2]
        )
        migration_bins = np.linspace(
            en_bins["migration"][0], en_bins["migration"][1], en_bins["migration"][2]
        )
        # Using the same binning as in pyirf for source_offset_bins and
        # FoV offset bins for background IRF
        # Angular bins
        ang_bins = self.cuts["angular_bins"]
        source_offset_bins = (
            np.arange(
                ang_bins["source_offset"][0],
                ang_bins["source_offset"][1],
                ang_bins["source_offset"][2],
            )
            * u.deg
        )

        if self.mc_particle["gamma"]["mc_type"] == "point-like":
            # Gammapy 0.18.2 needs offset bin centers for interpolation
            # Using just 2 'edges' like [0.2,0.6] works fine for reading the IRF but,
            # this workaround is necessary for further analysis using gammapy.
            fov_offset_bins = ang_bins["single_fov_offset"] * u.deg
        else:
            fov_offset_bins = ang_bins["multiple_fov_offset"] * u.deg

        if not self.only_gamma_irf:
            background = table.vstack(
                [
                    self.mc_particle["proton"]["events"],
                    self.mc_particle["electron"]["events"],
                ]
            )

            background = filter_events(background, self.cuts["events_filters"])
            background["selected_gh"] = background["gh_score"] > gh_cut
            for i in tel_ids:
                background["selected_tels"] = background["tel_id"] == i
            background["selected"] = background["selected_gh"] & background["selected_tels"]

            background_offset_bins = (
                np.arange(ang_bins["source_offset"][0], ang_bins["source_offset"][1])
                * u.deg
            )

        # For a fixed gh/theta cut, only a header value is added.
        # For energy dependent cuts, a new HDU should be created
        # GH_CUT and FOV_CUT are temporary non-standard header data
        extra_headers = {
            "TELESCOP": "CTA-N",
            "INSTRUME": "LST-" + " ".join(map(str, tel_ids)),
            "FOVALIGN": "RADEC",
            "GH_CUT": gh_cut,
        }
        if self.point_like:
            self.log.debug("Generating Point-Like IRF HDUs")
            extra_headers["RAD_MAX"] = str(
                u.Quantity(**self.cuts["fixed_cuts"]["theta_cut"])
            )
            extra_headers["FOV_CUT"] = str(
                u.Quantity(**self.cuts["fixed_cuts"]["source_fov_offset"])
            )
        else:
            self.log.debug("Generating Full-Enclosure IRF HDUs")

        # Write HDUs
        self.hdus = [
            fits.PrimaryHDU(),
        ]

        with np.errstate(invalid="ignore", divide="ignore"):
            if self.mc_particle["gamma"]["mc_type"] == "point-like":
                self.effective_area = effective_area_per_energy(
                    gammas[gammas["selected"]],
                    self.mc_particle["gamma"]["simulation_info"],
                    true_energy_bins,
                )
                # As mentioned above, gammapy 0.18.2 needs offset bin center Values
                # for doing more than just reading the IRF.The effective area for
                # point-like IRF with single offset (0.4 deg) needs to be
                # reshaped and repeat the same values for the area in the second axis
                self.hdus.append(
                    create_aeff2d_hdu(
                        np.repeat(self.effective_area[..., np.newaxis], 2, axis=1),
                        true_energy_bins,
                        fov_offset_bins,
                        point_like=self.point_like,
                        extname="EFFECTIVE AREA",
                        **extra_headers,
                    )
                )
            else:
                self.effective_area = effective_area_per_energy_and_fov(
                    gammas[gammas["selected"]],
                    self.mc_particle["gamma"]["simulation_info"],
                    true_energy_bins,
                    fov_offset_bins,
                )
                self.hdus.append(
                    create_aeff2d_hdu(
                        self.effective_area,
                        true_energy_bins,
                        fov_offset_bins,
                        point_like=self.point_like,
                        extname="EFFECTIVE AREA",
                        **extra_headers,
                    )
                )

        self.log.debug("Effective Area HDU created")
        self.edisp = energy_dispersion(
            gammas[gammas["selected"]],
            true_energy_bins,
            fov_offset_bins,
            migration_bins,
        )
        self.hdus.append(
            create_energy_dispersion_hdu(
                self.edisp,
                true_energy_bins,
                migration_bins,
                fov_offset_bins,
                point_like=self.point_like,
                extname="ENERGY DISPERSION",
                **extra_headers,
            )
        )
        self.log.debug("Energy Dispersion HDU created")

        if not self.only_gamma_irf:
            self.background = background_2d(
                background[background["selected"]],
                reco_energy_bins=reco_energy_bins,
                fov_offset_bins=background_offset_bins,
                t_obs=50 * u.hour,
            )
            self.hdus.append(
                create_background_2d_hdu(
                    self.background.T,
                    reco_energy_bins,
                    background_offset_bins,
                    extname="BACKGROUND",
                    **extra_headers,
                )
            )
            self.log.debug("Background HDU created")

        if not self.point_like:
            self.psf = psf_table(
                gammas[gammas["selected_gh"] & gammas["selected_tels"]],
                true_energy_bins,
                fov_offset_bins=fov_offset_bins,
                source_offset_bins=source_offset_bins,
            )
            self.hdus.append(
                create_psf_table_hdu(
                    self.psf,
                    true_energy_bins,
                    source_offset_bins,
                    fov_offset_bins,
                    extname="PSF",
                    **extra_headers,
                )
            )
            self.log.debug("PSF HDU created")

    def finish(self):

        fits.HDUList(self.hdus).writeto(self.output_irf_file, overwrite=self.overwrite)
        Provenance().add_output_file(self.output_irf_file)


def main():
    tool = IRFFITSWriter()
    tool.run()


if __name__ == "__main__":
    main()
