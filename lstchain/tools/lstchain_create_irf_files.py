"""
Create FITS file for IRFs from given MC DL2 files and selection cuts
taken either from command-line arguments or a config file.

MC gamma files can be point-like or diffuse.
IRFs can be point-like or Full Enclosure.
Background HDU maybe added if proton and electron MC are provided.

Change the selection parameters as need be using the aliases.
The default values are written in the DataSelection and DataBinning Component
and in lstchain/data/data_selection_cuts.json
The default fov_offset_bins is for single offset

Currently using spectral weighting with the spectra given in pyirf.
It has to be updated with the ones in lstchain.spectra

Usage for all 4 IRFs, argument aliases, flags and default parameter selection values:

lstchain_create_irf_files
    --fg /path/to/DL2_MC_gamma_file.h5
    --fp /path/to/DL2_MC_proton_file.h5
    --fe /path/to/DL2_MC_electron_file.h5
    --o /path/to/irf.fits.gz
    --overwrite
    --point_like (Only for point-like IRFs)
"""

import numpy as np

from ctapipe.core import Tool, traits, Provenance, ToolConfigurationError
from lstchain.io import read_mc_dl2_to_QTable
from lstchain.io import DataSelection, DataBinning

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
        default_value=False,
    ).tag(config=True)

    classes = [DataSelection, DataBinning]

    aliases = {
        ("fg", "input_gamma_dl2"): "IRFFITSWriter.input_gamma_dl2",
        ("fp", "input_proton_dl2"): "IRFFITSWriter.input_proton_dl2",
        ("fe", "input_electron_dl2"): "IRFFITSWriter.input_electron_dl2",
        ("o", "output_irf_file"): "IRFFITSWriter.output_irf_file",
        ("evt", "event_filters"): "DataSelection.event_filters",
        ("gh", "fixed_gh_cut"): "DataSelection.fixed_gh_cut",
        ("theta", "fixed_theta_cut"): "DataSelection.fixed_theta_cut",
        ("src_fov", "fixed_source_fov_offset_cut"):
            "DataSelection.fixed_source_fov_offset_cut",
        "lst_tel_ids": "DataSelection.lst_tel_ids",
        "config": "DataSelection.config",
        "overwrite": "IRFFITSWriter.overwrite",
    }

    flags = {
        "point_like": (
            {"IRFFITSWriter": {"point_like": True}},
            "Full Enclosure IRFs will be produced",
        ),
        "overwrite": (
            {"IRFFITSWriter": {"overwrite": True}},
            "overwrites output file",
        )
    }

    def setup(self):

        if self.output_irf_file.absolute().exists():
            if self.overwrite:
                self.log.warning(f"Overwriting {self.output_irf_file}")
                self.output_irf_file.unlink()
            else:
                raise ToolConfigurationError(
                    f"Output file {self.output_irf_file} already exists,"
                    " use --overwrite to overwrite"
                )

        filename = self.output_irf_file.name
        if filename.split(".")[1] != "fits":
            self.log.warning(
                f"{filename} is not a correct "
                "compressed FITS file name. It will be corrected."
            )
            filename = filename.split(".")[0] + ".fits.gz"
            self.output_irf_file = self.output_irf_file.parent / filename

        if self.input_proton_dl2 and self.input_electron_dl2 is not None:
            self.only_gamma_irf = False
        else:
            self.only_gamma_irf = True

        self.data_sel = DataSelection(parent=self)
        self.data_bin = DataBinning(parent=self)

        self.mc_particle = {
            "gamma": {
                "file": str(self.input_gamma_dl2),
                "target_spectrum": CRAB_HEGRA,
            },
        }
        Provenance().add_input_file(self.input_gamma_dl2)

        # Read and update MC information
        if not self.only_gamma_irf:
            self.mc_particle["proton"] = {
                "file": str(self.input_proton_dl2),
                "target_spectrum": IRFDOC_PROTON_SPECTRUM,
            }

            self.mc_particle["electron"] = {
                "file": str(self.input_electron_dl2),
                "target_spectrum": IRFDOC_ELECTRON_SPECTRUM,
            }

            Provenance().add_input_file(self.input_proton_dl2)
            Provenance().add_input_file(self.input_electron_dl2)

        self.provenance_log = self.output_irf_file.parent / (
            self.name + ".provenance.log"
        )

    def start(self):

        for particle_type, p in self.mc_particle.items():
            self.log.info(f"Simulated {particle_type.title()} Events:")
            p["events"], p["simulation_info"] = read_mc_dl2_to_QTable(p["file"])

            if p["simulation_info"].viewcone.value == 0.0:
                p["mc_type"] = "point-like"
            else:
                p["mc_type"] = "diffuse"
                # For diffuse gamma using Proton Spectra for calculating event weights
                if particle_type == "gamma":
                    if not self.point_like:
                        p["target_spectrum"] = IRFDOC_PROTON_SPECTRUM
                        self.log.debug(
                            "Proton spectrum used as target spectrum"
                            " for MC diffuse gamma"
                        )
                    else:
                        raise ToolConfigurationError(
                            "Diffuse MC gamma cannot be used for generating "
                            "point-like IRFs. Use appropriate MC and IRF type."
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

        self.log.info(f"Using fixed G/H cut of {self.data_sel.fixed_gh_cut}")

        gammas = self.data_sel.filter_cut(gammas)
        gammas = self.data_sel.tel_ids_filter(gammas)
        gammas = self.data_sel.gh_cut(gammas)

        # point_like = True for point like IRFs, False for Full Enclosure IRFs
        if self.point_like:
            gammas = self.data_sel.theta_cut(gammas)
            gammas = self.data_sel.true_src_fov_offset_cut(gammas)

        # Binning of parameters used in IRFs
        true_energy_bins = self.data_bin.true_energy_bins()
        reco_energy_bins = self.data_bin.reco_energy_bins()
        migration_bins = self.data_bin.energy_migration_bins()
        source_offset_bins = self.data_bin.source_offset_bins()

        if self.point_like:
            # Gammapy 0.18.2 needs offset bin centers for interpolation
            # Using just 2 'edges' like [0.2,0.6] works fine for reading the IRF but,
            # this workaround is necessary for further analysis using gammapy.
            if len(self.data_bin.fov_offset_bins()) != 3:
                self.log.critical("Offset binning is not appropriate for single offset")

        fov_offset_bins = self.data_bin.fov_offset_bins()

        if not self.only_gamma_irf:
            background = table.vstack(
                [
                    self.mc_particle["proton"]["events"],
                    self.mc_particle["electron"]["events"],
                ]
            )

            background = self.data_sel.filter_cut(background)
            background = self.data_sel.tel_ids_filter(background)
            background = self.data_sel.gh_cut(background)

            background_offset_bins = self.data_bin.bkg_fov_offset_bins()

        # For a fixed gh/theta cut, only a header value is added.
        # For energy dependent cuts, a new HDU should be created
        # GH_CUT and FOV_CUT are temporary non-standard header data
        extra_headers = {
            "TELESCOP": "CTA-N",
            "INSTRUME": "LST-" + " ".join(map(str, self.data_sel.lst_tel_ids)),
            "FOVALIGN": "RADEC",
            "GH_CUT": self.data_sel.fixed_gh_cut,
        }
        if self.point_like:
            self.log.info("Generating Point-Like IRF HDUs")
            extra_headers["RAD_MAX"] = str(
                u.Quantity(self.data_sel.fixed_theta_cut * u.deg)
            )
            extra_headers["FOV_CUT"] = str(
                u.Quantity(self.data_sel.fixed_source_fov_offset_cut * u.deg)
            )
        else:
            self.log.info("Generating Full-Enclosure IRF HDUs")

        # Write HDUs
        self.hdus = [fits.PrimaryHDU(), ]

        with np.errstate(invalid="ignore", divide="ignore"):
            if self.mc_particle["gamma"]["mc_type"] == "point-like":
                self.effective_area = effective_area_per_energy(
                    gammas,
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
                    gammas,
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

        self.log.info("Effective Area HDU created")
        self.edisp = energy_dispersion(
            gammas,
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
        self.log.info("Energy Dispersion HDU created")

        if not self.only_gamma_irf:
            self.background = background_2d(
                background,
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
            self.log.info("Background HDU created")

        if not self.point_like:
            self.psf = psf_table(
                gammas,
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
            self.log.info("PSF HDU created")

    def finish(self):

        fits.HDUList(self.hdus).writeto(self.output_irf_file, overwrite=self.overwrite)
        Provenance().add_output_file(self.output_irf_file)


def main():
    tool = IRFFITSWriter()
    tool.run()


if __name__ == "__main__":
    main()
