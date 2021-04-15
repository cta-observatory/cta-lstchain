"""
Create FITS file for IRFs from given MC DL2 files and selection cuts
taken either from command-line arguments or a config file.

MC gamma files can be point_like or diffuse.
IRFs can be point_like or Full Enclosure.
Background HDU maybe added if proton and electron MC are provided.

Change the selection parameters as need be using the aliases.
The default values are written in the EventSelector, DL3FixedCuts and
DataBinning Component and also given in some example configs in docs/examples/

To use a separate config file for providing the selection parameters,
copy and append the relevant example config files, into a custom config file.
"""

import numpy as np

from ctapipe.core import Tool, traits, Provenance, ToolConfigurationError
from lstchain.io import read_mc_dl2_to_QTable
from lstchain.io import EventSelector, DL3FixedCuts, DataBinning

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
    CRAB_MAGIC_JHEAP2015,
    IRFDOC_PROTON_SPECTRUM,
    IRFDOC_ELECTRON_SPECTRUM,
)
from pyirf.utils import calculate_source_fov_offset, calculate_theta

__all__ = ["IRFFITSWriter"]


class IRFFITSWriter(Tool):
    name = "IRFFITSWriter"
    description = __doc__
    example = """
    To generate IRFs from MC gamma only, using default cuts/binning:
    > lstchain_create_irf_files
        -g /path/to/DL2_MC_gamma_file.h5
        -o /path/to/irf.fits.gz
        --point-like (Only for point_like IRFs)
        --overwrite

    Or to generate all 4 IRFs, using default cuts/binning:
    > lstchain_create_irf_files
        -g /path/to/DL2_MC_gamma_file.h5
        -p /path/to/DL2_MC_proton_file.h5
        -e /path/to/DL2_MC_electron_file.h5
        -o /path/to/irf.fits.gz
        --point-like (Only for point_like IRFs)

    Or use a config file for cuts and binning information:
    > lstchain_create_irf_files
        -g /path/to/DL2_MC_gamma_file.h5
        -o /path/to/irf.fits.gz
        --point-like (Only for point_like IRFs)
        --config /path/to/config.json

    Or pass the selection cuts from command-line:
    > lstchain_create_irf_files
        -g /path/to/DL2_MC_gamma_file.h5
        -o /path/to/irf.fits.gz
        --point-like (Only for point_like IRFs)
        --fixed-gh-cut 0.9
        --fixed-theta-cut 0.2
        --irf-obs-time 50
    """

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
        default_value="./irf.fits.gz",
    ).tag(config=True)

    irf_obs_time = traits.Float(
        help="Observation time for IRF in hours",
        default_value=50,
    ).tag(config=True)

    point_like = traits.Bool(
        help="True for point_like IRF, False for Full Enclosure",
        default_value=False,
    ).tag(config=True)

    overwrite = traits.Bool(
        help="If True, overwrites existing output file without asking",
        default_value=False,
    ).tag(config=True)

    classes = [EventSelector, DL3FixedCuts, DataBinning]

    aliases = {
        ("g", "input-gamma-dl2"): "IRFFITSWriter.input_gamma_dl2",
        ("p", "input-proton-dl2"): "IRFFITSWriter.input_proton_dl2",
        ("e", "input-electron-dl2"): "IRFFITSWriter.input_electron_dl2",
        ("o", "output-irf-file"): "IRFFITSWriter.output_irf_file",
        "irf-obs-time": "IRFFITSWriter.irf_obs_time",
        "fixed-gh-cut": "DL3FixedCuts.fixed_gh_cut",
        "fixed-theta-cut": "DL3FixedCuts.fixed_theta_cut",
        "allowed-tels": "DL3FixedCuts.allowed_tels",
        "overwrite": "IRFFITSWriter.overwrite",
    }

    flags = {
        "point-like": (
            {"IRFFITSWriter": {"point_like": True}},
            "Point like IRFs will be produced, otherwise Full Enclosure",
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
        if not (filename.endswith('.fits') or filename.endswith('.fits.gz')):
            raise ValueError("f{filename} is not a correct compressed FITS file name (use .fits or .fits.gz).")

        if self.input_proton_dl2 and self.input_electron_dl2 is not None:
            self.only_gamma_irf = False
        else:
            self.only_gamma_irf = True

        self.event_sel = EventSelector(parent=self)
        self.fixed_cuts = DL3FixedCuts(parent=self)
        self.data_bin = DataBinning(parent=self)

        self.mc_particle = {
            "gamma": {
                "file": str(self.input_gamma_dl2),
                "target_spectrum": CRAB_MAGIC_JHEAP2015,
            },
        }
        Provenance().add_input_file(self.input_gamma_dl2)

        self.t_obs = self.irf_obs_time * u.hour

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
                p["mc_type"] = "point_like"
            else:
                p["mc_type"] = "diffuse"

            self.log.debug(f"Simulated {p['mc_type']} {particle_type.title()} Events:")

            # Calculating event weights for Background IRF
            if particle_type != "gamma":
                p["simulated_spectrum"] = PowerLaw.from_simulation(
                    p["simulation_info"], self.t_obs
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

        self.log.info(f"Using fixed G/H cut of {self.fixed_cuts.fixed_gh_cut}")

        gammas = self.event_sel.filter_cut(gammas)
        gammas = self.fixed_cuts.allowed_tels_filter(gammas)
        gammas = self.fixed_cuts.gh_cut(gammas)

        if self.point_like:
            gammas = self.fixed_cuts.theta_cut(gammas)
            self.log.info('Theta cuts applied for point like IRF')

        # Binning of parameters used in IRFs
        true_energy_bins = self.data_bin.true_energy_bins()
        reco_energy_bins = self.data_bin.reco_energy_bins()
        migration_bins = self.data_bin.energy_migration_bins()
        source_offset_bins = self.data_bin.source_offset_bins()

        if self.mc_particle["gamma"]["mc_type"] == "point_like":
            mean_fov_offset = round(gammas["true_source_fov_offset"].mean().to_value(), 1)
            fov_offset_bins = [mean_fov_offset - 0.1, mean_fov_offset + 0.1] * u.deg
            self.log.info('Single offset for point like gamma MC')
        else:
            fov_offset_bins = self.data_bin.fov_offset_bins()
            self.log.info('Multiple offset for diffuse gamma MC')

        if not self.only_gamma_irf:
            background = table.vstack(
                [
                    self.mc_particle["proton"]["events"],
                    self.mc_particle["electron"]["events"],
                ]
            )

            background = self.event_sel.filter_cut(background)
            background = self.fixed_cuts.allowed_tels_filter(background)
            background = self.fixed_cuts.gh_cut(background)

            background_offset_bins = self.data_bin.bkg_fov_offset_bins()

        # For a fixed gh/theta cut, only a header value is added.
        # For energy dependent cuts, a new HDU should be created
        # GH_CUT and FOV_CUT are temporary non-standard header data
        extra_headers = {
            "TELESCOP": "CTA-N",
            "INSTRUME": "LST-" + " ".join(map(str, self.fixed_cuts.allowed_tels)),
            "FOVALIGN": "RADEC",
            "GH_CUT": self.fixed_cuts.fixed_gh_cut,
        }
        if self.point_like:
            self.log.info("Generating point_like IRF HDUs")
            extra_headers["RAD_MAX"] = str(self.fixed_cuts.fixed_theta_cut * u.deg)
        else:
            self.log.info("Generating Full-Enclosure IRF HDUs")

        # Write HDUs
        self.hdus = [fits.PrimaryHDU(), ]

        with np.errstate(invalid="ignore", divide="ignore"):
            if self.mc_particle["gamma"]["mc_type"] == "point_like":
                self.effective_area = effective_area_per_energy(
                    gammas,
                    self.mc_particle["gamma"]["simulation_info"],
                    true_energy_bins,
                )
                self.hdus.append(
                    create_aeff2d_hdu(
                        # add one dimension for single FOV offset
                        self.effective_area[..., np.newaxis],
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
                t_obs=self.t_obs,
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
