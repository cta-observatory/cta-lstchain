"""
Create FITS file for IRFs from given MC DL2 files and selection cuts
taken either from command-line arguments or a config file.

MC gamma files can be point_like or diffuse.
IRFs can be point_like or Full Enclosure.
Background HDU maybe added if proton and electron MC are provided.

Change the selection parameters as need be using the aliases.
The default values are written in the EventSelector, DL3Cuts and
DataBinning Component and also given in some example configs in docs/examples/

By default, the Tool uses global cuts for gammaness and theta.

For using energy-dependent gammaness cuts, use the argument gh_efficiency
for passing the gamma efficiency value to calculate the gammaness cuts for
each reco energy bin and the flag energy-dependent-gh.
Similarly, for energy-dependent theta cuts, use the argument
theta_containment and the flag energy-dependent-theta.

The energy-dependent cuts are stored as HDUs - GH_CUTS and RAD_MAX,
and saved with other IRFs.

To use a separate config file for providing the selection parameters,
copy and append the relevant example config files, into a custom config file.

For source-dependent analysis, alpha cut can be used instead of theta cut.
If you want to generate source-dependent IRFs, source-dep flag should be activated.
The global alpha cut used to generate IRFs is stored as AL_CUT in the HDU header.

Modified IRFs with true energy scaled by a given factor can be created to evaluate 
the systematic uncertainty in the light collection efficiency. This can be done by 
setting a value different from one for the "scale_true_energy" argument present in 
the DataBinning Component of the configuration file of the IRF creation Tool.
(The true energy of the MC events will be scaled before filling the IRFs histograms 
when pyirf commands are used. The effects expected are a non-diagonal energy dispersion
matrix and a different spectrum).
                

"""

from astropy import table
from astropy.io import fits
from astropy.time import Time
import astropy.units as u
import numpy as np

from ctapipe.core import (
    Provenance,
    Tool,
    ToolConfigurationError,
    traits,
)

from pyirf.io.gadf import (
    create_aeff2d_hdu,
    create_background_2d_hdu,
    create_energy_dispersion_hdu,
    create_psf_table_hdu,
    create_rad_max_hdu,
)
from pyirf.irf import (
    background_2d,
    effective_area_per_energy,
    effective_area_per_energy_and_fov,
    energy_dispersion,
    psf_table,
)
from pyirf.spectral import (
    CRAB_MAGIC_JHEAP2015,
    IRFDOC_ELECTRON_SPECTRUM,
    IRFDOC_PROTON_SPECTRUM,
    PowerLaw,
    calculate_event_weights,
)
from pyirf.utils import (
    calculate_source_fov_offset,
    calculate_theta,
)

from lstchain.io import (
    DL3Cuts,
    DataBinning,
    EventSelector,
)
from lstchain.io import read_mc_dl2_to_QTable
from lstchain.io.io import check_mc_type, get_mc_fov_offset
from lstchain.__init__ import __version__

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
        --global-gh-cut 0.9
        --global-theta-cut 0.2
        --irf-obs-time 50

    Or use energy-dependent cuts based on a gamma efficiency:
    > lstchain_create_irf_files
        -g /path/to/DL2_MC_gamma_file.h5
        -o /path/to/irf.fits.gz
        --point-like (Only for point_like IRFs)
        --energy-dependent-gh
        --energy-dependent-theta
        --gh-efficiency 0.95
        --theta-containment 0.68

    Or generate source-dependent IRFs
    > lstchain_create_irf_files
        -g /path/to/DL2_MC_gamma_file.h5
        -o /path/to/irf.fits.gz
        --point-like
        --global-gh-cut 0.9
        --global-alpha-cut 10
        --source-dep

    To build modified IRFs by specifying a scaling factor applying to the true energy (without using a config file):
    > lstchain_create_irf_files
        -g /path/to/DL2_MC_gamma_file.h5
        -o /path/to/irf.fits.gz
        --scale-true-energy 1.15
        """

    input_gamma_dl2 = traits.Path(
        help="Input MC gamma DL2 file",
        default_value=None,
        allow_none=True,
        exists=True,
        directory_ok=False,
        file_ok=True
    ).tag(config=True)

    input_proton_dl2 = traits.Path(
        help="Input MC proton DL2 file",
        default_value=None,
        allow_none=True,
        exists=True,
        directory_ok=False,
        file_ok=True
    ).tag(config=True)

    input_electron_dl2 = traits.Path(
        help="Input MC electron DL2 file",
        default_value=None,
        allow_none=True,
        exists=True,
        directory_ok=False,
        file_ok=True
    ).tag(config=True)

    output_irf_file = traits.Path(
        help="IRF output file",
        allow_none=True,
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

    energy_dependent_gh = traits.Bool(
        help="True for applying energy-dependent gammaness cuts",
        default_value=False,
    ).tag(config=True)

    energy_dependent_theta = traits.Bool(
        help="True for applying energy-dependent theta cuts",
        default_value=False,
    ).tag(config=True)

    energy_dependent_alpha = traits.Bool(
        help="True for applying energy-dependent alpha cuts",
        default_value=False,
    ).tag(config=True)

    overwrite = traits.Bool(
        help="If True, overwrites existing output file without asking",
        default_value=False,
    ).tag(config=True)

    source_dep = traits.Bool(
        help="True for source-dependent analysis",
        default_value=False,
    ).tag(config=True)

    classes = [EventSelector, DL3Cuts, DataBinning]

    aliases = {
        ("g", "input-gamma-dl2"): "IRFFITSWriter.input_gamma_dl2",
        ("p", "input-proton-dl2"): "IRFFITSWriter.input_proton_dl2",
        ("e", "input-electron-dl2"): "IRFFITSWriter.input_electron_dl2",
        ("o", "output-irf-file"): "IRFFITSWriter.output_irf_file",
        "irf-obs-time": "IRFFITSWriter.irf_obs_time",
        "global-gh-cut": "DL3Cuts.global_gh_cut",
        "gh-efficiency": "DL3Cuts.gh_efficiency",
        "theta-containment": "DL3Cuts.theta_containment",
        "global-theta-cut": "DL3Cuts.global_theta_cut",
        "alpha-containment": "DL3Cuts.alpha_containment",
        "global-alpha-cut": "DL3Cuts.global_alpha_cut",
        "allowed-tels": "DL3Cuts.allowed_tels",
        "overwrite": "IRFFITSWriter.overwrite",
        "scale-true-energy": "DataBinning.scale_true_energy"
    }

    flags = {
        "point-like": (
            {"IRFFITSWriter": {"point_like": True}},
            "Point like IRFs will be produced, otherwise Full Enclosure",
        ),
        "overwrite": (
            {"IRFFITSWriter": {"overwrite": True}},
            "overwrites output file",
        ),
        "source-dep": (
            {"IRFFITSWriter": {"source_dep": True}},
            "Source-dependent analysis will be performed",
        ),
        "energy-dependent-gh": (
            {"IRFFITSWriter": {"energy_dependent_gh": True}},
            "Uses energy-dependent cuts for gammaness",
        ),
        "energy-dependent-theta": (
            {"IRFFITSWriter": {"energy_dependent_theta": True}},
            "Uses energy-dependent cuts for theta",
        ),
        "energy-dependent-alpha": (
            {"IRFFITSWriter": {"energy_dependent_alpha": True}},
            "Uses energy-dependent cuts for alpha",
        ),
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
        if not (filename.endswith(".fits") or filename.endswith(".fits.gz")):
            raise ValueError(
                f"{filename} is not a correct compressed FITS file name "
                "Use .fits or .fits.gz."
            )

        if self.input_proton_dl2 and self.input_electron_dl2 is not None:
            self.only_gamma_irf = False
        else:
            self.only_gamma_irf = True

        self.event_sel = EventSelector(parent=self)
        self.cuts = DL3Cuts(parent=self)
        self.data_bin = DataBinning(parent=self)

        self.mc_particle = {
            "gamma": {
                "file": self.input_gamma_dl2,
                "target_spectrum": CRAB_MAGIC_JHEAP2015,
            },
        }
        Provenance().add_input_file(self.input_gamma_dl2)

        self.t_obs = self.irf_obs_time * u.hour

        # Read and update MC information
        if not self.only_gamma_irf:
            self.mc_particle["proton"] = {
                "file":  self.input_proton_dl2,
                "target_spectrum": IRFDOC_PROTON_SPECTRUM,
            }

            self.mc_particle["electron"] = {
                "file": self.input_electron_dl2,
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
            (
                p["events"],
                p["simulation_info"],
                p["geomag_params"],
            ) = read_mc_dl2_to_QTable(p["file"])

            
            if self.data_bin.scale_true_energy != 1.0:
                p["events"]["true_energy"] *= self.data_bin.scale_true_energy
                p["simulation_info"].energy_min *= self.data_bin.scale_true_energy
                p["simulation_info"].energy_max *= self.data_bin.scale_true_energy

            p["mc_type"] = check_mc_type(p["file"])

            self.log.debug(
                f"Simulated {p['mc_type']} {particle_type.title()} Events:"
            )

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

            p["ZEN_PNT"] = round(
                90 - p["events"]["pointing_alt"][0].to_value(u.deg), 5
            )
            p["AZ_PNT"] = round(
                p["events"]["pointing_az"][0].to_value(u.deg), 5
            )

            if not self.source_dep:
                for prefix in ("true", "reco"):
                    k = f"{prefix}_source_fov_offset"
                    p["events"][k] = calculate_source_fov_offset(
                        p["events"], prefix=prefix
                    )

                # calculate theta / distance between reco and assumed source position
                p["events"]["theta"] = calculate_theta(
                    p["events"],
                    assumed_source_az=p["events"]["true_az"],
                    assumed_source_alt=p["events"]["true_alt"],
                )

            else:
                # Alpha cut is applied for source-dependent analysis.
                # To adapt source-dependent analysis to pyirf codes,
                # true position is set as reco position for survived events
                # after alpha cut
                p["events"]["true_source_fov_offset"] = calculate_source_fov_offset(
                    p["events"], prefix="true"
                )
                p["events"]["reco_source_fov_offset"] = p["events"]["true_source_fov_offset"]

        self.log.debug(p["simulation_info"])
        gammas = self.mc_particle["gamma"]["events"]
        geomag_params = self.mc_particle["gamma"]["geomag_params"]
        self.log.info(geomag_params)

        # Binning of parameters used in IRFs
        true_energy_bins = self.data_bin.true_energy_bins()
        reco_energy_bins = self.data_bin.reco_energy_bins()
        migration_bins = self.data_bin.energy_migration_bins()
        source_offset_bins = self.data_bin.source_offset_bins()

        if self.mc_particle["gamma"]["mc_type"] in ["point_like", "ring_wobble"]:
            # The 4 is semi-arbitray. This keeps the same precision as the previous code
            mean_fov_offset = np.round(get_mc_fov_offset(self.mc_particle["gamma"]["file"]), 4)
            fov_offset_bins = [mean_fov_offset - 0.1, mean_fov_offset + 0.1] * u.deg
            self.log.info(f"Single offset for point like gamma MC with offset {mean_fov_offset}")
        else:
            fov_offset_bins = self.data_bin.fov_offset_bins()
            self.log.info(f"Multiple offset for diffuse gamma MC : {fov_offset_bins}")

            if np.max(fov_offset_bins) > gammas["true_source_fov_offset"].max():
                self.log.warning(f'The highest FoV offset bin ({np.max(fov_offset_bins)}) is larger than the maximum offset simulated ({gammas["true_source_fov_offset"].max()})')

        gammas = self.event_sel.filter_cut(gammas)
        gammas = self.cuts.allowed_tels_filter(gammas)
        gammas = gammas[gammas['true_source_fov_offset'] <= np.max(fov_offset_bins)]

        if self.energy_dependent_gh:
            self.gh_cuts_gamma = self.cuts.energy_dependent_gh_cuts(
                gammas, reco_energy_bins
            )
            gammas = self.cuts.apply_energy_dependent_gh_cuts(
                gammas, self.gh_cuts_gamma
            )
            self.log.info(
                f"Using gamma efficiency of {self.cuts.gh_efficiency}"
            )
        else:
            gammas = self.cuts.apply_global_gh_cut(gammas)
            self.log.info(
                "Using a global gammaness cut of "
                f"{self.cuts.global_gh_cut}"
            )

        if self.point_like:
            if not self.source_dep:
                if self.energy_dependent_theta:
                    self.theta_cuts = self.cuts.energy_dependent_theta_cuts(
                        gammas, reco_energy_bins,
                    )
                    gammas = self.cuts.apply_energy_dependent_theta_cuts(
                        gammas, self.theta_cuts
                    )
                    self.log.info(
                        "Using a containment region for theta of "
                        f"{self.cuts.theta_containment}"
                    )
                else:
                    gammas = self.cuts.apply_global_theta_cut(gammas)
                    self.log.info(
                        "Using a global Theta cut of "
                        f"{self.cuts.global_theta_cut} for point-like IRF"
                    )
            else:
                if self.energy_dependent_alpha:
                    self.alpha_cuts = self.cuts.energy_dependent_alpha_cuts(
                        gammas, reco_energy_bins,
                    )
                    gammas = self.cuts.apply_energy_dependent_alpha_cuts(
                        gammas, self.alpha_cuts
                    )
                    self.log.info(
                        "Using a containment region for alpha of "
                        f"{self.cuts.alpha_containment} %"
                    )
                else:
                    gammas = self.cuts.apply_global_alpha_cut(gammas)
                    self.log.info(
                        "Using a global Alpha cut of "
                        f"{self.cuts.global_alpha_cut} for point like IRF"
                    )

        if not self.only_gamma_irf:
            background = table.vstack(
                [
                    self.mc_particle["proton"]["events"],
                    self.mc_particle["electron"]["events"],
                ]
            )

            # Check common parameters of the MCs used
            for par in ["ZEN_PNT", "AZ_PNT"]:
                k = [
                    self.mc_particle["gamma"][par],
                    self.mc_particle["proton"][par],
                    self.mc_particle["electron"][par],
                ]
                if len(set(k)) != 1:
                    raise ToolConfigurationError(
                        "MCs of different " + par + " used."
                        "Use MCs with same zenith pointing."
                    )

            if self.energy_dependent_gh:
                background = self.cuts.apply_energy_dependent_gh_cuts(
                    background, self.gh_cuts_gamma
                )
            else:
                background = self.cuts.apply_global_gh_cut(background)

            background = self.event_sel.filter_cut(background)
            background = self.cuts.allowed_tels_filter(background)

            background_offset_bins = self.data_bin.bkg_fov_offset_bins()

        # For a global gh/theta cut, only a header value is added.
        # For energy-dependent cuts, along with GADF specified RAD_MAX HDU,
        # a new HDU is created, GH_CUTS which is based on RAD_MAX table

        # NOTE: The GH_CUTS HDU is just for provenance and is not supported
        # by GADF or used by any Science Tools
        extra_headers = {
            "TELESCOP": "CTA-N",
            "INSTRUME": "LST-" + " ".join(map(str, self.cuts.allowed_tels)),
            "FOVALIGN": "RADEC",
        }

        extra_headers["ZEN_PNT"] = (
            self.mc_particle["gamma"]["ZEN_PNT"],
            "deg"
        )
        extra_headers["AZ_PNT"] = (
            self.mc_particle["gamma"]["AZ_PNT"],
            "deg"
        )
        extra_headers["B_TOTAL"] = (
            geomag_params["GEOMAG_TOTAL"].to_value(u.uT),
            "uT",
        )
        extra_headers["B_INC"] = (
            geomag_params["GEOMAG_INC"].to_value(u.rad),
            "rad",
        )
        extra_headers["B_DEC"] = (
            geomag_params["GEOMAG_DEC"].to_value(u.rad),
            "rad",
        )
        extra_headers["B_DELTA"] = (
            geomag_params["GEOMAG_DELTA"].to_value(u.deg),
            "deg",
        )
        # To avoid an astropy warning, we use HIERARCH cards for keywords longer
        # than eight characters. Later, they can be accessed like any other
        # (e.g. hdu.header['ETRUE_SCALE']).
        extra_headers["HIERARCH ETRUE_SCALE"]= (
            self.data_bin.scale_true_energy
        )
      
        if self.point_like:
            self.log.info("Generating point_like IRF HDUs")
        else:
            self.log.info("Generating Full-Enclosure IRF HDUs")

        # Updating the HDU headers with the gammaness and theta cuts/efficiency
        if not self.energy_dependent_gh:
            extra_headers["GH_CUT"] = self.cuts.global_gh_cut

        else:
            extra_headers["GH_EFF"] = (
                self.cuts.gh_efficiency,
                "gamma/hadron efficiency",
            )

        if self.point_like:
            if not self.source_dep:
                if self.energy_dependent_theta:
                    extra_headers["TH_CONT"] = (
                        self.cuts.theta_containment,
                        "Theta containment region in percentage",
                    )
                else:
                    extra_headers["RAD_MAX"] = (
                        self.cuts.global_theta_cut,
                        'deg'
                    )
            else:
                # add dummy "RAD_MAX" to adapt to 1D analysis with gammapy>0.20.1
                extra_headers["RAD_MAX"] = (
                    0.1,
                    'deg'
                )

                if self.energy_dependent_alpha:
                    extra_headers["AL_CONT"] = (
                        self.cuts.alpha_containment,
                        "Alpha containment region in percentage",
                    )
                else:
                    extra_headers["AL_CUT"] = (
                        self.cuts.global_alpha_cut,
                        'deg'
                    )

        # Write HDUs
        self.hdus = [fits.PrimaryHDU(), ]

        with np.errstate(invalid="ignore", divide="ignore"):
            if self.mc_particle["gamma"]["mc_type"] in ["point_like", "ring_wobble"]:
                self.effective_area = effective_area_per_energy(
                    gammas,
                    self.mc_particle["gamma"]["simulation_info"],
                    true_energy_bins=true_energy_bins,
                )
                self.effective_area = np.nan_to_num(self.effective_area)  # To be added in pyirf
                self.hdus.append(
                    create_aeff2d_hdu(
                        # add one dimension for single FOV offset
                        effective_area=self.effective_area[..., np.newaxis],
                        true_energy_bins=true_energy_bins,
                        fov_offset_bins=fov_offset_bins,
                        point_like=self.point_like,
                        extname="EFFECTIVE AREA",
                        **extra_headers,
                    )
                )
            else:
                self.effective_area = effective_area_per_energy_and_fov(
                    gammas,
                    self.mc_particle["gamma"]["simulation_info"],
                    true_energy_bins=true_energy_bins,
                    fov_offset_bins=fov_offset_bins,
                )
                self.effective_area = np.nan_to_num(self.effective_area)
                self.hdus.append(
                    create_aeff2d_hdu(
                        effective_area=self.effective_area,
                        true_energy_bins=true_energy_bins,
                        fov_offset_bins=fov_offset_bins,
                        point_like=self.point_like,
                        extname="EFFECTIVE AREA",
                        **extra_headers,
                    )
                )

        self.log.info("Effective Area HDU created")
        self.edisp = energy_dispersion(
            gammas,
            true_energy_bins=true_energy_bins,
            fov_offset_bins=fov_offset_bins,
            migration_bins=migration_bins,
        )
        self.hdus.append(
            create_energy_dispersion_hdu(
                energy_dispersion=self.edisp,
                true_energy_bins=true_energy_bins,
                migration_bins=migration_bins,
                fov_offset_bins=fov_offset_bins,
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
                    background_2d=self.background.T,
                    reco_energy_bins=reco_energy_bins,
                    fov_offset_bins=background_offset_bins,
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
                    psf=self.psf,
                    true_energy_bins=true_energy_bins,
                    source_offset_bins=source_offset_bins,
                    fov_offset_bins=fov_offset_bins,
                    extname="PSF",
                    **extra_headers,
                )
            )
            self.log.info("PSF HDU created")

        if self.energy_dependent_gh:
            # Create a separate temporary header
            gh_header = fits.Header()
            gh_header["CREATOR"] = f"lstchain v{__version__}"
            gh_header["DATE"] = Time.now().utc.iso

            for k, v in extra_headers.items():
                gh_header[k] = v

            self.hdus.append(
                fits.BinTableHDU(
                    self.gh_cuts_gamma, header=gh_header, name="GH_CUTS"
                )
            )
            self.log.info("GH CUTS HDU added")

        if self.energy_dependent_theta and self.point_like:
            if not self.source_dep:
                self.hdus.append(
                    create_rad_max_hdu(
                        self.theta_cuts["cut"][:, np.newaxis],
                        reco_energy_bins, fov_offset_bins[[fov_offset_bins.argmin(),
                                                           fov_offset_bins.argmax()]],
                        **extra_headers
                    )
                )
                self.log.info("RAD MAX HDU added")

        if self.energy_dependent_alpha and self.source_dep:
            # Create a separate temporary header
            alpha_header = fits.Header()
            alpha_header["CREATOR"] = f"lstchain v{__version__}"
            alpha_header["DATE"] = Time.now().utc.iso

            for k, v in extra_headers.items():
                alpha_header[k] = v

            self.hdus.append(
                fits.BinTableHDU(
                    self.alpha_cuts, header=gh_header, name="AL_CUTS"
                )
            )
            self.log.info("ALPHA CUTS HDU added")

    def finish(self):

        fits.HDUList(self.hdus).writeto(
            self.output_irf_file, overwrite=self.overwrite
        )
        Provenance().add_output_file(self.output_irf_file)


def main():
    tool = IRFFITSWriter()
    tool.run()


if __name__ == "__main__":
    main()
