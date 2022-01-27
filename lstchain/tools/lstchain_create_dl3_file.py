"""
Create DL3 FITS file from given data DL2 file,
selection cuts and/or IRF FITS files.

For an interpolated IRF, based on the data provided by the event list,
provide multiple IRFs. For that provide the common path to the IRFs,
glob search pattern for the IRFs and a final interpolated IRF file name.

Change the selection parameters as need be using the aliases.
The default values are written in the EventSelector and DL3FixedCuts Component
and also given in some example configs in docs/examples/

To use a separate config file for providing the selection parameters,
copy and append the relevant example config files, into a custom config file.
"""

from astropy.coordinates import SkyCoord
from astropy.io import fits
from ctapipe.core import (
    Provenance,
    Tool,
    ToolConfigurationError,
    traits,
)

from ctapipe.core import Tool, traits, Provenance, ToolConfigurationError
from lstchain.io import read_data_dl2_to_QTable
from lstchain.reco.utils import get_effective_time
from lstchain.paths import run_info_from_filename, dl2_to_dl3_filename
from lstchain.irf.hdu_table import create_event_list, add_icrs_position_params
from lstchain.irf.interpolate import (
    check_in_delaunay_triangle, compare_irfs, interpolate_irf
)
from lstchain.io import EventSelector, DL3FixedCuts
from lstchain.io import read_data_dl2_to_QTable
from lstchain.irf import (
    add_icrs_position_params,
    create_event_list,
)
from lstchain.paths import (
    dl2_to_dl3_filename,
    run_info_from_filename,
)
from lstchain.reco.utils import get_effective_time

__all__ = ["DataReductionFITSWriter"]


class DataReductionFITSWriter(Tool):
    name = "DataReductionFITSWriter"
    description = __doc__
    example = """
    To generate DL3 file from an observed data DL2 file, using default cuts:
    > lstchain_create_dl3_file
        -d /path/to/DL2_data_file.h5
        -o /path/to/DL3/file/
        -i /path/to/irf/
        --source-name Crab
        --source-ra 83.633deg
        --source-dec 22.01deg

    Or use a config file for the cuts:
    > lstchain_create_dl3_file
        -d /path/to/DL2_data_file.h5
        -o /path/to/DL3/file/
        -i /path/to/irf/
        --source-name Crab
        --source-ra 83.633deg
        --source-dec 22.01deg
        --overwrite
        --config /path/to/config.json

    Or pass the selection cuts from command-line:
    > lstchain_create_dl3_file
        -d /path/to/DL2_data_file.h5
        -o /path/to/DL3/file/
        --input-irf-path /path/to/irf/
        --irf-file-pattern irf*.fits.gz
        --final-irf-file final_interp_irf.fits.gz
        --source-name Crab
        --source-ra 83.633deg
        --source-dec 22.01deg
        --fixed-gh-cut 0.9
        --fixed-theta-cut 0.2
        --overwrite
    Or use a list of IRFs for including interpolated IRF:
    > lstchain_create_dl3_file
        -d /path/to/DL2_data_file.h5
        -o /path/to/DL3/file/
        -i /path/to/irf/
        -p irf*.fits.gz
        -f final_interp_irf.fits.gz
        --source-name Crab
        --source-ra 83.633deg
        --source-dec 22.01deg
        --overwrite
        --config /path/to/config.json
    """

    input_dl2 = traits.Path(
        help="Input data DL2 file",
        exists=True,
        directory_ok=False,
        file_ok=True,
    ).tag(config=True)

    output_dl3_path = traits.Path(
        help="DL3 output filedir",
        directory_ok=True,
        file_ok=False,
    ).tag(config=True)

    input_irf_path = traits.Path(
        help="Path for compressed FITS file of IRFs",
        exists=True,
        directory_ok=True,
        file_ok=False,
    ).tag(config=True)

    irf_file_pattern = traits.Unicode(
        help="IRF file pattern to search in the given IRF files path",
        default_value="*irf*.fits.gz",
    ).tag(config=True)

    final_irf_file = traits.Path(
        help="Final IRF file included with DL3 file",
        directory_ok=False,
        file_ok=True,
        default_value="final_irf.fits.gz",
    ).tag(config=True)

    source_name = traits.Unicode(
        help="Name of Source",
    ).tag(config=True)

    source_ra = traits.Unicode(
        help="RA position of the source",
    ).tag(config=True)

    source_dec = traits.Unicode(
        help="DEC position of the source",
    ).tag(config=True)

    interp_method = traits.Unicode(
        help="Interpolation method to be used, when required",
        default_value="linear",
    ).tag(config=True)

    overwrite = traits.Bool(
        help="If True, overwrites existing output file without asking",
        default_value=False,
    ).tag(config=True)

    classes = [EventSelector, DL3FixedCuts]

    aliases = {
        ("d", "input-dl2"): "DataReductionFITSWriter.input_dl2",
        ("o", "output-dl3-path"): "DataReductionFITSWriter.output_dl3_path",
        ("i", "input-irf-path"): "DataReductionFITSWriter.input_irf_path",
        ("p", "irf-file-pattern"): "DataReductionFITSWriter.irf_file_pattern",
        ("f", "final-irf-file"): "DataReductionFITSWriter.final_irf_file",
        "interp-method": "DataReductionFITSWriter.interp_method",
        "fixed-gh-cut": "DL3FixedCuts.fixed_gh_cut",
        "fixed-theta-cut": "DL3FixedCuts.fixed_theta_cut",
        "source-name": "DataReductionFITSWriter.source_name",
        "source-ra": "DataReductionFITSWriter.source_ra",
        "source-dec": "DataReductionFITSWriter.source_dec",
    }

    flags = {
        "overwrite": (
            {"DataReductionFITSWriter": {"overwrite": True}},
            "overwrite output file if True",
        ),
    }

    def setup(self):

        self.filename_dl3 = dl2_to_dl3_filename(self.input_dl2)
        self.provenance_log = self.output_dl3_path / (self.name + ".provenance.log")

        Provenance().add_input_file(self.input_dl2)

        self.event_sel = EventSelector(parent=self)
        self.fixed_cuts = DL3FixedCuts(parent=self)

        self.output_file = self.output_dl3_path.absolute() / self.filename_dl3
        if self.output_file.exists():
            if self.overwrite:
                self.log.warning(f"Overwriting {self.output_file}")
                self.output_file.unlink()
            else:
                raise ToolConfigurationError(
                    f"Output file {self.output_file} already exists,"
                    " use --overwrite to overwrite"
                )

        if self.input_irf_path:
            self.irf_list = sorted(
                self.input_irf_path.glob(self.irf_file_pattern)
            )
            if self.irf_list == []:
                self.log.critical(
                    f"No IRF files found with pattern {self.irf_file_pattern}"
                )

        if self.input_irf_path:
            if len(self.irf_list) > 1:
                # self.log.info(self.irf_list)
                # Compare the IRFs for its metadata and cuts
                if not compare_irfs(self.irf_list):
                    raise ToolConfigurationError(
                        f"IRF files in {self.input_irf_path} with pattern, "
                        f"{self.irf_file_pattern} are not similar and cannot"
                        " be used to interpolate. Use different list of IRFs."
                    )

            elif len(self.irf_list) == 1:
                self.log.info(
                    f"Only single IRF {self.irf_list[0]} provided."
                    " No interpolation possible"
                )
            else:
                raise ToolConfigurationError(
                    f"No IRF files in {self.input_irf_path} with pattern, "
                    f"{self.irf_file_pattern} found. Use different parameters"
                )

        self.final_irf_output = self.output_dl3_path.absolute() / str(self.final_irf_file.name)

        if self.final_irf_output.exists():
            if self.overwrite:
                self.log.warning(f"Overwriting {self.final_irf_output}")
                self.final_irf_output.unlink()
            else:
                raise ToolConfigurationError(
                    f"Final IRF file {self.final_irf_output} already exists,"
                    " use --overwrite to overwrite"
                )

        if not (self.source_ra or self.source_dec):
            self.source_pos = SkyCoord.from_name(self.source_name)
        elif bool(self.source_ra) != bool(self.source_dec):
            raise ToolConfigurationError(
                "Either provide both RA and DEC values for the source or none"
            )
        else:
            self.source_pos = SkyCoord(ra=self.source_ra, dec=self.source_dec)

        self.log.debug(f"Output DL3 file: {self.output_file}")

    def start(self):

        self.data = read_data_dl2_to_QTable(str(self.input_dl2))
        ## To reduce the table columns further, add a selection of columns to be read and used.
        self.effective_time, self.elapsed_time = get_effective_time(self.data)
        self.run_number = run_info_from_filename(self.input_dl2)[1]

        self.data = self.event_sel.filter_cut(self.data)
        self.data = self.fixed_cuts.gh_cut(self.data)
        self.data = add_icrs_position_params(self.data, self.source_pos)

        self.log.info("Generating event list")
        self.events, self.gti, self.pointing, self.data_params = create_event_list(
            data=self.data,
            run_number=self.run_number,
            source_name=self.source_name,
            source_pos=self.source_pos,
            effective_time=self.effective_time.value,
            elapsed_time=self.elapsed_time.value,
        )
        self.log.info(f"Target parameters for interpolation: {self.data_params}")

        self.hdulist = fits.HDUList(
            [fits.PrimaryHDU(), self.events, self.gti, self.pointing]
        )

        if self.input_irf_path:
            if len(self.irf_list) > 1:
                # Get the optimal number of IRFs necessary for interpolation

                # IF the target parameter is not inside a simplex formed by
                # the given list of IRFs, use a point closest to the nearest
                # simplex, ie, perpendicular distance from the target to a
                # point on the closest simplex side.
                self.data_params_new, self.irf_list = check_in_delaunay_triangle(
                    self.irf_list, self.data_params
                )

                self.log.info(f"Paths of Irfs used for interpolation {self.irf_list}")
                if self.data_params_new != self.data_params:
                    self.log.info(
                        "Updated target parameters for interpolation:"
                        f" {self.data_params_new}"
                    )

            if len(self.irf_list) > 1:
                self.irf_final_hdu = interpolate_irf(
                    self.irf_list, self.data_params_new, self.interp_method
                )
                self.irf_final_hdu.writeto(
                    self.final_irf_output, overwrite=self.overwrite
                )
                self.irf_final_hdu = fits.open(self.final_irf_output)

                self.log.info("Adding IRF HDUs")
                self.mc_params = dict()

                h = self.irf_final_hdu[1].header
                for p in self.data_params_new.keys():
                    self.mc_params[p] = u.Quantity(h[p]).to(u.deg)
                mc_gamma_offset = u.Quantity(h["G_OFFSET"]).to(u.deg)

                self.log.info(f"Gamma offset for MC is {mc_gamma_offset:.2f}")
                self.log.info(
                    f"Zenith pointing of MC at {self.mc_params['ZEN_PNT']:.2f}"
                )
                self.log.info(
                    f"Azimuth pointing of MC at {self.mc_params['AZ_PNT']:.2f}"
                )
                self.log.info(
                    f"Geomagnetic delta for the MC is {self.mc_params['B_DELTA']:.2f}"
                )

                for irf_hdu in self.irf_final_hdu[1:]:
                    self.hdulist.append(irf_hdu)
            else:
                self.log.info("Adding IRF HDUs without interpolation")
                self.irf_final_hdu = fits.open(self.irf_list[0])

                h = self.irf_final_hdu[1].header
                mc_gamma_offset = u.Quantity(h["G_OFFSET"]).to(u.deg)
                zen_pnt = u.Quantity(h["ZEN_PNT"]).to(u.deg)
                az_pnt = u.Quantity(h["AZ_PNT"]).to(u.deg)
                b_delta = u.Quantity(h["B_DELTA"]).to(u.deg)

                self.log.info(f"Gamma offset for MC is {mc_gamma_offset:.2f}")
                self.log.info(f"Zenith pointing of MC at {zen_pnt:.2f}")
                self.log.info(f"Azimuth pointing of MC at {az_pnt:.2f}")
                self.log.info(f"Geomagnetic delta for the MC is {b_delta:.2f}")
                for irf_hdu in self.irf_final_hdu[1:]:
                    self.hdulist.append(irf_hdu)

    def finish(self):

        self.hdulist.writeto(self.output_file, overwrite=self.overwrite)
        if len(self.irf_list) > 1:
            Provenance().add_output_file(self.final_irf_output)

        Provenance().add_output_file(self.output_file)


def main():
    tool = DataReductionFITSWriter()
    tool.run()


if __name__ == "__main__":
    main()
