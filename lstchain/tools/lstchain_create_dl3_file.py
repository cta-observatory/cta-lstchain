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

from astropy.io import fits
from astropy.coordinates import SkyCoord

from ctapipe.core import Tool, traits, Provenance, ToolConfigurationError
from lstchain.io import read_data_dl2_to_QTable
from lstchain.reco.utils import get_effective_time
from lstchain.paths import run_info_from_filename, dl2_to_dl3_filename
from lstchain.irf.hdu_table import create_event_list
from lstchain.irf.interpolate import compare_irfs, interpolate_irf
from lstchain.io import EventSelector, DL3FixedCuts

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
        -p irf*.fits.gz
        -f final_interp_irf.fits.gz
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
        --overwrite
    """

    input_dl2 = traits.Path(
        help="Input data DL2 file",
        exists=True,
        directory_ok=False,
        file_ok=True
    ).tag(config=True)

    output_dl3_path = traits.Path(
        help="DL3 output filedir",
        directory_ok=True,
        file_ok=False
    ).tag(config=True)

    input_irf_path = traits.Path(
        help="Compressed FITS file of IRFs",
        exists=True,
        directory_ok=True,
        file_ok=False,
    ).tag(config=True)

    irf_file_pattern = traits.Unicode(
        help="IRF file pattern to search in the given IRF files path",
        default_value="*irf*.fits.gz"
    ).tag(config=True)

    final_irf_file = traits.Path(
        help="Final IRF file included with DL3 file",
        directory_ok=False,
        file_ok=True,
        default_value="final_irf.fits.gz",
    ).tag(config=True)

    source_name = traits.Unicode(
        help="Name of Source"
    ).tag(config=True)

    source_ra = traits.Unicode(
        help="RA position of the source"
    ).tag(config=True)

    source_dec = traits.Unicode(
        help="DEC position of the source"
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
        "fixed-gh-cut": "DL3FixedCuts.fixed_gh_cut",
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
                self.log.info(self.irf_list)
                if not compare_irfs(self.irf_list):
                    raise ToolConfigurationError(
                        f"IRF files in {self.input_irf_path} with pattern, "
                        f"{self.irf_file_pattern} are not similar and "
                        "cannot be used to interpolate. Use different list of IRFs"
                    )
            elif len(self.irf_list) == 1:
                self.log.info(f"Only single IRF {self.irf_list[0]} provided, no interpolation possible")
            else:
                raise ToolConfigurationError(
                    f"No IRF files in {self.input_irf_path} with pattern, "
                    f"{self.irf_file_pattern} found. Use different parameters"
                )

        self.final_irf_output = self.input_irf_path.absolute() / str(self.final_irf_file.name)

        self.log.info(self.final_irf_output)

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
        self.effective_time, self.elapsed_time = get_effective_time(self.data)
        self.run_number = run_info_from_filename(self.input_dl2)[1]

        self.data = self.event_sel.filter_cut(self.data)
        self.data = self.fixed_cuts.gh_cut(self.data)

        self.log.info("Generating event list")
        self.events, self.gti, self.pointing, self.data_params = create_event_list(
            data=self.data,
            run_number=self.run_number,
            source_name=self.source_name,
            source_pos=self.source_pos,
            effective_time=self.effective_time.value,
            elapsed_time=self.elapsed_time.value,
        )
        self.log.info(self.data_params)
        ## Add a user defined option to select the parameters to interpolate IRFs if present

        self.hdulist = fits.HDUList(
            [fits.PrimaryHDU(), self.events, self.gti, self.pointing]
        )

        if self.input_irf_path:
            if len(self.irf_list) > 1:
                ## Add another check to have at least 2 files for each
                ## parameter to perform the interpolation
                self.irf_final_hdu = interpolate_irf(self.irf_list, self.data_params)

                self.log.info("Adding IRF HDUs")

                self.mc_params = dict()
                for p in self.data_params.keys():
                    # Assuming all the header values have units with 4 spaces
                    self.mc_params[p] = float(self.irf_final_hdu[1].header[p][:-4])
                mc_gamma_offset = float(self.irf_final_hdu[1].header["G_OFFSET"][:-4])

                self.log.info(f"Gamma offset for MC is {mc_gamma_offset}")
                self.log.info(f"Zenith pointing of MC at {self.mc_params['ZEN_PNT']}")
                self.log.info(f"Azimuth pointing of MC at {self.mc_params['AZ_PNT']}")

                for irf_hdu in self.irf_final_hdu[1:]:
                    self.hdulist.append(irf_hdu)
            else:
                self.log.info("Adding IRF HDUs without interpolation")
                self.irf_final_hdu = fits.open(self.irf_list[0])
                self.log.info(type(self.irf_final_hdu))
                self.log.info(self.irf_final_hdu.info())
                for irf_hdu in self.irf_final_hdu[1:]:
                    self.hdulist.append(irf_hdu)

    def finish(self):

        self.hdulist.writeto(self.output_file, overwrite=self.overwrite)
        if len(self.irf_list) > 1:
            fits.HDUList(self.irf_final_hdu).writeto(self.final_irf_output, overwrite=self.overwrite)
            Provenance().add_output_file(self.final_irf_output)

        Provenance().add_output_file(self.output_file)


def main():
    tool = DataReductionFITSWriter()
    tool.run()


if __name__ == "__main__":
    main()
