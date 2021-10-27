"""
Create DL3 FITS file from given data DL2 file,
selection cuts and/or IRF FITS files.

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
from lstchain.irf import create_event_list, add_icrs_position_params
from lstchain.io import EventSelector, DL3FixedCuts

from lstchain.reco.utils import calculate_theta

__all__ = ["DataReductionFITSWriter"]


class DataReductionFITSWriter(Tool):
    name = "DataReductionFITSWriter"
    description = __doc__
    example = """
    To generate DL3 file from an observed data DL2 file, using default cuts:
    > lstchain_create_dl3_file
        -d /path/to/DL2_data_file.h5
        -o /path/to/DL3/file/
        --input-irf /path/to/irf.fits.gz
        --source-name Crab
        --source-ra 83.633deg
        --source-dec 22.01deg

    Or use a config file for the cuts:
    > lstchain_create_dl3_file
        -d /path/to/DL2_data_file.h5
        -o /path/to/DL3/file/
        --input-irf /path/to/irf.fits.gz
        --source-name Crab
        --source-ra 83.633deg
        --source-dec 22.01deg
        --overwrite
        --config /path/to/config.json

    Or pass the selection cuts from command-line:
    > lstchain_create_dl3_file
        -d /path/to/DL2_data_file.h5
        -o /path/to/DL3/file/
        --input-irf /path/to/irf.fits.gz
        --source-name Crab
        --source-ra 83.633deg
        --source-dec 22.01deg
        --fixed-gh-cut 0.9
        --fixed-theta-cut 0.2
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

    input_irf = traits.Path(
        help="Compressed FITS file of IRFs",
        exists=True,
        directory_ok=False,
        file_ok=True,
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

    apply_theta_cut = traits.Bool(
        help="If True, apply theta cuts",
        default=False,
    ).tag(config=True)

    classes = [EventSelector, DL3FixedCuts]

    aliases = {
        ("d", "input-dl2"): "DataReductionFITSWriter.input_dl2",
        ("o", "output-dl3-path"): "DataReductionFITSWriter.output_dl3_path",
        "input-irf": "DataReductionFITSWriter.input_irf",
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
        "apply-theta-cut": (
            {"DataReductionFITSWriter": {"apply_theta_cut": True}},
            "Apply theta cut if true",
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

        # Check if the IRF to be used is POINT-LIKE or FULL ENCLOSURE
        if self.input_irf.exists():
            self.irf_type = fits.open(self.input_irf)[1].header["HDUCLAS3"]
            if self.irf_type == "POINT-LIKE":
                if self.apply_theta_cut:
                    self.point_like = True
                else:
                    raise ToolConfigurationError(
                        "Given IRF is point-like, use --apply-theta-cut"
                    )
            elif self.apply_theta_cut:
                raise ToolConfigurationError(
                    "Provided IRF is not point-like, select appropriate IRF "
                    "when theta cuts are to be applied to real data"
                )
            else:
                self.point_like = False

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

        if self.point_like:
            ## Add another function to get OFF region.
            ## before applying the cuts
            self.data = self.fixed_cuts.theta_cut(self.data)

            self.log.info(
                f"Theta cut applied as {self.irf_type} "
                "IRF is being included"
            )
        else:
            self.log.info(
                f"Theta cut is not applied as {self.irf_type} "
                "IRF is being included"
            )

        self.log.info("Generating event list")
        self.events, self.gti, self.pointing = create_event_list(
            data=self.data,
            run_number=self.run_number,
            source_name=self.source_name,
            source_pos=self.source_pos,
            effective_time=self.effective_time.value,
            elapsed_time=self.elapsed_time.value,
        )

        self.hdulist = fits.HDUList(
            [fits.PrimaryHDU(), self.events, self.gti, self.pointing]
        )
        if self.input_irf:
            irf = fits.open(self.input_irf)
            self.log.info("Adding IRF HDUs")

            for irf_hdu in irf[1:]:
                self.hdulist.append(irf_hdu)

    def finish(self):
        self.hdulist.writeto(self.output_file, overwrite=self.overwrite)

        Provenance().add_output_file(self.output_file)


def main():
    tool = DataReductionFITSWriter()
    tool.run()


if __name__ == "__main__":
    main()
