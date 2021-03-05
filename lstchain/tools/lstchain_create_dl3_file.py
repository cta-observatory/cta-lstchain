"""
Create DL3 FITS file from given data DL2 file,
selection cuts and/or IRF FITS files

Simple usage with argument aliases and standard config file for selection cuts:

lstchain_create_irf_files
    --d /path/to/DL2_data_file.h5
    --o /path/to/DL3/file/
    --irf /path/to/irf.fits.gz
    --source_name Crab
    --config /path/to/cta-lstchain/lstchain/data/data_selection_cuts.json
"""

from astropy.io import fits
import astropy.units as u

from ctapipe.core import Tool, traits, Provenance, ToolConfigurationError
from lstchain.io import read_data_dl2_to_QTable, read_configuration_file
from lstchain.reco.utils import filter_events, get_effective_time
from lstchain.paths import run_info_from_filename, dl2_to_dl3_filename
from lstchain.irf import create_event_list

from pyirf.utils import calculate_source_fov_offset

__all__ = ["DataReductionFITSWriter"]


class DataReductionFITSWriter(Tool):
    name = "DataReductionFITSWriter"
    description = __doc__

    input_dl2 = traits.Path(
        help="Input data DL2 file", exists=True, directory_ok=False, file_ok=True
    ).tag(config=True)

    output_dl3_path = traits.Path(
        help="DL3 output filedir", directory_ok=True, file_ok=False
    ).tag(config=True)

    input_irf = traits.Path(
        help="Compressed FITS file of IRFs",
        exists=True,
        directory_ok=False,
        file_ok=True,
    ).tag(config=True)

    source_name = traits.Unicode(help="Name of Source").tag(config=True)

    overwrite = traits.Bool(
        help="If True, overwrites existing output file without asking",
        default_value=True,
    ).tag(config=True)

    aliases = {
        ("d", "input_dl2"): "DataReductionFITSWriter.input_dl2",
        ("o", "output_dl3_path"): "DataReductionFITSWriter.output_dl3_path",
        ("irf", "input_irf"): "DataReductionFITSWriter.input_irf",
        "source_name": "DataReductionFITSWriter.source_name",
    }

    flags = {
        "overwrite": (
            {"DataReductionFITSWriter": {"overwrite": True}},
            "overwrite output file",
        ),
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setup(self):

        self.cuts = read_configuration_file(self.config_file)
        self.filename_dl3 = dl2_to_dl3_filename(self.input_dl2)
        self.provenance_log = self.output_dl3_path / (self.name + ".provenance.log")

        Provenance().add_input_file(self.input_dl2)

        self.output_file = self.output_dl3_path / self.filename_dl3
        if self.output_file.exists() and not self.overwrite:
            raise ToolConfigurationError(
                f"Output file {self.output_file} already exists,"
                " use --overwrite to overwrite"
            )

    def start(self):

        self.data = read_data_dl2_to_QTable(str(self.input_dl2))
        self.effective_time, self.elapsed_time = get_effective_time(self.data)
        self.run_number = run_info_from_filename(self.input_dl2)[1]

        self.data["reco_source_fov_offset"] = calculate_source_fov_offset(
            self.data, prefix="reco"
        )

        self.data = filter_events(self.data, self.cuts["events_filters"])
        # Separate cuts for angular separations, for now
        self.data = self.data[
            self.data["gh_score"] > self.cuts["fixed_cuts"]["gh_score"][0]
        ]
        self.data = self.data[
            self.data["reco_source_fov_offset"]
            < u.Quantity(**self.cuts["fixed_cuts"]["source_fov_offset"])
        ]

        self.log.info("Generating event list")
        self.events, self.gti, self.pointing = create_event_list(
            data=self.data,
            run_number=self.run_number,
            source_name=self.source_name,
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
