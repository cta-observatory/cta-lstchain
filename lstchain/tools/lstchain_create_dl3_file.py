"""
Create DL3 FITS file from given data DL2 file,
selection cuts and/or IRF FITS files

Change the selection parameters as need be.
The default values are also written in lstchain/data/data_selection_cuts.json

Simple usage with argument aliases and default parameter selection values:

lstchain_create_irf_files
    --d /path/to/DL2_data_file.h5
    --o /path/to/DL3/file/
    --irf /path/to/irf.fits.gz
    --source_name Crab
"""

from astropy.io import fits
import astropy.units as u
import numpy as np

from ctapipe.core import Tool, traits, Provenance, ToolConfigurationError
from lstchain.io import read_data_dl2_to_QTable
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

    event_filters = traits.Dict(
        help="Enter the event filters for standard parameters - "
        "intensity, leakage_intensity_width_2, r, wl",
        default_value=dict(
            {
                "intensity": [100, np.inf],
                "r": [0, 1],
                "wl": [0.1, 1],
                "leakage_intensity_width_2": [0, 0.2],
            }
        ),
    ).tag(config=True)

    fixed_cuts = traits.Dict(
        help="Enter the fixed selection cut values for "
        "gh_score(gammaness), theta and source_fov_offset",
        default_value=dict(
            {
                "gh_score": 0.6,
                "theta_cut": 0.2,
                "source_fov_offset": 2.83,
            }
        ),
    ).tag(config=True)

    alpha = traits.Float(
        help="Enter the selection cut for source dependent parameter - alpha",
        default_value=8.0,
    ).tag(config=True)

    tel_ids = traits.Dict(
        help="Enter the relevant tel ids for LST and MAGIC",
        default_value=dict(
            {
                "LST_tels": [1],
            }
        ),
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

        self.data = filter_events(self.data, self.event_filters)
        # Separate cuts for angular separations, for now
        self.data = self.data[self.data["gh_score"] > self.fixed_cuts["gh_score"]]
        self.data = self.data[
            self.data["reco_source_fov_offset"]
            < u.Quantity(self.fixed_cuts["source_fov_offset"] * u.deg)
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
