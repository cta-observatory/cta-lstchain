"""
Create DL3 FITS file from given data DL2 file,
selection cuts and IRF FITS files.

Change the selection parameters as need be using the aliases.
The default values are written in the EventSelector and DL3Cuts Component
and also given in some example configs in docs/examples/

For the cuts on gammaness, the Tool looks at the IRF provided, to either use
global cuts, based on the header value of the global cut, present in each HDU,
or energy-dependent cuts, based on the GH_CUTS HDU.

To use a separate config file for providing the selection parameters,
copy and append the relevant example config files, into a custom config file.
"""

from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.table import QTable

from ctapipe.core import Tool, traits, Provenance, ToolConfigurationError

from lstchain.io import (
    read_data_dl2_to_QTable, EventSelector, DL3Cuts
)
from lstchain.high_level import (
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
        --global-gh-cut 0.9
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

    classes = [EventSelector, DL3Cuts]

    aliases = {
        ("d", "input-dl2"): "DataReductionFITSWriter.input_dl2",
        ("o", "output-dl3-path"): "DataReductionFITSWriter.output_dl3_path",
        "input-irf": "DataReductionFITSWriter.input_irf",
        "global-gh-cut": "DL3Cuts.global_gh_cut",
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
        self.cuts = DL3Cuts(parent=self)

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
        if not (self.source_ra or self.source_dec):
            self.source_pos = SkyCoord.from_name(self.source_name)
        elif bool(self.source_ra) != bool(self.source_dec):
            raise ToolConfigurationError(
                "Either provide both RA and DEC values for the source or none"
            )
        else:
            self.source_pos = SkyCoord(ra=self.source_ra, dec=self.source_dec)

        self.log.debug(f"Output DL3 file: {self.output_file}")

        try:
            with fits.open(self.input_irf) as hdul:
                self.use_energy_dependent_cuts = (
                    "GH_CUT" not in hdul["EFFECTIVE AREA"].header
                )
        except:
            raise ToolConfigurationError(
                f"{self.input_irf} does not have GH CUTS HDU, "
                "the energy-dependent gammaness cuts HDU, or "
                " any global cut information in the Header value"
            )

    def start(self):

        self.data = read_data_dl2_to_QTable(str(self.input_dl2))
        self.effective_time, self.elapsed_time = get_effective_time(self.data)
        self.run_number = run_info_from_filename(self.input_dl2)[1]

        self.data = self.event_sel.filter_cut(self.data)

        if self.use_energy_dependent_cuts:
            self.energy_dependent_gh_cuts = QTable.read(
                self.input_irf, hdu="GH_CUTS"
            )

            self.data = self.cuts.apply_energy_dependent_gh_cuts(
                self.data, self.energy_dependent_gh_cuts
            )
            self.data = add_icrs_position_params(self.data, self.source_pos)
            self.log.info(
                "Using gamma efficiency of "
                f'{self.energy_dependent_gh_cuts.meta["GH_EFF"]}'
            )
        else:
            self.cuts.global_gh_cut = QTable.read(
                self.input_irf, hdu=1
            ).meta["GH_CUT"]
            self.data = self.cuts.apply_global_gh_cut(self.data)
            self.data = add_icrs_position_params(self.data, self.source_pos)
            self.log.info(f"Using global G/H cut of {self.cuts.global_gh_cut}")

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
