"""
Create DL3 FITS file from given data DL2 file,
selection cuts and IRF FITS files.

Change the selection parameters as need be using the aliases.
The default values are written in the EventSelector and DL3Cuts Component
and also given in some example configs in docs/examples/

For the cuts on gammaness, the Tool looks at the IRF provided, to either use
global cuts, based on the header value of the global gammaness cut, GH_CUT,
present in each HDU, or energy-dependent cuts, based on the GH_CUTS HDU.

To use a separate config file for providing the selection parameters,
copy and append the relevant example config files, into a custom config file.

For source-dependent analysis, a source-dep flag should be activated.
Similarly to the cuts on gammaness, the global alpha cut values are provided
from AL_CUT stored in the HDU header.
The alpha cut is already applied on this step, and all survived events with
each assumed source position (on and off) are saved after the gammaness and
alpha cut.
To adapt to the high-level analysis used by gammapy, assumed source position
(on and off) is set as a reco source position just as a trick to obtain
survived events easily.
"""

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import vstack, QTable
from ctapipe.core import (
    Provenance,
    Tool,
    ToolConfigurationError,
    traits,
)

from lstchain.io import (
    EventSelector,
    DL3Cuts,
    get_srcdep_assumed_positions,
    read_data_dl2_to_QTable,
)
from lstchain.high_level import (
    add_icrs_position_params,
    create_event_list,
    set_expected_pos_to_reco_altaz,
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

    Or generate source-dependent DL3 files
    > lstchain_create_dl3_file
        -d /path/to/DL2_data_file.h5
        -o /path/to/DL3/file/
        --input-irf /path/to/irf.fits.gz
        --source-name Crab
        --source-dep
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

    source_dep = traits.Bool(
        help="If True, source-dependent analysis will be performed.",
        default_value=False,
    ).tag(config=True)

    gzip = traits.Bool(
        help="If True, the DL3 file will be gzipped",
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
        "source-dep": (
            {"DataReductionFITSWriter": {"source_dep": True}},
            "source-dependent analysis if True",
        ),
        "gzip": (
            {"DataReductionFITSWriter": {"gzip": True}},
            "gzip the DL3 files if True",
        ),
    }

    def setup(self):

        self.filename_dl3 = dl2_to_dl3_filename(self.input_dl2, compress=self.gzip)
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
                self.use_energy_dependent_gh_cuts = (
                    "GH_CUT" not in hdul["EFFECTIVE AREA"].header
                )
        except:
            raise ToolConfigurationError(
                f"{self.input_irf} does not have EFFECTIVE AREA HDU, "
                " to check for global cut information in the Header value"
            )

        if self.source_dep:
            with fits.open(self.input_irf) as hdul:
                self.use_energy_dependent_alpha_cuts = (
                    "AL_CUT" not in hdul["EFFECTIVE AREA"].header
                )
            
    def apply_srcindep_gh_cut(self):
        ''' apply gammaness cut '''
        self.data = self.event_sel.filter_cut(self.data)

        if self.use_energy_dependent_gh_cuts:
            self.energy_dependent_gh_cuts = QTable.read(
                self.input_irf, hdu="GH_CUTS"
            )

            self.data = self.cuts.apply_energy_dependent_gh_cuts(
                self.data, self.energy_dependent_gh_cuts
            )
            self.log.info(
                "Using gamma efficiency of "
                f"{self.energy_dependent_gh_cuts.meta['GH_EFF']}"
            )
        else:
            with fits.open(self.input_irf) as hdul:
                self.cuts.global_gh_cut = hdul[1].header["GH_CUT"]
            self.data = self.cuts.apply_global_gh_cut(self.data)
            self.log.info(f"Using global G/H cut of {self.cuts.global_gh_cut}")

    def apply_srcdep_gh_alpha_cut(self):
        ''' apply gammaness and alpha cut for source-dependent analysis '''
        srcdep_assumed_positions = get_srcdep_assumed_positions(self.input_dl2)

        for i, srcdep_pos in enumerate(srcdep_assumed_positions):
            data_temp = read_data_dl2_to_QTable(
                self.input_dl2, srcdep_pos=srcdep_pos
            )

            data_temp = self.event_sel.filter_cut(data_temp)
            
            if self.use_energy_dependent_gh_cuts:
                self.energy_dependent_gh_cuts = QTable.read(
                    self.input_irf, hdu="GH_CUTS"
                )

                data_temp = self.cuts.apply_energy_dependent_gh_cuts(
                    data_temp, self.energy_dependent_gh_cuts
                )
                self.log.info(
                    "Using gamma efficiency of "
                    f"{self.energy_dependent_gh_cuts.meta['GH_EFF']}"
                )
            else:
                with fits.open(self.input_irf) as hdul:
                    self.cuts.global_gh_cut = hdul[1].header["GH_CUT"]
                data_temp = self.cuts.apply_global_gh_cut(data_temp)
                    
            if self.use_energy_dependent_alpha_cuts:
                self.energy_dependent_alpha_cuts = QTable.read(
                    self.input_irf, hdu="AL_CUTS"
                )
                data_temp = self.cuts.apply_energy_dependent_alpha_cuts(
                    data_temp, self.energy_dependent_alpha_cuts
                )
                self.log.info(
                    "Using alpha containment region of "
                    f'{self.energy_dependent_alpha_cuts.meta["AL_CONT"]}'
                )
            else:
                with fits.open(self.input_irf) as hdul:
                    self.cuts.global_alpha_cut = hdul[1].header["AL_CUT"]
                data_temp = self.cuts.apply_global_alpha_cut(data_temp)

            # set expected source positions as reco positions
            set_expected_pos_to_reco_altaz(data_temp)

            if i == 0:
                self.data = data_temp
            else:
                self.data = vstack([self.data, data_temp])

    def start(self):

        if not self.source_dep:
            self.data = read_data_dl2_to_QTable(self.input_dl2)
        else:
            self.data = read_data_dl2_to_QTable(self.input_dl2, 'on')
        self.effective_time, self.elapsed_time = get_effective_time(self.data)
        self.run_number = run_info_from_filename(self.input_dl2)[1]

        if not self.source_dep:
            self.apply_srcindep_gh_cut()
        else:
            self.apply_srcdep_gh_alpha_cut()

        self.data = add_icrs_position_params(self.data, self.source_pos)

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
