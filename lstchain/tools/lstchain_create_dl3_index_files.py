"""
Create HDU index files for HDU tables and Obs tables,
from a given path of DL3 files and a glob pattern to select DL3 files
The index filenames are the standard as per
http://gamma-astro-data-formats.readthedocs.io/en/latest/

To add the FITS directory information in the HDU index table, enter
--add_fits_dir True

Simple usage with argument aliases:

lstchain_create_dl3_index_files
    --d /path/to/DL3/files/
    --p dl3*[run_1-run_n]*.fits.gz
"""
from lstchain.irf import create_hdu_index_hdu, create_obs_index_hdu
from ctapipe.core import Tool, traits, Provenance, ToolConfigurationError

__all__ = ["FITSIndexWriter"]


class FITSIndexWriter(Tool):
    name = "FITSIndexWriter"
    description = __doc__

    input_dl3_dir = traits.Path(
        help="Input path of DL3 files", exists=True, directory_ok=True, file_ok=False
    ).tag(config=True)

    file_pattern = traits.Unicode(
        help="File pattern to search in the given Path", default_value="dl3*.fits"
    ).tag(config=True)

    add_fits_dir = traits.Bool(
        help="If True, adds the path of fits files in HDU index table",
        default_value=False,
    ).tag(config=True)

    overwrite = traits.Bool(
        help="If True, overwrites existing output file without asking",
        default_value=True,
    ).tag(config=True)

    aliases = {
        ("d", "input_dl3_dir"): "FITSIndexWriter.input_dl3_dir",
        ("p", "file_pattern"): "FITSIndexWriter.file_pattern",
        "add_fits_dir": "FITSIndexWriter.add_fits_dir",
        "overwrite": "FITSIndexWriter.overwrite",
    }

    flags = {
        "overwrite": (
            {"FITSIndexWriter": {"overwrite": True}},
            "overwrite output files if True",
        ),
        "add_fits_dir": (
            {"FITSIndexWriter": {"add_fits_dir": False}},
            "Add directory of FITS file to HDU Index table",
        ),
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.file_list = []
        self.hdu_index_filename = "hdu-index.fits.gz"
        self.obs_index_filename = "obs-index.fits.gz"

    def setup(self):
        list_files = sorted(self.input_dl3_dir.glob(self.file_pattern))
        if list_files == []:
            self.log.critical(f"No files found with pattern {self.file_pattern}")

        for f in list_files:
            self.file_list.append(f.name)
            Provenance().add_input_file(f)

        self.hdu_index_file = self.input_dl3_dir.absolute() / self.hdu_index_filename
        self.obs_index_file = self.input_dl3_dir.absolute() / self.obs_index_filename

        self.provenance_log = self.input_dl3_dir / (self.name + ".provenance.log")

        if self.hdu_index_file.exists():
            if self.overwrite:
                self.log.warning(f"Overwriting {self.hdu_index_file}")
                self.hdu_index_file.unlink()
            else:
                raise ToolConfigurationError(
                    f"Output file {self.hdu_index_file} already exists,"
                    "use --overwrite to overwrite"
                )

        if self.obs_index_file.exists():
            if self.overwrite:
                self.log.warning(f"Overwriting {self.obs_index_file}")
                self.obs_index_file.unlink()
            else:
                raise ToolConfigurationError(
                    f"Output file {self.obs_index_file} already exists,"
                    " use --overwrite to overwrite"
                )

        self.log.debug("HDU Index file: %s", self.hdu_index_file)
        self.log.debug("OBS Index file: %s", self.obs_index_file)

    def start(self):

        # Retrieving HDULists for both index files
        self.hdu_index_list = create_hdu_index_hdu(
            self.file_list,
            self.input_dl3_dir,
            self.hdu_index_filename,
            self.add_fits_dir,
        )
        self.obs_index_list = create_obs_index_hdu(
            self.file_list,
            self.input_dl3_dir,
            self.obs_index_filename,
        )
        self.log.debug("HDULists created for the index files")

    def finish(self):

        self.hdu_index_list.writeto(self.hdu_index_file, overwrite=self.overwrite)
        self.obs_index_list.writeto(self.obs_index_file, overwrite=self.overwrite)

        Provenance().add_output_file(self.hdu_index_file)
        Provenance().add_output_file(self.obs_index_file)


def main():
    tool = FITSIndexWriter()
    tool.run()


if __name__ == "__main__":
    main()
