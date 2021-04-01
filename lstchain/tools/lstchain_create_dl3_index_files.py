"""
Create HDU index files for HDU tables and Obs tables,
from a given path of DL3 files and a glob pattern to select DL3 files
The index filenames are the standard as per
http://gamma-astro-data-formats.readthedocs.io/en/latest/

The Index files can be stored in a different path, but by default
they are stored at the same place as the DL3 files.
"""
from lstchain.irf import create_hdu_index_hdu, create_obs_index_hdu
from ctapipe.core import Tool, traits, Provenance, ToolConfigurationError

__all__ = ["FITSIndexWriter"]


class FITSIndexWriter(Tool):
    name = "FITSIndexWriter"
    description = __doc__
    example = """
    To create DL3 index files with default values:
    > lstchain_create_dl3_index_files
        -d /path/to/DL3/files/

    Or specify some more configurations:
    > lstchain_create_dl3_index_files
        -d /path/to/DL3/files/
        -o /path/to/DL3/index/files
        -p dl3*[run_1-run_n]*.fits.gz
        --overwrite
    """

    input_dl3_dir = traits.Path(
        help="Input path of DL3 files",
        exists=True,
        directory_ok=True,
        file_ok=False
    ).tag(config=True)

    file_pattern = traits.Unicode(
        help="File pattern to search in the given Path",
        default_value="dl3*.fits*"
    ).tag(config=True)

    output_index_path = traits.Path(
        help="Output path for the Index files",
        exists=True,
        directory_ok=True,
        file_ok=False,
        default_value=None
    ).tag(config=True)

    overwrite = traits.Bool(
        help="If True, overwrites existing output file without asking",
        default_value=False,
    ).tag(config=True)

    aliases = {
        ("d", "input-dl3-dir"): "FITSIndexWriter.input_dl3_dir",
        ("o", "output-index-path"): "FITSIndexWriter.output_index_path",
        ("p", "file-pattern"): "FITSIndexWriter.file_pattern",
    }

    flags = {
        "overwrite": (
            {"FITSIndexWriter": {"overwrite": True}},
            "overwrite output files if True",
        )
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

        if not self.output_index_path:
            self.output_index_path = self.input_dl3_dir

        self.hdu_index_file = self.output_index_path / self.hdu_index_filename
        self.obs_index_file = self.output_index_path / self.obs_index_filename

        self.provenance_log = self.output_index_path / (self.name + ".provenance.log")

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

        create_hdu_index_hdu(
            self.file_list,
            self.input_dl3_dir,
            self.hdu_index_file,
            self.overwrite,
        )
        create_obs_index_hdu(
            self.file_list,
            self.input_dl3_dir,
            self.obs_index_file,
            self.overwrite
        )
        self.log.debug("HDULists created for the index files")

    def finish(self):

        Provenance().add_output_file(self.hdu_index_file)
        Provenance().add_output_file(self.obs_index_file)


def main():
    tool = FITSIndexWriter()
    tool.run()


if __name__ == "__main__":
    main()
