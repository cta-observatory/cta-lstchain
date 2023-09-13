"""
Create HDU index files for HDU tables and Obs tables,
from a given path of DL3 files and a glob pattern to select DL3 files
The index filenames are the standard as per
http://gamma-astro-data-formats.readthedocs.io/en/latest/

The Index files can be stored in a different path, but by default
they are stored at the same place as the DL3 files.
"""
from ctapipe.core import (
    Provenance,
    Tool,
    ToolConfigurationError,
    traits,
)

from lstchain.high_level import (
    create_hdu_index_hdu,
    create_obs_index_hdu,
)


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
        -p "dl3*[run_1-run_n]*.fits"
        --overwrite

    Or if the DL3 files are stored in sub-directories:
    > lstchain_create_dl3_index_files
       -d /path/to/DL3/files/
       -o /path/to/DL3/index/files
       -p "/sub-directory*/dl3*[run_1-run_n]*.fits"
       --overwrite

    Or if the DL3 files are stored in the current directory:
    > lstchain_create_dl3_index_files
       -d ./
       -o ./
       -p "dl3*[run_1-run_n]*.fits"
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
        default_value="dl3*.fits"
    ).tag(config=True)

    output_index_path = traits.Path(
        help="Output path for the Index files",
        allow_none=True,
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
        self.hdu_index_filename = "hdu-index.fits.gz"
        self.obs_index_filename = "obs-index.fits.gz"

    def setup(self):

        self.list_files = sorted(self.input_dl3_dir.glob(self.file_pattern))
        if len(self.list_files) == 0:
            raise ToolConfigurationError(
                f"No files found with pattern {self.file_pattern} in {self.input_dl3_dir}"
            )

        for f in self.list_files:
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
            self.list_files,
            self.hdu_index_file,
            self.overwrite,
        )
        create_obs_index_hdu(
            self.list_files,
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
