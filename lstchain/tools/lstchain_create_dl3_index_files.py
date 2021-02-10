"""
Create HDU index files for HDU tables and Obs tables,
from a given path of DL3 files and a glob pattern to select DL3 files

Simple usage with argument aliases:

lstchain_create_dl3_index_files
    --d /path/to/DL3/files/
    --p dl3*[run_1-run_n]*.fits
"""

from lstchain.irf import create_obs_hdu_index
from ctapipe.core import Tool, traits, Provenance

__all__ = [
    'FITSIndexWriter'
    ]

class FITSIndexWriter(Tool):
    name = "FITSIndexWriter"
    description = __doc__

    input_dl3_dir = traits.Path(
        help = "Input path of DL3 files",
        exists = True,
        directory_ok = True,
        file_ok = False
        ).tag(config=True)

    file_pattern = traits.Unicode(
        help = "File pattern to search in the given Path",
        default_value = "dl3*.fits"
        ).tag(config=True)

    aliases = {
        "input_dl3_dir" : "FITSIndexWriter.input_dl3_dir",
        "d" : "FITSIndexWriter.input_dl3_dir",
        "file_pattern" : "FITSIndexWriter.file_pattern",
        "p" : "FITSIndexWriter.file_pattern"
        }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.file_list = []
        self.hdu_index_file = None
        self.obs_index_file = None
        self.hdu_index_list = None
        self.obs_index_list = None
        self.hdu_index_filename = 'hdu-index.fits.gz'
        self.obs_index_filename = 'obs-index.fits.gz'

    def setup(self):
        list_files = sorted(self.input_dl3_dir.glob(self.file_pattern))
        if list_files==[]:
            self.log.critical(f"No files found with pattern {self.file_pattern}")

        for f in list_files:
            self.file_list.append(f.name)

    def start(self):
        # Retrieving HDULists for both index files
        self.hdu_index_list, self.obs_index_list = create_obs_hdu_index(
                                                    self.file_list,
                                                    self.input_dl3_dir,
                                                    self.hdu_index_filename,
                                                    self.obs_index_filename
                                                    )
        self.log.info("HDULists created for the index files")

    def finish(self):
        self.hdu_index_file = self.input_dl3_dir/self.hdu_index_filename
        self.obs_index_file = self.input_dl3_dir/self.obs_index_filename

        if self.hdu_index_file.exists():
            self.log.info(f"The HDU index file {self.hdu_index_file} exists,"
                                        "it will be overwritten")
        if self.obs_index_file.exists():
            self.log.info(f"The Obs index file {self.obs_index_file} exists,"
                                        "it will be overwritten")

        self.hdu_index_list.writeto(self.hdu_index_file, overwrite=True)
        self.obs_index_list.writeto(self.obs_index_file, overwrite=True)

        Provenance().add_output_file(self.hdu_index_file)
        Provenance().add_output_file(self.obs_index_file)

def main():
    tool = FITSIndexWriter()
    tool.run()

if __name__ == "__main__":
    main()
