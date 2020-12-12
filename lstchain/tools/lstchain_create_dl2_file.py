"""
Create h5 file with reconstruction of energy, disp and gamma/hadron separation of events.
"""
import os

import joblib
import numpy as np
import pandas as pd
from ctapipe.core import Provenance, Tool, traits
from lstchain.io import (get_dataset_keys, read_configuration_file,
                         replace_config, standard_config, write_dl2_dataframe)
from lstchain.io.io import (dl1_images_lstcam_key, dl1_params_lstcam_key,
                            dl1_params_src_dep_lstcam_key)
from lstchain.reco import dl1_to_dl2
from lstchain.reco.utils import filter_events, impute_pointing
from tables import open_file
from tqdm.autonotebook import tqdm


class ReconstructionHDF5Writer(Tool):

    name = "CalibrationHDF5Writer"
    description = (
        "Generate a HDF5 file with reconstructed energy, disp and hadroness of events"
    )

    input = traits.Path(
        help="Path to a DL1 HDF5 file", directory_ok=False, exists=True
    ).tag(config=True)
    path_models = traits.Path(
        help="Path where to find the trained RF", file_ok=False
    ).tag(config=True)
    output_dir = traits.Path(
        help="Path where to store the reconstructed DL2", file_ok=False
    ).tag(config=True)
    config_file = traits.Path(
        help="Path to a configuration file. If none is given, a standard configuration is applied",
        directory_ok=False
    ).tag(config=True)

    # progress_bar = traits.Bool(
    #     help="show progress bar during processing",
    #     default_value=True,
    # ).tag(config=True)

    aliases = {
        "input": "ReconstructionHDF5Writer.input",
        "i": "ReconstructionHDF5Writer.input",
        "models": "ReconstructionHDF5Writer.path_models",
        "output": "ReconstructionHDF5Writer.output_dir",
        "o": "ReconstructionHDF5Writer.output_dir",
        "config": "ReconstructionHDF5Writer.config_file",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        """
        Tool that generates a HDF5 file with reconstructed energy, disp and hadroness of events.

        For getting help run:
        lstchain_create_dl2_file --help
        """

    def setup(self):
        pass

    def start(self):
        pass

    def finish(self):
        pass


def main():
    exe = ReconstructionHDF5Writer()
    exe.run()


if __name__ == "__main__":
    main()
