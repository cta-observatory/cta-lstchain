"""
Create h5 file with reconstruction of energy, disp and gamma/hadron separation of events.
"""

import joblib
import numpy as np
import pandas as pd
from ctapipe.core import Provenance, Tool, traits
from lstchain.io import (
    get_dataset_keys,
    read_configuration_file,
    standard_config,
    write_dl2_dataframe,
)
from lstchain.io.io import (
    dl1_images_lstcam_key,
    dl1_params_lstcam_key,
    dl1_params_src_dep_lstcam_key,
)
from lstchain.reco import dl1_to_dl2
from lstchain.reco.utils import filter_events, impute_pointing
from tables import open_file


class ReconstructionHDF5Writer(Tool):

    name = "ReconstructionHDF5Writer"
    description = "Generate a HDF5 file with reconstructed energy, disp and hadroness of events"

    input = traits.Path(
        help="Path to a DL1 HDF5 file", directory_ok=False, exists=True
    ).tag(config=True)
    path_models = traits.Path(
        help="Path where to find the trained RF models", file_ok=False
    ).tag(config=True)
    output_dir = traits.Path(
        help="Path where to store the reconstructed DL2", file_ok=False
    ).tag(config=True)
    config_file = traits.Path(
        help="Path to a configuration file. If none is given, a standard configuration is applied",
        directory_ok=False,
    ).tag(config=True)

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
        self.configuration = None
        self.data = None
        self.reg_energy = None
        self.reg_disp_vector = None
        self.cls_gh = None
        self.dl2 = None

    def setup(self):

        self.log.info("Reading configuration")
        self.configuration = standard_config
        if self.config_file:
            self.configuration = read_configuration_file(self.config_file)

        self.log.info("Reading DL1 file")
        self.data = pd.read_hdf(self.input, key=dl1_params_lstcam_key)
        if self.configuration["source_dependent"]:
            data_src_dep = pd.read_hdf(self.input, key=dl1_params_src_dep_lstcam_key)
            self.data = pd.concat([self.data, data_src_dep], axis=1)

        self.log.info("Reading RF models")
        self.reg_energy = joblib.load(self.path_models / "reg_energy.sav")
        self.reg_disp_vector = joblib.load(self.path_models / "reg_disp_vector.sav")
        self.cls_gh = joblib.load(self.path_models / "cls_gh.sav")

    def start(self):

        # dealing with pointing missing values
        # this happened when `ucts_time` was invalid
        if (
            "alt_tel" in self.data.columns and "az_tel" in self.data.columns
            and (np.isnan(self.data.alt_tel).any() or np.isnan(self.data.az_tel).any())
        ):
            # make sure there is a least one good pointing value to interpolate from
            if np.isfinite(self.data.alt_tel).any() and np.isfinite(self.data.az_tel).any():
                self.data = impute_pointing(self.data)
            else:
                self.data.alt_tel = -np.pi / 2.0
                self.data.az_tel = -np.pi / 2.0
        self.data = filter_events(
            self.data,
            filters=self.configuration["events_filters"],
            finite_params=self.configuration["regression_features"]
            + self.configuration["classification_features"],
        )

        self.log.info("Applying models to data")
        self.dl2 = dl1_to_dl2.apply_models(
            self.data,
            self.cls_gh,
            self.reg_energy,
            self.reg_disp_vector,
            custom_config=self.configuration,
        )

    def finish(self):

        # prepare output dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_file = self.output_dir / self.input.name.replace("dl1", "dl2")
        if output_file.exists():
            raise IOError(f"{output_file} already exists")
        self.log.info(f"Generating {output_file}")

        # prepare dl1_keys
        dl1_keys = get_dataset_keys(self.input)
        if dl1_images_lstcam_key in dl1_keys:
            dl1_keys.remove(dl1_images_lstcam_key)
        if dl1_params_lstcam_key in dl1_keys:
            dl1_keys.remove(dl1_params_lstcam_key)
        if dl1_params_src_dep_lstcam_key in dl1_keys:
            dl1_keys.remove(dl1_params_src_dep_lstcam_key)

        # prepare h5 file
        with open_file(self.input, "r") as h5in:
            with open_file(output_file, "a") as h5out:
                for k in dl1_keys:
                    if not k.startswith("/"):
                        k = "/" + k
                    path, name = k.rsplit("/", 1)
                    if path not in h5out:
                        group_path, group_name = path.rsplit("/", 1)
                        g = h5out.create_group(group_path, group_name, createparents=True)
                    else:
                        g = h5out.get_node(path)
                    h5in.copy_node(k, g, overwrite=True)

        # write h5 file
        write_dl2_dataframe(self.dl2, output_file)
        Provenance().add_output_file(
            output_file,
            role="DL2/Event"
        )


def main():
    exe = ReconstructionHDF5Writer()
    exe.run()


if __name__ == "__main__":
    main()
