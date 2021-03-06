"""
Create h5 file with reconstruction of energy, disp and gamma/hadron separation of events.
"""

import joblib
import numpy as np
import pandas as pd
from ctapipe.core import Provenance, Tool, ToolConfigurationError, traits
from lstchain.io import get_dataset_keys, replace_config, standard_config, write_dl2_dataframe
from lstchain.io.io import (dl1_images_lstcam_key, dl1_params_lstcam_key,
                            dl1_params_src_dep_lstcam_key,
                            dl2_params_src_dep_lstcam_key, write_dataframe)
from lstchain.reco import dl1_to_dl2
from lstchain.reco.utils import add_delta_t_key, filter_events, impute_pointing
from tables import open_file


class ReconstructionHDF5Writer(Tool):
    name = "ReconstructionHDF5Writer"
    description = "Generate a HDF5 file with reconstructed energy, disp and gammaness of events"

    input = traits.Path(help="Path to a DL1 HDF5 file", directory_ok=False, exists=True).tag(config=True)
    path_energy_model = traits.Path(
        help="Path where to find the Energy trained RF model file",
        directory_ok=False,
        exists=True,
    ).tag(config=True)
    path_disp_model = traits.Path(
        help="Path where to find the Disp trained RF model file",
        directory_ok=False,
        exists=True,
    ).tag(config=True)
    path_gh_model = traits.Path(
        help="Path where to find the Gammaness trained RF model file",
        directory_ok=False,
        exists=True,
    ).tag(config=True)
    output_dir = traits.Path(help="Path where to store the reconstructed DL2", file_ok=False).tag(config=True)

    aliases = {
        ("i", "input"): "ReconstructionHDF5Writer.input",
        ("o", "output"): "ReconstructionHDF5Writer.output_dir",
        "energy-model": "ReconstructionHDF5Writer.path_energy_model",
        "disp-model": "ReconstructionHDF5Writer.path_disp_model",
        "gh-model": "ReconstructionHDF5Writer.path_gh_model",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        """
        Tool that generates a HDF5 file with reconstructed energy, disp and gammaness of events.

        For getting help run:
        lstchain_create_dl2_file --help
        """
        self.data_ind = None
        self.data_src_dep = None
        self.dl2_ind = None
        self.dl2_src_dep_level = None
        self.dl2_src_dep_dict = {}
        self.reg_energy = None
        self.reg_disp_vector = None
        self.cls_gh = None

        self._dict_conf = replace_config(standard_config, self.get_current_config())

    def setup(self):

        if self.config_file is None:
            raise ToolConfigurationError("Please provide configuration file with --config parameter")

        self.log.info("Reading DL1 file")
        self.data_ind = pd.read_hdf(self.input, key=dl1_params_lstcam_key)
        if self._dict_conf["source_dependent"]:
            self.data_src_dep = pd.read_hdf(self.input, key=dl1_params_src_dep_lstcam_key)
        self.log.info("Reading RF models")
        self.reg_energy = joblib.load(self.path_energy_model)
        self.reg_disp_vector = joblib.load(self.path_disp_model)
        self.cls_gh = joblib.load(self.path_gh_model)

    def start(self):

        self.data_ind = add_delta_t_key(self.data_ind)
        # dealing with pointing missing values
        # this happened when `ucts_time` was invalid
        if (
            "alt_tel" in self.data_ind.columns
            and "az_tel" in self.data_ind.columns
            and (np.isnan(self.data_ind.alt_tel).any() or np.isnan(self.data_ind.az_tel).any())
        ):
            # make sure there is a least one good pointing value to interpolate from
            if np.isfinite(self.data_ind.alt_tel).any() and np.isfinite(self.data_ind.az_tel).any():
                self.data_ind = impute_pointing(self.data_ind)
            else:
                self.data_ind.alt_tel = -np.pi / 2.0
                self.data_ind.az_tel = -np.pi / 2.0

        # source-independent analysis
        self.log.info("Applying models")
        if not self._dict_conf["source_dependent"]:
            self.data_ind = filter_events(
                self.data_ind,
                filters=self._dict_conf["events_filters"],
                finite_params=self._dict_conf["regression_features"] + self._dict_conf["classification_features"],
            )
            self.dl2_ind = dl1_to_dl2.apply_models(
                self.data_ind,
                self.cls_gh,
                self.reg_energy,
                self.reg_disp_vector,
                custom_config=self._dict_conf,
            )
        # source-dependent analysis
        else:
            self.data_src_dep.columns = pd.MultiIndex.from_tuples(
                [tuple(col[1:-1].replace("'", "").replace(" ", "").split(",")) for col in self.data_src_dep.columns]
            )

            for i, k in enumerate(self.data_src_dep.columns.levels[0]):
                self.data_src_dep.append(self.data_src_dep[k])
                self.data_src_dep = filter_events(
                    self.data_src_dep,
                    filters=self._dict_conf["events_filters"],
                    finite_params=self._dict_conf["regression_features"] + self._dict_conf["classification_features"],
                )
                dl2_level_df = dl1_to_dl2.apply_models(
                    self.data_src_dep,
                    self.cls_gh,
                    self.reg_energy,
                    self.reg_disp_vector,
                    custom_config=self._dict_conf,
                )
                self.dl2_src_dep_dict[k] = dl2_level_df.drop(self.data_ind.keys(), axis=1)
                if i == 0:
                    self.dl2_src_dep_level = dl2_level_df.drop(self.data_src_dep[k].keys(), axis=1)

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
        if not self._dict_conf["source_dependent"]:
            write_dl2_dataframe(self.dl2_ind, output_file)
        else:
            write_dl2_dataframe(self.dl2_src_dep_level, output_file)
            write_dataframe(
                pd.concat(self.dl2_src_dep_dict, axis=1),
                output_file,
                dl2_params_src_dep_lstcam_key,
            )
        Provenance().add_output_file(output_file, role="DL2/Event")


def main():
    exe = ReconstructionHDF5Writer()
    exe.run()


if __name__ == "__main__":
    main()
