import os

import matplotlib.pyplot as plt
import pandas as pd

from lstchain.io.io import dl2_params_lstcam_key
from lstchain.visualization import plot_dl2


def test_plot_disp(simulated_dl2_file):
    dl2_df = pd.read_hdf(simulated_dl2_file, key=dl2_params_lstcam_key)
    plot_dl2.plot_disp(dl2_df)


def test_direction_results(tmp_path, simulated_dl2_file):
    dl2_df = pd.read_hdf(simulated_dl2_file, key=dl2_params_lstcam_key)
    # Strings are required as input for the output files not PosixPath
    plot_dl2.direction_results(
        dl2_df,
        points_outfile=os.path.join(tmp_path, 'dir.h5'),
        plot_outfile=os.path.join(tmp_path, 'dir.png')
    )


def test_energy_results(tmp_path, simulated_dl2_file):
    dl2_df = pd.read_hdf(simulated_dl2_file, key=dl2_params_lstcam_key)
    # Strings are required as input for the output files not PosixPath
    plot_dl2.energy_results(
        dl2_df,
        points_outfile=os.path.join(tmp_path, 'ene.h5'),
        plot_outfile=os.path.join(tmp_path, 'ene.png')
    )


def test_plot_models_features_importances(rf_models):
    fig, axes = plt.subplots(1, 3, figsize=(15, 10))
    plot_dl2.plot_models_features_importances(rf_models["path"], axes=axes, alpha=0.5, fill=False)
