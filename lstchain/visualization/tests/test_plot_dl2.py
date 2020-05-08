import pandas as pd
from lstchain.visualization import plot_dl2

from lstchain.tests.test_lstchain import dl2_file, dl2_params_lstcam_key, test_dir

def test_plot_disp():
    dl2_df = pd.read_hdf(dl2_file, key=dl2_params_lstcam_key)
    plot_dl2.plot_disp(dl2_df)

def test_direction_results():
    dl2_df = pd.read_hdf(dl2_file, key=dl2_params_lstcam_key)
    plot_dl2.direction_results(dl2_df, points_outfile='dir.h5', plot_outfile='dir.png')

def test_energy_results():
    dl2_df = pd.read_hdf(dl2_file, key=dl2_params_lstcam_key)
    plot_dl2.energy_results(dl2_df, points_outfile='ene.h5', plot_outfile='ene.png')

def test_plot_models_features_importances():
    plot_dl2.plot_models_features_importances(test_dir)

