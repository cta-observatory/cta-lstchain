import pandas as pd
import os
from lstchain.visualization import plot_dl2
from lstchain.tests.test_lstchain import dl2_file, dl2_params_lstcam_key, test_dir
from lstchain.scripts.tests.test_lstchain_scripts import output_dir

def test_plot_disp():
    dl2_df = pd.read_hdf(dl2_file, key=dl2_params_lstcam_key)
    plot_dl2.plot_disp(dl2_df)

def test_direction_results():
    dl2_df = pd.read_hdf(dl2_file, key=dl2_params_lstcam_key)
    plot_dl2.direction_results(dl2_df,
                               points_outfile=os.path.join(test_dir, 'dir.h5'),
                               plot_outfile=os.path.join(test_dir, 'dir.png'))

def test_energy_results():
    dl2_df = pd.read_hdf(dl2_file, key=dl2_params_lstcam_key)
    plot_dl2.energy_results(dl2_df,
                            points_outfile=os.path.join(test_dir, 'ene.h5'),
                            plot_outfile=os.path.join(test_dir, 'ene.png'))

def test_plot_models_features_importances():
    plot_dl2.plot_models_features_importances(output_dir)

