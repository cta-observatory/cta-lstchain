import pandas as pd
from lstchain.visualization import plot_dl2

from lstchain.tests.test_lstchain import dl2_file, dl2_params_lstcam_key

def test_plot_disp():
    dl2_df = pd.read_hdf(dl2_file, key=dl2_params_lstcam_key)
    plot_dl2.plot_disp(dl2_df)