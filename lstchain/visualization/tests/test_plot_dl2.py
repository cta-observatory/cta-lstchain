import pandas as pd
from lstchain.visualization import plot_dl2

def test_plot_disp():
    dl2_df = pd.read_hdf('dl2_gamma_test_large.h5', key='events/LSTCam')
    plot_dl2.plot_disp(dl2_df)