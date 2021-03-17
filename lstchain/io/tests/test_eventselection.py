import pytest
from lstchain.io import EventSelector

def test_event_selection(simulated_dl1_file):
   from lstchain.io.io import dl1_params_lstcam_key
   import pandas as pd

   data = pd.read_hdf(simulated_dl1_file, key=dl1_params_lstcam_key)
   evt_fil = EventSelector()

   evt_fil.filters = {
        "intensity": [0, 1000],
        "width": [0, 100],
        "length": [0, 100],
        "r": [0, 1],
        "wl": [0.1, 1],
        "leakage_intensity_width_2": [0, 1]
    }
   data = evt_fil.filter_cut(data)

   assert data["intensity"].max() < 1000
