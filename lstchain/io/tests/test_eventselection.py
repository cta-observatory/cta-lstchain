import pytest
from lstchain.io import EventSelector

def test_event_selection():
   import pandas as pd
   import numpy as np

   evt_fil = EventSelector()

   data = pd.DataFrame({"a": [1, 2, 3], "b": [2.2, 3.2, np.nan], "c": [1, 3, np.inf]})

   evt_fil.filters = dict(a=[0, 2.5], b=[0, 5], c=[0, 4])
   evt_fil.finite_params = ["b"]

   data = evt_fil.filter_cut(data)

   np.testing.assert_array_equal(data, pd.DataFrame({"a": [1, 2], "b": [2.2, 3.2], "c": [1, 3]}))
