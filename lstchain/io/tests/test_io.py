from lstchain.io import io
import numpy as np
import pandas as pd
import tables
import tempfile


def test_write_dataframe():
    a = np.ones(3)
    df = pd.DataFrame(a, columns=['a'])
    with tempfile.NamedTemporaryFile() as f:
        io.write_dataframe(df, f.name, 'data/awesome_table')
        with tables.open_file(f.name) as file:
            np.testing.assert_array_equal(file.root.data.awesome_table[:]['a'], a)
