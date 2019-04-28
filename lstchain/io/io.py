import os
import h5py
from shutil import copyfile
import tables

__all__ = ['get_dataset_keys',
           'merge_events_tables',
           ]

def get_dataset_keys(filename):
    """
    Return a list of all dataset keys in a HDF5 file

    Parameters
    ----------
    filename: str - path to the HDF5 file

    Returns
    -------
    list of keys
    """
    dataset_keys = []
    def walk(name, obj):
        if type(obj) == h5py._hl.dataset.Dataset:
            dataset_keys.append(name)

    with h5py.File(filename, 'r') as file:
        file.visititems(walk)

    return dataset_keys


def merge_events_tables(filelist, outfile, events_table='events'):
    """
    Merge events table from files in filelist and save the output in outfile.
    Events in the files are supposed to be tables organised per camera type in the events table.

    Parameters
    ----------
    filelist: list of files
    outfile: str
    events_table: str
    """
    assert len(filelist) > 1

    filename0 = filelist[0]
    copyfile(filename0, outfile)

    filenames = [os.path.basename(filename0)]

    table_names = [k for k in get_dataset_keys(filename0) if events_table in k]

    with tables.File(outfile, mode='a') as merged_file:

        for filename in filelist[1:]:
            file = tables.File(filename)
            filenames.append(os.path.basename(file.filename))

            for table_name in table_names:
                merged_table = merged_file.root[table_name]

                table = file.root[table_name]
                table.append_where(merged_table)
                file.close()

        merged_file.root._v_attrs.filenames = filenames