# <<<<<<< HEAD
# import os
# import h5py
# from shutil import copyfile
# import tables
#
# __all__ = ['get_dataset_keys',
#            'merge_events_tables',
#            ]
#
# def get_dataset_keys(filename):
#     """
#     Return a list of all dataset keys in a HDF5 file
#
#     Parameters
#     ----------
#     filename: str - path to the HDF5 file
#
#     Returns
#     -------
#     list of keys
#     """
#     dataset_keys = []
#     def walk(name, obj):
#         if type(obj) == h5py._hl.dataset.Dataset:
#             dataset_keys.append(name)
#
#     with h5py.File(filename, 'r') as file:
#         file.visititems(walk)
#
#     return dataset_keys
#
#
# def merge_events_tables(filelist, outfile, events_table='events'):
#     """
#     Merge events table from files in filelist and save the output in outfile.
#     Events in the files are supposed to be tables organised per camera type in the events table.
#
#     Parameters
#     ----------
#     filelist: list of files
#     outfile: str
#     events_table: str
#     """
#     assert len(filelist) > 1
#
#     filename0 = filelist[0]
#     copyfile(filename0, outfile)
#
#     filenames = [os.path.basename(filename0)]
#
#     table_names = [k for k in get_dataset_keys(filename0) if events_table in k]
#
#     with tables.File(outfile, mode='a') as merged_file:
#
#         for filename in filelist[1:]:
#             file = tables.File(filename)
#             filenames.append(os.path.basename(file.filename))
#
#             for table_name in table_names:
#                 merged_table = merged_file.root[table_name]
#
#                 table = file.root[table_name]
#                 table.append_where(merged_table)
#                 file.close()
#
#         merged_file.root._v_attrs.filenames = filenames
# =======

from ctapipe.io import HDF5TableReader
from ctapipe.io.containers import MCHeaderContainer
from tables import open_file
import h5py
import numpy as np

__all__ = ['read_simu_info_hdf5',
           'read_simu_info_merged_hdf5',
           'get_dataset_keys',
           ]

def read_simu_info_hdf5(filename):
    """
    Read simu info from an hdf5 file

    Returns
    -------
    `ctapipe.containers.MCHeaderContainer`
    """

    with HDF5TableReader(filename) as reader:
        mcheader = reader.read('/simulation/run_config', MCHeaderContainer())
        mc = next(mcheader)

    return mc


def read_simu_info_merged_hdf5(filename):
    """
    Read simu info from a merged hdf5 file.
    Check that simu info are the same for all runs from merged file
    Combine relevant simu info such as num_showers (sum)
    Note: works for a single run file as well

    Parameters
    ----------
    filename: path to an hdf5 merged file

    Returns
    -------
    `ctapipe.containers.MCHeaderContainer`

    """
    with open_file(filename) as file:
        simu_info = file.root['simulation/run_config']
        colnames = simu_info.colnames
        colnames.remove('num_showers')
        colnames.remove('shower_prog_start')
        colnames.remove('detector_prog_start')
        for k in colnames:
            assert np.all(simu_info[:][k] == simu_info[0][k])
        num_showers = simu_info[:]['num_showers'].sum()

    combined_mcheader = read_simu_info_hdf5(filename)
    combined_mcheader['num_showers'] = num_showers
    return combined_mcheader


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

    with h5py.File(filename,'r') as file:
        file.visititems(walk)

    return dataset_keys

