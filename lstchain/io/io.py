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