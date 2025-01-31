import json
import h5py
from ctapipe.core import Provenance
import logging

logger = logging.getLogger()

def write_provenance(hdf5_file_path, stage_name):
    """
    Write JSON provenance information to an HDF5 file.
    It uses the current activity's provenance information and should typically be called within a ctapipe Tool.

    Parameters:
    -----------
    hdf5_file_path : str or Path
        Path to the HDF5 file
    stage_name : str
        Name of the stage generating the provenance

    Returns:
    --------
    None
    """
    try:
        with h5py.File(hdf5_file_path, 'a') as h5file:
            if 'provenance' not in h5file:
                h5file.create_group('provenance')
            
            # Get the provenance dictionary from the current activity
            provenance_data = Provenance().current_activity.provenance
            # Dump the dictionary to a JSON string and write it to the HDF5 file
            h5file['provenance'].create_dataset(stage_name, data=json.dumps(provenance_data, default=str))
    
    except Exception as e:
        raise Exception(f"Error writing provenance: {e}")


def read_provenance(hdf5_file_path, dataset_name):
    """
    Read JSON provenance from HDF5 file's dataset attributes.

    Parameters:
    -----------
    hdf5_file_path : str
        Path to the HDF5 file
    dataset_name : s
        Name of the dataset containing provenance

    Returns:
    --------
    dict
        Provenance information as JSON-decoded dictionary
    """
    logger.log(logging.INFO, f"reading provenance from {hdf5_file_path}")
    with h5py.File(hdf5_file_path, 'r') as h5file:
        if 'provenance' not in h5file:
            raise ValueError("No provenance found in HDF5 file")
        elif dataset_name not in h5file['provenance']:
            raise ValueError(f"No provenance found for {dataset_name}")
        else:
            return json.loads(h5file['provenance'][dataset_name][()])


def read_dl2_provenance(hdf5_file_path):
    """
    Read JSON provenance from HDF5 file's dataset attributes.
    This function is a wrapper around read_provenance() that reads the provenance for the 'dl2' dataset.

    Parameters:
    -----------
    hdf5_file_path : str
        Path to the HDF5 file

    Returns:
    --------
    dict
        Provenance information as JSON-decoded dictionary
    """
    return read_provenance(hdf5_file_path, 'dl1_to_dl2')