import h5py
import numpy as np
from astropy.table import Table, vstack
import tables
from tables import open_file

from ctapipe.io import HDF5TableReader
from ctapipe.io.containers import MCHeaderContainer
from ctapipe.io import HDF5TableWriter
from eventio import Histograms
from eventio.search_utils import yield_toplevel_of_type
from .lstcontainers import ThrownEventsHistogram, ExtraMCInfo


__all__ = ['read_simu_info_hdf5',
           'read_simu_info_merged_hdf5',
           'get_dataset_keys',
           'write_simtel_energy_histogram',
           'write_mcheader',
           'write_array_info',
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

    with h5py.File(filename, 'r') as file:
        file.visititems(walk)

    return dataset_keys


def get_stacked_table(filenames_list, node):
    """
    Stack tables at node from each file in files

    Parameters
    ----------
    filenames_list: list of paths
    node: str

    Returns
    -------
    `astropy.table.Table`
    """
    try:
        files = [tables.open_file(filename) for filename in filenames_list]
    except:
        print("Can't open files")

    table_list = [Table(file.root[node][:]) for file in files]
    [file.close() for file in files]

    return vstack(table_list)


def stack_tables_h5files(filenames_list, output_filename='merged.h5', keys=None):
    """
    In theory similar to auto_merge_h5files but slower. Keeping it for reference.
    Merge h5 files produced by lstchain using astropy.
    A list of keys (corresponding to file nodes) that need to be included in the merge can be given.
    If None, all tables in the file will be merged.

    Parameters
    ----------
    filenames_list: list of str
    output_filename: str
    keys: None or list of str
    """

    keys = get_dataset_keys(filenames_list[0]) if keys is None else keys

    for k in keys:
        merged_table = get_stacked_table(filenames_list, k)
        merged_table.write(output_filename, path=k, append=True)



def auto_merge_h5files(file_list, output_filename='merged.h5', nodes_keys=None):
    """
    Automatic merge of HDF5 files.
    A list of nodes keys can be provided to merge only these nodes. If None, all nodes are merged.

    Parameters
    ----------
    file_list: list of path
    output_filename: path
    nodes_keys: list of path
    """

    keys = get_dataset_keys(file_list[0]) if nodes_keys is None else nodes_keys
    groups = set([k.split('/')[0] for k in keys])

    f1 = tables.open_file(file_list[0])
    merge_file = tables.open_file(output_filename, 'w')

    nodes = {}
    for g in groups:
        nodes[g] = f1.copy_node('/', name=g, newparent=merge_file.root, newname=g, recursive=True)


    for filename in file_list[1:]:
        with tables.open_file(filename) as file:
            for k in keys:
                try:
                    merge_file.root[k].append(file.root[k].read())
                except:
                    print("Can't append node {} from file {}".format(k, filename))

    merge_file.close()


def smart_merge_h5files(file_list, output_filename='merged.h5'):
    """
    Merge HDF5 DL1 or DL2 files in a smart way.
        - stack events tables
        - check that extra information are the same in all files
        - update simulation information such as number of simulated events for you

    Parameters
    ----------
    file_list: list of paths
    output_filename: path
    """
    events_tables_keys = [k for k in get_dataset_keys(file_list[0]) if 'events' in k]
    auto_merge_h5files(file_list, output_filename=output_filename, nodes_keys=events_tables_keys)
    # TODO:
    #  - check that extra info is coherent in all files.
    #  - copy extra info
    #  - update extra info number of simulated shower



def write_simtel_energy_histogram(source, output_filename, obs_id=None, filters=None):
    """
    Write the energy histogram from a simtel source to a HDF5 file

    Parameters
    ----------
    source: `ctapipe.io.event_source`
    output_filename: str
    obs_id: float, int, str or None
    """
    # Writing histograms
    with HDF5TableWriter(filename=output_filename, group_name="simulation", mode="a", filters=filters) as writer:
        for hist in yield_toplevel_of_type(source.file_, Histograms):
            pass
        # find histogram id 6 (thrown energy)
        thrown = None
        for hist in source.file_.histograms:
            if hist['id'] == 6:
                thrown = hist

        thrown_hist = ThrownEventsHistogram()
        thrown_hist.fill_from_simtel(thrown)
        thrown_hist.obs_id = obs_id
        writer.write('thrown_event_distribution', thrown_hist)


def write_mcheader(event, output_filename, filters=None):
    """
    Write the mcheader from an event container to a HDF5 file

    Parameters
    ----------
    output_filename: str
    event: `ctapipe.io.DataContainer`
    """

    extramc = ExtraMCInfo()
    extramc.prefix = ''  # get rid of the prefix

    with HDF5TableWriter(filename=output_filename, group_name="simulation", mode="a", filters=filters) as writer:
        extramc.obs_id = event.dl0.obs_id
        writer.write("run_config", [extramc, event.mcheader])


def write_array_info(event, output_filename):
    """
    Write the array info to a HDF5 file
        - layout info is writen in '/instrument/subarray/layout'
        - optics info is writen in '/instrument/telescope/optics'
        - camera info is writen in '/instrument/telescope/camera/{camera}' for each camera in the array

    Parameters
    ----------
    event: `ctapipe.io.DataContainer`
    output_filename: str
    """

    serialize_meta = True

    sub = event.inst.subarray
    sub.to_table().write(
        output_filename,
        path="/instrument/subarray/layout",
        serialize_meta=serialize_meta,
        append=True
    )
    sub.to_table(kind='optics').write(
        output_filename,
        path='/instrument/telescope/optics',
        append=True,
        serialize_meta=serialize_meta
    )
    for telescope_type in sub.telescope_types:
        ids = set(sub.get_tel_ids_for_type(telescope_type))
        if len(ids) > 0:  # only write if there is a telescope with this camera
            tel_id = list(ids)[0]
            camera = sub.tel[tel_id].camera
            camera.to_table().write(
                output_filename,
                path=f'/instrument/telescope/camera/{camera}',
                append=True,
                serialize_meta=serialize_meta,
            )


def read_mcheader(filename):
    """
    Read the MCHeaderContainer info from a HD5 file.

    Parameters
    ----------
    filename: path

    Returns
    -------
    `ctapipe.io.containers.MCHeaderContainer`
    """
    with HDF5TableReader(filename=filename) as reader:
        mctab = reader.read('/simulation/run_config', MCHeaderContainer())
        mcheader = next(mctab)
    return mcheader


def read_array_info(filename):
    """
    Read array information from HDF5 file.

    Parameters
    ----------
    filename: path

    Returns
    -------
    dict
    """
    array_info = dict()
    with open_file(filename) as file:
        array_info['layout'] = Table(file.root['/instrument/subarray/layout'].read())
        array_info['optics'] = Table(file.root['/instrument/telescope/optics'].read())
        for camera in file.root['/instrument/telescope/camera/']:
            if type(camera) is tables.table.Table:
                array_info[camera.name] = Table(camera.read())
    return array_info
