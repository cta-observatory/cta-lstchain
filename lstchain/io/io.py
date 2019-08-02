import h5py
import numpy as np
from astropy.table import Table, vstack
import tables
from tables import open_file
import os

import ctapipe
import lstchain
from ctapipe.io import HDF5TableReader
from ctapipe.io.containers import MCHeaderContainer
from ctapipe.io import HDF5TableWriter
from eventio import Histograms
from eventio.search_utils import yield_toplevel_of_type
from .lstcontainers import ThrownEventsHistogram, ExtraMCInfo, MetaData


__all__ = ['read_simu_info_hdf5',
           'read_simu_info_merged_hdf5',
           'get_dataset_keys',
           'write_simtel_energy_histogram',
           'write_mcheader',
           'write_array_info',
           'write_metadata',
           'check_thrown_events_histogram',
           'check_mcheader',
           'check_metadata',
           'read_metadata'
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
        files = [open_file(filename) for filename in filenames_list]
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

    f1 = open_file(file_list[0])
    merge_file = open_file(output_filename, 'w')

    nodes = {}
    for g in groups:
        nodes[g] = f1.copy_node('/', name=g, newparent=merge_file.root, newname=g, recursive=True)


    for filename in file_list[1:]:
        with open_file(filename) as file:
            for k in keys:
                try:
                    merge_file.root[k].append(file.root[k].read())
                except:
                    print("Can't append node {} from file {}".format(k, filename))

    merge_file.close()


# def smart_merge_h5files(file_list, output_filename='merged.h5'):
#     """
#     Merge HDF5 DL1 or DL2 files in a smart way.
#         - stack events tables
#         - check that extra information are the same in all files
#         - update simulation information such as number of simulated events for you
#
#     Parameters
#     ----------
#     file_list: list of paths
#     output_filename: path
#     """
#     assert len(file_list)>1, "The file list is not long enough (len = {})".format(len(file_list))
#
#     events_tables_keys = [k for k in get_dataset_keys(file_list[0]) if 'events' in k]
#     auto_merge_h5files(file_list, output_filename=output_filename, nodes_keys=events_tables_keys)
#     # TODO:
#     #  - check that extra info is coherent in all files.
#     #  - copy extra info
#     #  - update extra info number of simulated shower
#
#     filename0 = file_list[0]
#     array_info0 = read_array_info(filename0)
#     mcheader0 = read_mcheader(filename0)
#     thrown_events_hist0 = read_simtel_energy_histogram(filename0)
#     metadata0 = read_metadata(filename0)
#     for filename in file_list[1:]:
#         mcheader = read_mcheader(filename)
#         check_mcheader(mcheader0, mcheader)
#         mcheader0.num_showers += mcheader.num_showers
#
#         thrown_events_hist = read_simtel_energy_histogram(filename)
#         check_thrown_events_histogram(thrown_events_hist0, read_simtel_energy_histogram(filename))
#         thrown_events_hist0.histogram += thrown_events_hist.histogram
#
#         metadata = read_metadata(filename)
#         check_metadata(metadata0, metadata)
#
#         for ii, table in read_array_info(filename).items():
#             assert (table == array_info0[ii]).all()
#
#     write_mcheader(mcheader0, output_filename)



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


def read_simtel_energy_histogram(filename):
    """
    Read the simtel energy histogram from a HDF5 file.

    Parameters
    ----------
    filename: path

    Returns
    -------
    `lstchain.io.lstcontainers.ThrownEventsHistogram`
    """
    with HDF5TableReader(filename=filename) as reader:
        histtab = reader.read('/simulation/thrown_event_distribution', ThrownEventsHistogram())
        hist = next(histtab)
    return hist


def write_mcheader(mcheader, output_filename, obs_id=None, filters=None):
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
        extramc.obs_id = obs_id
        writer.write("run_config", [extramc, mcheader])


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


def check_mcheader(mcheader1, mcheader2):
    """
    Check that the information in two mcheaders are physically consistent.

    Parameters
    ----------
    mcheader1: `ctapipe.io.containers.MCHeaderContainer`
    mcheader2: `ctapipe.io.containers.MCHeaderContainer`

    Returns
    -------

    """
    assert mcheader1.keys() == mcheader2.keys()
    # It does not matter that the number of simulated showers is the same
    keys = list(mcheader1.keys()).remove('num_showers')

    for k in keys:
        assert mcheader1[k] == mcheader2[k]


def check_thrown_events_histogram(thrown_events_hist1, thrown_events_hist2):
    """
    Check that two ThrownEventsHistogram class are compatible with each other

    Parameters
    ----------
    thrown_events_hist1: `lstchain.io.lstcontainers.ThrownEventsHistogram`
    thrown_events_hist2: `lstchain.io.lstcontainers.ThrownEventsHistogram`
    """
    assert thrown_events_hist1.keys() == thrown_events_hist2.keys()
    # It does not matter that the number of simulated showers is the same
    keys = ['bins_energy', 'bins_core_dist']
    for k in keys:
        assert (thrown_events_hist1[k] == thrown_events_hist2[k]).all()


def write_metadata(source, output_filename, obs_id=None):
    """
    Write metadata to a HDF5 file

    Parameters
    ----------
    source: `ctapipe.io.event_source`
    output_filename: path
    """
    metadata = MetaData()
    metadata.filename = os.path.basename(source.input_url)
    metadata.obs_id = obs_id
    metadata.ctapipe_version = ctapipe.__version__
    metadata.lstchain_version = lstchain.__version__
    # One cannot write strings with ctapipe HDF5Writer and Tables can write only fixed length string
    # So this metadata is written in the file attributes
    with open_file(output_filename, mode='a') as file:
        for k, item in metadata.as_dict().items():
            if k in file.root._v_attrs:
                if type(file.root._v_attrs[k]) is list:
                    attribute = file.root._v_attrs[k]
                    attribute.append(metadata[k])
                    file.root._v_attrs[k] = attribute
                else:
                    attribute = [file.root._v_attrs[k]]
                    attribute.append(metadata[k])
                    file.root._v_attrs[k] = attribute
            else:
                file.root._v_attrs[k] = metadata[k]


def read_metadata(filename):
    """
    Read metadata from a HDF5 file

    Parameters
    ----------
    filename: path
    """
    with HDF5TableReader(filename) as reader:
        metatab = reader.read('/metadata', MetaData())
        metadata = next(metatab)

    return metadata


def check_metadata(metadata1, metadata2):
    """
    Check that to MetaData class are compatible with each other

    Parameters
    ----------
    metadata1: `lstchain.io.MetaData`
    metadata2: `lstchain.io.MetaData`
    """
    assert metadata1.keys() == metadata2.keys()
    keys = ['ctapipe_version', 'lstchain_version']
    for k in keys:
        assert metadata1[k] == metadata2[k]



