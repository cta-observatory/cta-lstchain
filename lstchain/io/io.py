import h5py
from multiprocessing import Pool
import numpy as np
import pandas as pd
from astropy.table import Table, vstack
import tables
from tables import open_file
import os
import astropy.units as u
import ctapipe
import lstchain
from ctapipe.io import HDF5TableReader
from ctapipe.containers import SimulationConfigContainer
from ctapipe.io import HDF5TableWriter
from eventio import Histograms
from eventio.search_utils import yield_toplevel_of_type
from .lstcontainers import ThrownEventsHistogram, ExtraMCInfo, MetaData
from tqdm import tqdm
#from ctapipe.tools.stage1 import Stage1ProcessorTool
from ctapipe.instrument import OpticsDescription, CameraGeometry, CameraDescription, CameraReadout, \
    TelescopeDescription, SubarrayDescription
from pyirf.simulations import SimulatedEventsInfo
from astropy import table

import logging

log = logging.getLogger(__name__)


__all__ = [
    'read_simu_info_hdf5',
    'read_simu_info_merged_hdf5',
    'get_dataset_keys',
    'write_simtel_energy_histogram',
    'write_mcheader',
    'check_thrown_events_histogram',
    'check_mcheader',
    'check_metadata',
    'read_metadata',
    'auto_merge_h5files',
    'smart_merge_h5files',
    'global_metadata',
    'add_global_metadata',
    'write_subarray_tables',
    'write_metadata',
    'write_dataframe',
    'write_dl2_dataframe',
    'write_calibration_data',
    'read_dl2_to_pyirf',
    'read_dl2_params',
    'extract_observation_time',
    'merge_dl2_runs'
]


dl1_params_tel_mon_ped_key = 'dl1/event/telescope/monitoring/pedestal'
dl1_params_tel_mon_cal_key = '/dl1/event/telescope/monitoring/calibration'
dl1_params_lstcam_key = 'dl1/event/telescope/parameters/LST_LSTCam'
dl1_images_lstcam_key = 'dl1/event/telescope/image/LST_LSTCam'
dl2_params_lstcam_key = 'dl2/event/telescope/parameters/LST_LSTCam'
dl1_params_src_dep_lstcam_key = 'dl1/event/telescope/parameters_src_dependent/LST_LSTCam'
dl2_params_src_dep_lstcam_key = 'dl2/event/telescope/parameters_src_dependent/LST_LSTCam'

HDF5_ZSTD_FILTERS = tables.Filters(
    complevel=5,            # enable compression, 5 is a good tradeoff between compression and speed
    complib='blosc:zstd',   # compression using blosc/zstd
    fletcher32=True,        # attach a checksum to each chunk for error correction
    bitshuffle=False,       # for BLOSC, shuffle bits for better compression
)


def read_simu_info_hdf5(filename):
    """
    Read simu info from an hdf5 file

    Returns
    -------
    `ctapipe.containers.SimulationConfigContainer`
    """

    with HDF5TableReader(filename) as reader:
        mc_reader = reader.read('/simulation/run_config', SimulationConfigContainer())
        mc = next(mc_reader)

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
    `ctapipe.containers.SimulationConfigContainer`

    """
    with open_file(filename) as file:
        simu_info = file.root['simulation/run_config']
        colnames = simu_info.colnames
        skip = {'num_showers', 'shower_prog_start', 'detector_prog_start', 'obs_id'}
        for k in filter(lambda k: k not in skip, colnames):
            assert np.all(simu_info[:][k] == simu_info[0][k])
        num_showers = simu_info[:]['num_showers'].sum()

    combined_mcheader = read_simu_info_hdf5(filename)
    combined_mcheader['num_showers'] = num_showers

    for k in combined_mcheader.keys():
        if combined_mcheader[k] is not None and combined_mcheader.fields[k].unit is not None:
            combined_mcheader[k] = u.Quantity(combined_mcheader[k], combined_mcheader.fields[k].unit)

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


def auto_merge_h5files(file_list, output_filename='merged.h5', nodes_keys=None, merge_arrays=False, filters=HDF5_ZSTD_FILTERS):
    """
    Automatic merge of HDF5 files.
    A list of nodes keys can be provided to merge only these nodes. If None, all nodes are merged.

    Parameters
    ----------
    file_list: list of path
    output_filename: path
    nodes_keys: list of path
    merge_arrays: bool
    filters
    """

    if nodes_keys is None:
        keys = set(get_dataset_keys(file_list[0]))
    else:
        keys = set(nodes_keys)

    bar = tqdm(total=len(file_list))
    with open_file(output_filename, 'w', filters=filters) as merge_file:
        with open_file(file_list[0]) as f1:
            for k in keys:
                if type(f1.root[k]) == tables.table.Table:
                    merge_file.create_table(
                        os.path.join('/', k.rsplit('/', maxsplit=1)[0]),
                        os.path.basename(k),
                        createparents=True,
                        obj=f1.root[k].read()
                    )
                if type(f1.root[k]) == tables.array.Array:
                    if merge_arrays:
                        merge_file.create_earray(
                            os.path.join('/', k.rsplit('/', maxsplit=1)[0]),
                            os.path.basename(k),
                            createparents=True,
                            obj=f1.root[k].read()
                        )
                    else:
                        merge_file.create_array(
                            os.path.join('/', k.rsplit('/', maxsplit=1)[0]),
                            os.path.basename(k),
                            createparents=True,
                            obj=f1.root[k].read()
                        )
        bar.update(1)
        for filename in file_list[1:]:
            common_keys = keys.intersection(get_dataset_keys(filename))
            with open_file(filename) as file:
                for k in common_keys:
                    try:
                        if merge_arrays:
                            merge_file.root[k].append(file.root[k].read())
                        else:
                            if type(file.root[k]) == tables.table.Table:
                                merge_file.root[k].append(file.root[k].read())
                    except:
                        print("Can't append node {} from file {}".format(k, filename))
            bar.update(1)


def merging_check(file_list):
    """
    Check that a list of hdf5 files are compatible for merging regarding:
     - array info
     - metadata
     - MC simu info (only for simulations)
     - MC histograms (only for simulations)

    Parameters
    ----------
    file_list: list of paths to hdf5 files

    Returns
    -------
    list: list of paths of files that can be merged
    """
    assert len(file_list) > 1, "The list of files is too short"
    mergeable_list = file_list.copy()

    first_file = mergeable_list[0]
    subarray_info0 = SubarrayDescription.from_hdf(first_file)
    metadata0 = read_metadata(first_file)

    if subarray_info0.name == "MonteCarloArray":
        mcheader0 = read_simu_info_hdf5(first_file)
        thrown_events_hist0 = read_simtel_energy_histogram(first_file)

    for filename in mergeable_list[1:]:
        try:
            metadata = read_metadata(filename)
            check_metadata(metadata0, metadata)
            subarray_info = SubarrayDescription.from_hdf(filename)

            if subarray_info0.name == "MonteCarloArray":
                mcheader = read_simu_info_hdf5(filename)
                thrown_events_hist = read_simtel_energy_histogram(filename)
                check_mcheader(mcheader0, mcheader)
                check_thrown_events_histogram(thrown_events_hist0, thrown_events_hist)

            assert subarray_info == subarray_info0

        except AssertionError:
            log.exception(f"{filename} cannot be smart merged ¯\_(ツ)_/¯")
            mergeable_list.remove(filename)

    return mergeable_list


def smart_merge_h5files(file_list, output_filename='merged.h5', node_keys=None, merge_arrays=False):
    """
    Check that HDF5 files are compatible for merging and merge them

    Parameters
    ----------
    file_list: list of paths to hdf5 files
    output_filename: path to the merged file
    node_keys
    merge_arrays: bool
    """
    smart_list = merging_check(file_list)
    auto_merge_h5files(smart_list, output_filename, nodes_keys=node_keys, merge_arrays=merge_arrays)

    # Merge metadata
    metadata0 = read_metadata(smart_list[0])
    for file in smart_list[1:]:
        metadata = read_metadata(file)
        check_metadata(metadata0, metadata)
        metadata0.SOURCE_FILENAMES.extend(metadata.SOURCE_FILENAMES)
    write_metadata(metadata0, output_filename)


def write_simtel_energy_histogram(source, output_filename, obs_id=None, filters=HDF5_ZSTD_FILTERS, metadata={}):
    """
    Write the energy histogram from a simtel source to a HDF5 file

    Parameters
    ----------
    source: `ctapipe.io.EventSource`
    output_filename: str
    obs_id: float, int, str or None
    """
    # Writing histograms
    with HDF5TableWriter(filename=output_filename, group_name="simulation", mode="a", filters=filters) as writer:
        writer.meta = metadata
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
        if metadata is not None:
            add_global_metadata(thrown_hist, metadata)
        writer.write('thrown_event_distribution', [thrown_hist])


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


def write_mcheader(mcheader, output_filename, obs_id=None, filters=HDF5_ZSTD_FILTERS, metadata=None):
    """
    Write the mcheader from an event container to a HDF5 file

    Parameters
    ----------
    output_filename: str
    """

    extramc = ExtraMCInfo()
    extramc.prefix = ''  # get rid of the prefix
    if metadata is not None:
        add_global_metadata(mcheader, metadata)
        add_global_metadata(extramc, metadata)

    with HDF5TableWriter(filename=output_filename, group_name="simulation", mode="a", filters=filters) as writer:
        extramc.obs_id = obs_id
        writer.write("run_config", [extramc, mcheader])


def read_single_optics(filename, telescope_name):
    """
    Read a specific telescope optics from a DL1 file

    Parameters
    ----------
    filename: str
    telescope_name: str

    Returns
    -------
    `ctapipe.instrument.optics.OpticsDescription`
    """
    from astropy.units import Quantity
    telescope_optics_path = "/configuration/instrument/telescope/optics"
    telescope_optic_table = Table.read(filename, path=telescope_optics_path)
    row = telescope_optic_table[np.where(telescope_name == telescope_optic_table['name'])[0][0]]
    optics_description = OpticsDescription(
        name=row['name'],
        num_mirrors=row['num_mirrors'],
        equivalent_focal_length=row['equivalent_focal_length'] * telescope_optic_table['equivalent_focal_length'].unit,
        mirror_area=row['mirror_area'] * telescope_optic_table['mirror_area'].unit,
        num_mirror_tiles=Quantity(row['num_mirror_tiles']),
    )
    return optics_description


def read_optics(filename):
    """
    Read all telescope optics from a DL1 file

    Parameters
    ----------
    filename: str

    Returns
    -------
    dictionary of ctapipe.instrument.optics.OpticsDescription by telescope names
    """
    telescope_optics_path = "/configuration/instrument/telescope/optics"
    telescope_optics_table = Table.read(filename, path=telescope_optics_path)
    optics_dict = {}
    for telescope_name in telescope_optics_table['name']:
        optics_dict[telescope_name] = read_single_optics(filename, telescope_name)
    return optics_dict


def read_single_camera_geometry(filename, camera_name):
    """
    Read a specific camera geometry from a DL1 file

    Parameters
    ----------
    filename: str
    camera_name: str

    Returns
    -------
    `ctapipe.instrument.camera.geometry.CameraGeometry`
    """
    camera_geometry_path = f"/configuration/instrument/telescope/camera/geometry_{camera_name}"
    camera_geometry = CameraGeometry.from_table(Table.read(filename, camera_geometry_path))
    return camera_geometry


def read_camera_geometries(filename):
    """
    Read all camera geometries from a DL1 file

    Parameters
    ----------
    filename

    Returns
    -------
    dictionary of `ctapipe.instrument.camera.geometry.CameraGeometry` by camera name
    """
    subarray_layout_path = 'configuration/instrument/subarray/layout'
    camera_geoms = {}
    for camera_name in set(Table.read(filename, path=subarray_layout_path)['camera_type']):
        camera_geoms[camera_name] = read_single_camera_geometry(filename, camera_name)
    return camera_geoms


def read_single_camera_readout(filename, camera_name):
    """
    Read a specific camera readout from a DL1 file

    Parameters
    ----------
    filename: str
    camera_name: str

    Returns
    -------
    `ctapipe.instrument.camera.readout.CameraReadout`
    """
    camera_readout_path = f"/configuration/instrument/telescope/camera/readout_{camera_name}"
    return CameraReadout.from_table(Table.read(filename, path=camera_readout_path))


def read_camera_readouts(filename):
    """
    Read  all camera readouts from a DL1 file

    Parameters
    ----------
    filename: str

    Returns
    -------
    dict of `ctapipe.instrument.camera.description.CameraDescription` by tel_id
    """
    subarray_layout_path = 'configuration/instrument/subarray/layout'
    camera_readouts = {}
    for row in Table.read(filename, path=subarray_layout_path):
        camera_name = row['camera_type']
        camera_readouts[row['tel_id']] = read_single_camera_readout(filename, camera_name)
    return camera_readouts


def read_single_camera_description(filename, camera_name):
    """
    Read a specific camera description from a DL1 file

    Parameters
    ----------
    filename: str
    camera_name: str

    Returns
    -------
    `ctapipe.instrument.camera.description.CameraDescription`
    """
    geom = read_single_camera_geometry(filename, camera_name)
    readout = read_single_camera_readout(filename, camera_name)
    return CameraDescription(camera_name, geometry=geom, readout=readout)


def read_single_telescope_description(filename, telescope_name, telescope_type, camera_name):
    """
    Read a specific telescope description from a DL1 file

    Parameters
    ----------
    filename: str
    telescope_name: str
    camera_name: str

    Returns
    -------
    `ctapipe.instrument.telescope.TelescopeDescription`
    """
    optics = read_single_optics(filename, telescope_name)
    camera_descr = read_single_camera_description(filename, camera_name)
    return TelescopeDescription(telescope_name, telescope_type, optics=optics, camera=camera_descr)


def read_subarray_table(filename):
    """
    Read the subarray as a table from a DL1 file

    Parameters
    ----------
    filename: str

    Returns
    -------
    `astropy.table.table.Table`
    """
    subarray_layout_path = 'configuration/instrument/subarray/layout'
    return Table.read(filename, path=subarray_layout_path)


def read_telescopes_descriptions(filename):
    """
    Read telescopes descriptions from DL1 file

    Parameters
    ----------
    filename: str

    Returns
    -------
    dict of `ctapipe.instrument.telescope.TelescopeDescription` by tel_id
    """
    subarray_table = read_subarray_table(filename)
    descriptions = {}
    for row in subarray_table:
        tel_name = row['name']
        camera_type = row['camera_type']
        optics = read_single_optics(filename, tel_name)
        camera = read_single_camera_description(filename, camera_type)
        descriptions[row['tel_id']] = TelescopeDescription(row['name'], row['type'], optics=optics, camera=camera)
    return descriptions


def read_telescopes_positions(filename):
    """
    Read telescopes positions from DL1 file

    Parameters
    ----------
    filename: str

    Returns
    -------
    dictionary of telescopes positions by tel_id
    """
    subarray_table = read_subarray_table(filename)
    pos_dict = {}
    pos_unit = subarray_table['pos_x'].unit
    for row in subarray_table:
        pos_dict[row['tel_id']] = np.array([row['pos_x'], row['pos_y'], row['pos_z']]) * pos_unit
    return pos_dict


def read_subarray_description(filename, subarray_name='LST-1'):
    """
    Read subarray description from an HDF5 DL1 file

    Parameters
    ----------
    filename: str
    subarray_name : str

    Returns
    -------
    `ctapipe.instrument.subarray.SubarrayDescription`
    """
    tel_pos = read_telescopes_positions(filename)
    tel_descrp = read_telescopes_descriptions(filename)
    return SubarrayDescription(subarray_name, tel_positions=tel_pos, tel_descriptions=tel_descrp)


def check_mcheader(mcheader1, mcheader2):
    """
    Check that the information in two mcheaders are physically consistent.

    Parameters
    ----------
    mcheader1: `ctapipe.containers.SimulationConfigContainer`
    mcheader2: `ctapipe.containers.SimulationConfigContainer`

    Returns
    -------

    """
    assert mcheader1.keys() == mcheader2.keys()
    # It does not matter that the number of simulated showers is the same
    keys = list(mcheader1.keys())
    """keys that don't need to be checked: """
    for k in ['num_showers', 'shower_reuse', 'detector_prog_start', 'detector_prog_id', 'shower_prog_id',
              'shower_prog_start',
              ]:
        if k in keys:
            keys.remove(k)

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


def write_metadata(metadata, output_filename):
    """
    Write metadata to a HDF5 file

    Parameters
    ----------
    metadata: `lstchain.io.MetaData()`
    output_filename: path
    """
    # One cannot write strings with ctapipe HDF5Writer and Tables can write only fixed length string
    # So this metadata is written in the file attributes
    with open_file(output_filename, mode='a') as file:
        for k, item in metadata.as_dict().items():
            if k in file.root._v_attrs and type(file.root._v_attrs) is list:
                attribute = file.root._v_attrs[k].extend(metadata[k])
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
    metadata = MetaData()
    with open_file(filename) as file:
        for k in metadata.keys():
            try:
                metadata[k] = file.root._v_attrs[k]
            except:
                # this ensures retro and forward reading compatibility
                print("Metadata {} does not exist in file {}".format(k, filename))
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
    keys = ['LSTCHAIN_VERSION']
    for k in keys:
        assert metadata1[k] == metadata2[k]


def global_metadata(source):
    """
    Get global metadata container

    Returns
    -------
    `lstchain.io.lstcontainers.MetaData`
    """
    metadata = MetaData()
    metadata.LSTCHAIN_VERSION = lstchain.__version__
    metadata.CTAPIPE_VERSION = ctapipe.__version__
    metadata.CONTACT = 'LST Consortium'
    metadata.SOURCE_FILENAMES.append(os.path.basename(source.input_url))

    return metadata


def add_global_metadata(container, metadata):
    """
    Add global metadata to a container in container.meta

    Parameters
    ----------
    container: `ctapipe.containers.Container`
    metadata: `lstchain.io.lstchainers.MetaData`
    """
    meta_dict = metadata.as_dict()
    for k, item in meta_dict.items():
        container.meta[k] = item


def write_subarray_tables(writer, event, metadata=None):
    """
    Write subarray tables info to a HDF5 file

    Parameters
    ----------
    writer: `ctapipe.io.HDF5Writer`
    event: `ctapipe.containers.ArrayEventContainer`
    metadata: `lstchain.io.lstcontainers.MetaData`
    """
    if metadata is not None:
        add_global_metadata(event.index, metadata)
        add_global_metadata(event.simulation, metadata)
        add_global_metadata(event.trigger, metadata)

    writer.write(table_name="subarray/mc_shower", containers=[event.index, event.simulation])
    writer.write(table_name="subarray/trigger", containers=[event.index, event.trigger])


def write_dataframe(dataframe, outfile, table_path, mode='a', index=False):
    """
    Write a pandas dataframe to a HDF5 file using pytables formatting.

    Parameters
    ----------
    dataframe: `pandas.DataFrame`
    outfile: path
    table_path: str
        path to the table to write in the HDF5 file
    """
    if not table_path.startswith('/'):
        table_path = '/' + table_path

    with tables.open_file(outfile, mode=mode) as f:
        path, table_name = table_path.rsplit('/', maxsplit=1)

        f.create_table(
            path,
            table_name,
            dataframe.to_records(index=index),
            createparents=True,
        )


def write_dl2_dataframe(dataframe, outfile):
    """
    Write DL2 dataframe to a HDF5 file

    Parameters
    ----------
    dataframe: `pandas.DataFrame`
    outfile: path
    """
    write_dataframe(dataframe, outfile=outfile, table_path=dl2_params_lstcam_key)


def add_column_table(table, ColClass, col_label, values):
    """
    Add a column to an pytable Table

    Parameters
    ----------
    table: `tables.table.Table`
    ColClass: `tables.atom.MetaAtom`
    col_label: str
    values: list or `numpy.ndarray`

    Returns
    -------
    `tables.table.Table`
    """
    # Step 1: Adjust table description
    d = table.description._v_colobjects.copy()  # original description
    d[col_label] = ColClass()  # add column

    # Step 2: Create new temporary table:
    newtable = tables.Table(table._v_file.root, '_temp_table', d, filters=table.filters)  # new table
    table.attrs._f_copy(newtable)  # copy attributes
    # Copy table rows, also add new column values:
    for row, value in zip(table, values):
        newtable.append([tuple(list(row[:]) + [value])])
    newtable.flush()

    # Step 3: Move temporary table to original location:
    parent = table._v_parent  # original table location
    name = table._v_name  # original table name
    table.remove()  # remove original table
    newtable.move(parent, name)  # move temporary table to original location

    return newtable


def recursive_copy_node(src_file, dir_file, path):
    """
    Copy a node recursively from a src file to a dir file without copying the tables/arrays in the node

    Parameters
    ----------
    src_file: opened `tables.file.File`
    dir_file: `tables.file.File` opened in writing mode
    path: path to the node in `src_file`

    """
    path_split = path.split('/')
    while '' in path_split:
        path_split.remove('')
    assert len(path_split) > 0
    src_file.copy_node('/',
                       name=path_split[0],
                       newparent=dir_file.root,
                       newname=path_split[0],
                       recursive=False)
    if len(path_split) > 1:
        recursive_path = os.path.join('/', path_split[0])
        for p in path_split[1:]:
            src_file.copy_node(recursive_path,
                               name=p,
                               newparent=dir_file.root[recursive_path],
                               newname=p, recursive=False)
            recursive_path = os.path.join(recursive_path, p)


def write_calibration_data(writer, mon_index, mon_event, new_ped=False, new_ff=False):
    mon_event.pedestal.prefix = ''
    mon_event.flatfield.prefix = ''
    mon_event.calibration.prefix = ''
    mon_index.prefix = ''

    # update index
    if new_ped:
        mon_index.pedestal_id += 1

    if new_ff:
        mon_index.flatfield_id += 1
        mon_index.calibration_id += 1

    if new_ped:
        # write ped container
        writer.write(
            table_name=f'telescope/monitoring/pedestal',
            containers=[mon_index, mon_event.pedestal]
        )

    if new_ff:
        # write calibration container
        writer.write(
            table_name="telescope/monitoring/flatfield",
            containers=[mon_index, mon_event.flatfield]
        )

        # write ff container
        writer.write(
            table_name="telescope/monitoring/calibration",
            containers=[mon_index, mon_event.calibration]
        )


def read_dl2_to_pyirf(filename):
    """
    Read DL2 files from lstchain and convert into pyirf internal format
    Parameters
    ----------
    filename: path
    Returns
    -------
    `astropy.table.QTable`, `pyirf.simulations.SimulatedEventsInfo`
    """

    ## mapping
    name_mapping = {
        'mc_energy': 'true_energy',
        'mc_alt': 'true_alt',
        'mc_az': 'true_az',
        'mc_alt_tel': 'pointing_alt',
        'mc_az_tel': 'pointing_az',
        'gammaness': 'gh_score',
    }

    unit_mapping = {
        'true_energy': u.TeV,
        'reco_energy': u.TeV,
        'pointing_alt': u.rad,
        'pointing_az': u.rad,
        'true_alt': u.rad,
        'true_az': u.rad,
        'reco_alt': u.rad,
        'reco_az': u.rad,
    }

    simu_info = read_simu_info_merged_hdf5(filename)
    pyirf_simu_info = SimulatedEventsInfo(n_showers=simu_info.num_showers * simu_info.shower_reuse,
                                          energy_min=simu_info.energy_range_min,
                                          energy_max=simu_info.energy_range_max,
                                          max_impact=simu_info.max_scatter_range,
                                          spectral_index=simu_info.spectral_index,
                                          viewcone=simu_info.max_viewcone_radius,
                                          )

    events = pd.read_hdf(filename, key=dl2_params_lstcam_key).rename(columns=name_mapping)
    events = table.QTable.from_pandas(events)

    for k, v in unit_mapping.items():
        events[k] *= v

    return events, pyirf_simu_info


def read_dl2_params(t_filename, columns_to_read=None):
    """
    Read specified parameters from a file with DL2 data

    Parameters
    ----------
    t_filename: Input file name
    columns_to_read: List of interesting columns, optional. If None, then all columns will be read

    Returns
    -------
    Pandas dataframe with DL2 data
    """
    if columns_to_read is not None:
        return pd.read_hdf(t_filename, key=dl2_params_lstcam_key)[columns_to_read]
    else:
        return pd.read_hdf(t_filename, key=dl2_params_lstcam_key)


def extract_observation_time(t_df):
    """
    Calculate observation time

    Parameters
    ----------
    pandas.DataFrame t_df: Recorded data

    Returns
    -------
    Observation duration in seconds
    """
    return pd.to_datetime(t_df.dragon_time.iat[len(t_df)-1], unit='s') -\
           pd.to_datetime(t_df.dragon_time.iat[0], unit='s')


def merge_dl2_runs(data_tag, runs, columns_to_read=None, n_process=4):
    """
    Merge the run sequence in a single dataset and extract correct observation time based on first and last event timestamp in each file.

    Parameters
    ----------
    data_tag: lstchain version tag
    runs: List of run numbers
    columns_to_read
    n_process: Number of parallel read processes to use

    Returns
    -------
    Pair (observation time, data)
    """
    from functools import partial
    from glob import glob
    filepath_glob = glob(f'/fefs/aswg/data/real/DL2/*/{data_tag}/*')  # Current format of LST data path

    pool = Pool(n_process)
    filelist = []
    # Create a list of files with matching run numbers
    for filename in filepath_glob:
        if any(f'Run{run:05}' in filename for run in runs):
            filelist.append(filename)

    df_list = pool.map(partial(read_dl2_params, columns_to_read=columns_to_read), filelist)

    observation_times = pool.map(extract_observation_time, df_list)

    observation_time = sum([t.total_seconds() for t in observation_times])
    df = pd.concat(df_list)
    return observation_time, df
