import logging
import os
import re
import warnings
from multiprocessing import Pool
from contextlib import ExitStack

import numpy as np
import pandas as pd
import tables
from tables import open_file
from tqdm import tqdm

import astropy.units as u
from astropy.table import Table, vstack, QTable

from ctapipe.containers import SimulationConfigContainer
from ctapipe.instrument import SubarrayDescription
from ctapipe.io import HDF5TableReader, HDF5TableWriter

from eventio import Histograms, EventIOFile
from eventio.search_utils import yield_toplevel_of_type, yield_all_subobjects
from eventio.simtel.objects import History, HistoryConfig

from pyirf.simulations import SimulatedEventsInfo

from .lstcontainers import (
    ExtraMCInfo,
    MetaData,
    ThrownEventsHistogram,
)

log = logging.getLogger(__name__)

__all__ = [
    'add_column_table',
    'add_config_metadata',
    'add_global_metadata',
    'add_source_filenames',
    'auto_merge_h5files',
    'check_mcheader',
    'check_metadata',
    'check_mc_type',
    'check_thrown_events_histogram',
    'copy_h5_nodes',
    'extract_simulation_nsb',
    'extract_observation_time',
    'get_dataset_keys',
    'get_srcdep_assumed_positions',
    'get_srcdep_params',
    'get_stacked_table',
    'global_metadata',
    'merge_dl2_runs',
    'merging_check',
    'parse_cfg_bytestring',
    'read_data_dl2_to_QTable',
    'read_dl2_params',
    'read_mc_dl2_to_QTable',
    'read_metadata',
    'read_simtel_energy_histogram',
    'read_simu_info_hdf5',
    'read_simu_info_merged_hdf5',
    'recursive_copy_node',
    'stack_tables_h5files',
    'write_calibration_data',
    'write_dataframe',
    'write_dl2_dataframe',
    'write_mcheader',
    'write_metadata',
    'write_simtel_energy_histogram',
    'write_subarray_tables',

]

dl1_params_tel_mon_ped_key = "/dl1/event/telescope/monitoring/pedestal"
dl1_params_tel_mon_cal_key = "/dl1/event/telescope/monitoring/calibration"
dl1_params_tel_mon_flat_key = "/dl1/event/telescope/monitoring/flatfield"
dl1_params_lstcam_key = "/dl1/event/telescope/parameters/LST_LSTCam"
dl1_images_lstcam_key = "/dl1/event/telescope/image/LST_LSTCam"
dl2_params_lstcam_key = "/dl2/event/telescope/parameters/LST_LSTCam"
dl1_params_src_dep_lstcam_key = "/dl1/event/telescope/parameters_src_dependent/LST_LSTCam"
dl2_params_src_dep_lstcam_key = "/dl2/event/telescope/parameters_src_dependent/LST_LSTCam"
dl1_likelihood_params_lstcam_key = "/dl1/event/telescope/likelihood_parameters/LST_LSTCam"
dl2_likelihood_params_lstcam_key = "/dl2/event/telescope/likelihood_parameters/LST_LSTCam"

HDF5_ZSTD_FILTERS = tables.Filters(
    complevel=5,  # enable compression, 5 is a good tradeoff between compression and speed
    complib='blosc:zstd',  # compression using blosc/zstd
    fletcher32=True,  # attach a checksum to each chunk for error correction
    bitshuffle=False,  # for BLOSC, shuffle bits for better compression
)


def read_simu_info_hdf5(filename):
    """
    Read simu info from an hdf5 file

    Parameters
    ----------
    filename: str
        path to the HDF5 file

    Returns
    -------
    `ctapipe.containers.SimulationConfigContainer`
    """

    with HDF5TableReader(filename) as reader:
        mc_reader = reader.read("/simulation/run_config", SimulationConfigContainer())
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
        simu_info = file.root["simulation/run_config"]
        colnames = simu_info.colnames
        skip = {"num_showers", "shower_prog_start", "detector_prog_start", "obs_id"}
        for k in filter(lambda k: k not in skip, colnames):
            assert np.all(simu_info[:][k] == simu_info[0][k])
        num_showers = simu_info[:]["num_showers"].sum()

    combined_mcheader = read_simu_info_hdf5(filename)
    combined_mcheader["num_showers"] = num_showers

    for k in combined_mcheader.keys():
        if (
                combined_mcheader[k] is not None
                and combined_mcheader.fields[k].unit is not None
        ):
            combined_mcheader[k] = u.Quantity(
                combined_mcheader[k], combined_mcheader.fields[k].unit
            )

    return combined_mcheader


def get_dataset_keys(h5file):
    """
    Return a list of all dataset keys in a HDF5 file

    Parameters
    ----------
    filename: str - path to the HDF5 file

    Returns
    -------
    list of keys
    """
    # we use exit_stack to make sure we close the h5file again if it
    # was not an already open tables.File
    exit_stack = ExitStack()

    with exit_stack:

        if not isinstance(h5file, tables.File):
            h5file = exit_stack.enter_context(tables.open_file(h5file, 'r'))

        dataset_keys = [
            node._v_pathname
            for node in h5file.root._f_walknodes()
            if not isinstance(node, tables.Group)
        ]


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


def stack_tables_h5files(filenames_list, output_filename="merged.h5", keys=None):
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


def copy_h5_nodes(from_file, to_file, nodes=None):
    '''
    Copy dataset (Table and Array) nodes from ``from_file`` to ``to_file``.

    Parameters
    ----------
    from_file: tables.File
        input h5 file opened with tables
    to_file: tables.File
        output h5 file opened with tables, must be writable
    node_keys: Iterable[str]
        Keys to copy, if None, all Table and Array nodes in ``from_file``
        are copied.
    '''
    if nodes is None:
        keys = set(get_dataset_keys(from_file))
    else:
        keys = set(nodes)

    groups = set()

    with warnings.catch_warnings():
        # when copying nodes, we have no control over names
        # so it does not make sense to warn about them
        warnings.simplefilter('ignore', tables.NaturalNameWarning)

        for k in keys:
            in_node = from_file.root[k]
            parent_path = in_node._v_parent._v_pathname
            name = in_node._v_name
            groups.add(parent_path)

            if isinstance(in_node, tables.Table):
                t = to_file.create_table(
                    parent_path,
                    name,
                    createparents=True,
                    obj=from_file.root[k].read()
                )
                for att in from_file.root[k].attrs._f_list():
                    t.attrs[att] = from_file.root[k].attrs[att]

            elif isinstance(in_node, tables.Array):
                a = to_file.create_array(
                    parent_path,
                    name,
                    createparents=True,
                    obj=from_file.root[k].read()
                )
                for att in from_file.root[k].attrs._f_list():
                    a.attrs[att] = in_node.attrs[att]

    # after copying the datasets, also make sure we copy group metadata
    # of all copied groups
    for k in groups:
        from_node = from_file.root[k]
        to_node = to_file.root[k]
        for attr in from_node._v_attrs._f_list():
            to_node._v_attrs[attr] = from_node._v_attrs[attr]


def auto_merge_h5files(
        file_list,
        output_filename="merged.h5",
        nodes_keys=None,
        merge_arrays=False,
        filters=HDF5_ZSTD_FILTERS,
        progress_bar=True
):
    """
    Automatic merge of HDF5 files.
    A list of nodes keys can be provided to merge only these nodes. If None, all nodes are merged.
    It may be also used to create a new file output_filename from content stored in another file.

    Parameters
    ----------
    file_list: list of path
    output_filename: path
    nodes_keys: list of path
    merge_arrays: bool
    filters
    progress_bar : bool
        Enabling the display of the progress bar during event processing.
    """

    file_list = list(file_list)
    if len(file_list) > 1:
        file_list = merging_check(file_list)

    if nodes_keys is None:
        keys = set(get_dataset_keys(file_list[0]))
    else:
        keys = set(nodes_keys)

    bar = tqdm(total=len(file_list), disable=not progress_bar)
    with open_file(output_filename, 'w', filters=filters) as merge_file:
        with open_file(file_list[0]) as f1:
            copy_h5_nodes(f1, merge_file, nodes=keys)

        bar.update(1)
        for filename in file_list[1:]:
            common_keys = keys.intersection(get_dataset_keys(filename))
            with open_file(filename) as file:
                for k in common_keys:
                    in_node = file.root[k]
                    out_node = merge_file.root[k]
                    try:
                        if isinstance(in_node, tables.table.Table) or merge_arrays:
                            # doing `.astype(out_node.dtype)` fixes an issue
                            # when dtypes do not exactly match but are convertible
                            # https://github.com/cta-observatory/cta-lstchain/issues/671
                            out_node.append(in_node.read().astype(out_node.dtype))
                    except:
                        log.error("Can't append node {} from file {}".format(k, filename))
                        raise
            bar.update(1)

        add_source_filenames(merge_file, file_list)

    # merge global metadata and store source file names
    metadata0 = read_metadata(file_list[0])
    write_metadata(metadata0, output_filename)



def add_source_filenames(h5file, file_list):
    exit_stack = ExitStack()

    with exit_stack:
        if not isinstance(h5file, tables.File):
            h5file = exit_stack.enter_context(tables.open_file(h5file, 'a'))


        # we replace any existing node
        if "/source_filenames" in h5file.root:
            h5file.remove_node("/", "source_filenames", recursive=True)

        file_list = [str(p) for p in file_list]

        sources_group = h5file.create_group("/", "source_filenames", "List of input files")
        h5file.create_array(sources_group, "filenames", file_list, "List of files merged")


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
    if len(file_list) < 2:
        raise ValueError("Need at least two files for merging")

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

            if subarray_info != subarray_info0:
                raise ValueError('Subarrays do not match')

        except ValueError as e:
            log.error(rf"{filename} cannot be merged '¯\_(ツ)_/¯: {e}'")
            mergeable_list.remove(filename)

    return mergeable_list


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
    with HDF5TableWriter(
            filename=output_filename, group_name="simulation", mode="a", filters=filters
    ) as writer:
        writer.meta = metadata
        for hist in yield_toplevel_of_type(source.file_, Histograms):
            pass
        # find histogram id 6 (thrown energy)
        thrown = None
        for hist in source.file_.histograms:
            if hist["id"] == 6:
                thrown = hist

        thrown_hist = ThrownEventsHistogram()
        thrown_hist.fill_from_simtel(thrown)
        thrown_hist.obs_id = obs_id
        if metadata is not None:
            add_global_metadata(thrown_hist, metadata)
        writer.write("thrown_event_distribution", [thrown_hist])


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
        histtab = reader.read(
            "/simulation/thrown_event_distribution", ThrownEventsHistogram()
        )
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
    extramc.prefix = ""  # get rid of the prefix
    if metadata is not None:
        add_global_metadata(mcheader, metadata)
        add_global_metadata(extramc, metadata)

    with HDF5TableWriter(
            filename=output_filename, group_name="simulation", mode="a", filters=filters
    ) as writer:
        extramc.obs_id = obs_id
        writer.write("run_config", [extramc, mcheader])


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
    if mcheader1.keys() != mcheader2.keys():
        different = set(mcheader1.keys()).symmetric_difference(mcheader2.keys())
        raise ValueError(f'MC header keys do not match, differing keys: {different}')

    # It does not matter that the number of simulated showers is the same
    keys = list(mcheader1.keys())
    """keys that don't need to be checked: """
    for k in [
        "num_showers",
        "shower_reuse",
        "detector_prog_start",
        "detector_prog_id",
        "shower_prog_id",
        "shower_prog_start",
    ]:
        if k in keys:
            keys.remove(k)

    for k in keys:
        v1 = mcheader1[k]
        v2 = mcheader2[k]
        if v1 != v2:
            raise ValueError(f'MC headers do not match for key {k}:  {v1!r} / {v2!r}')


def check_thrown_events_histogram(thrown_events_hist1, thrown_events_hist2):
    """
    Check that two ThrownEventsHistogram class are compatible with each other

    Parameters
    ----------
    thrown_events_hist1: `lstchain.io.lstcontainers.ThrownEventsHistogram`
    thrown_events_hist2: `lstchain.io.lstcontainers.ThrownEventsHistogram`
    """
    keys1 = set(thrown_events_hist1.keys())
    keys2 = set(thrown_events_hist2.keys())
    if keys1 != keys2:
        different = keys1.symmetric_difference(keys2)
        raise ValueError(f'Histogram keys do not match, differing keys: {different}')


    # It does not matter that the number of simulated showers is the same
    keys = ["bins_energy", "bins_core_dist"]
    for k in keys:
        if (thrown_events_hist1[k] != thrown_events_hist2[k]).all():
            raise ValueError(f'Key {k} does not match for histograms')


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
    with open_file(output_filename, mode="a") as file:
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
    keys1 = set(metadata1.keys())
    keys2 = set(metadata2.keys())
    if keys1 != keys2:
        different = keys1.symmetric_difference(keys2)
        raise ValueError(f'Metadata keys do not match, differing keys: {different}')

    keys = ["LSTCHAIN_VERSION"]
    for k in keys:
        v1 = metadata1[k]
        v2 = metadata2[k]
        if v1 != v2:
            raise ValueError(f'Metadata does not match for key {k}:  {v1!r} / {v2!r}')


def global_metadata():
    """
    Get global metadata container

    Returns
    -------
    `lstchain.io.lstcontainers.MetaData`
    """
    from ctapipe import __version__ as ctapipe_version
    from ctapipe_io_lst import __version__ as ctapipe_io_lst_version
    from .. import __version__ as lstchain_version

    metadata = MetaData()
    metadata.LSTCHAIN_VERSION = lstchain_version
    metadata.CTAPIPE_VERSION = ctapipe_version
    metadata.CTAPIPE_IO_LST_VERSION = ctapipe_io_lst_version
    metadata.CONTACT = "LST Consortium"

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


def add_config_metadata(container, configuration):
    """
    Add configuration parameters to a container in container.meta.config

    Parameters
    ----------
    container: `ctapipe.containers.Container`
    configuration: config dict
    """
    linted_config = str(configuration)
    linted_config = linted_config.replace("<LazyConfigValue {}>", "None")
    linted_config = re.sub(r"<LazyConfigValue\svalue=(.*?)>", "\\1", linted_config)
    linted_config = re.sub(r"DeferredConfigString\((.*?)\)", "\\1", linted_config)
    linted_config = re.sub(r"PosixPath\((.*?)\)", "\\1", linted_config)
    linted_config = linted_config.replace("\'", "\"")
    linted_config = linted_config.replace("None", "\"None\"")
    linted_config = linted_config.replace("inf", "\"inf\"")
    linted_config = linted_config.replace("True", "true")
    linted_config = linted_config.replace("False", "false")

    container.meta["config"] = linted_config


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

    writer.write(table_name="subarray/trigger", containers=[event.index, event.trigger])


def write_dataframe(dataframe, outfile, table_path, mode="a", index=False, config=None, meta=None):
    """
    Write a pandas dataframe to a HDF5 file using pytables formatting.

    Parameters
    ----------
    dataframe: `pandas.DataFrame`
    outfile: path
    table_path: str
        path to the table to write in the HDF5 file
    config: config metadata
    meta: global metadata
    """
    if not table_path.startswith("/"):
        table_path = "/" + table_path

    with tables.open_file(outfile, mode=mode) as f:
        path, table_name = table_path.rsplit("/", maxsplit=1)

        t = f.create_table(
            path,
            table_name,
            dataframe.to_records(index=index),
            createparents=True,
        )
        if config:
            t.attrs["config"] = config
        if meta:
            for k, item in meta.as_dict().items():
                t.attrs[k] = item


def write_dl2_dataframe(dataframe, outfile, config=None, meta=None):
    """
    Write DL2 dataframe to a HDF5 file

    Parameters
    ----------
    dataframe: `pandas.DataFrame`
    outfile: path
    config: config metadata
    meta: global metadata
    """
    write_dataframe(dataframe, outfile=outfile, table_path=dl2_params_lstcam_key, config=config, meta=meta)


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
    newtable = tables.Table(
        table._v_file.root, "_temp_table", d, filters=table.filters
    )  # new table
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
        recursive_path = os.path.join("/", path_split[0])
        for p in path_split[1:]:
            src_file.copy_node(
                recursive_path,
                name=p,
                newparent=dir_file.root[recursive_path],
                newname=p,
                recursive=False,
            )
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
            table_name="telescope/monitoring/pedestal",
            containers=[mon_index, mon_event.pedestal],
        )

    if new_ff:
        # write calibration container
        writer.write(
            table_name="telescope/monitoring/flatfield",
            containers=[mon_index, mon_event.flatfield],
        )

        # write ff container
        writer.write(
            table_name="telescope/monitoring/calibration",
            containers=[mon_index, mon_event.calibration],
        )


def read_mc_dl2_to_QTable(filename):
    """
    Read MC DL2 files from lstchain and convert into pyirf internal format
    - astropy.table.QTable

    Parameters
    ----------
    filename: path

    Returns
    -------
    `astropy.table.QTable`, `pyirf.simulations.SimulatedEventsInfo`
    """

    # mapping
    name_mapping = {
        "mc_energy": "true_energy",
        "mc_alt": "true_alt",
        "mc_az": "true_az",
        "mc_alt_tel": "pointing_alt",
        "mc_az_tel": "pointing_az",
        "gammaness": "gh_score",
    }

    unit_mapping = {
        "true_energy": u.TeV,
        "reco_energy": u.TeV,
        "pointing_alt": u.rad,
        "pointing_az": u.rad,
        "true_alt": u.rad,
        "true_az": u.rad,
        "reco_alt": u.rad,
        "reco_az": u.rad,
    }

    # add alpha for source-dependent analysis
    srcdep_flag = dl2_params_src_dep_lstcam_key in get_dataset_keys(filename)

    if srcdep_flag:
        unit_mapping['alpha'] = u.deg

    simu_info = read_simu_info_merged_hdf5(filename)
    pyirf_simu_info = SimulatedEventsInfo(
        n_showers=simu_info.num_showers * simu_info.shower_reuse,
        energy_min=simu_info.energy_range_min,
        energy_max=simu_info.energy_range_max,
        max_impact=simu_info.max_scatter_range,
        spectral_index=simu_info.spectral_index,
        viewcone=simu_info.max_viewcone_radius,
    )

    events = pd.read_hdf(filename, key=dl2_params_lstcam_key)

    if srcdep_flag:
        events_srcdep = get_srcdep_params(filename, 'on')
        events = pd.concat([events, events_srcdep], axis=1)

    events = events.rename(columns=name_mapping)

    events = QTable.from_pandas(events)

    for k, v in unit_mapping.items():
        events[k] *= v

    return events, pyirf_simu_info


def read_data_dl2_to_QTable(filename, srcdep_pos=None):
    """
    Read data DL2 files from lstchain and return QTable format
    Parameters
    ----------
    filename: path to the lstchain DL2 file
    srcdep_pos: assumed source position for source-dependent analysis

    Returns
    -------
    `astropy.table.QTable`
    """

    # Mapping
    name_mapping = {
        "gammaness": "gh_score",
        "alt_tel": "pointing_alt",
        "az_tel": "pointing_az",
    }
    unit_mapping = {
        "reco_energy": u.TeV,
        "pointing_alt": u.rad,
        "pointing_az": u.rad,
        "reco_alt": u.rad,
        "reco_az": u.rad,
        "dragon_time": u.s,
    }

    # add alpha for source-dependent analysis
    srcdep_flag = dl2_params_src_dep_lstcam_key in get_dataset_keys(filename)

    if srcdep_flag:
        unit_mapping['alpha'] = u.deg

    data = pd.read_hdf(filename, key=dl2_params_lstcam_key)

    if srcdep_flag:
        data_srcdep = get_srcdep_params(filename, srcdep_pos)
        data = pd.concat([data, data_srcdep], axis=1)

    data = data.rename(columns=name_mapping)

    data = QTable.from_pandas(data)

    # Make the columns as Quantity
    for k, v in unit_mapping.items():
        data[k] *= v

    return data


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
    return pd.to_datetime(t_df.dragon_time.iat[len(t_df) - 1], unit='s') - \
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
        if any(f"Run{run:05}" in filename for run in runs):
            filelist.append(filename)

    df_list = pool.map(
        partial(read_dl2_params, columns_to_read=columns_to_read), filelist
    )

    observation_times = pool.map(extract_observation_time, df_list)

    observation_time = sum([t.total_seconds() for t in observation_times])
    df = pd.concat(df_list)
    return observation_time, df


def get_srcdep_assumed_positions(filename):
    """
    get assumed positions of source-dependent multi index columns

    Parameters
    ----------
    filename: str - path to the HDF5 file

    Returns
    -------
    assumed positions for source-dependent parameters
    """
    dataset_keys = get_dataset_keys(filename)

    if dl2_params_src_dep_lstcam_key in dataset_keys:
        data = pd.read_hdf(filename, key=dl2_params_src_dep_lstcam_key)

    elif dl1_params_src_dep_lstcam_key in dataset_keys:
        data = pd.read_hdf(filename, key=dl1_params_src_dep_lstcam_key)

    else:
        raise IOError('File does not contain source-dependent parameters')

    if not isinstance(data.columns, pd.MultiIndex):
        data.columns = pd.MultiIndex.from_tuples(
            [tuple(col[1:-1].replace('\'', '').replace(' ', '').split(",")) for col in data.columns])

    return data.columns.levels[0]


def get_srcdep_params(filename, wobble_angles=None):
    """
    get srcdep parameter data frame

    Parameters
    ----------
    filename: str - path to the HDF5 file
    wobble_angles: `str` - multi index key corresponding to an expected source position (e.g. 'on', 'off_180')
    If it is not specified, source-dependent parameters with each assumed position are loaded

    Returns
    -------
    `pandas.DataFrame`
    """
    dataset_keys = get_dataset_keys(filename)

    if dl2_params_src_dep_lstcam_key in dataset_keys:
        data = pd.read_hdf(filename, key=dl2_params_src_dep_lstcam_key)

    elif dl1_params_src_dep_lstcam_key in dataset_keys:
        data = pd.read_hdf(filename, key=dl1_params_src_dep_lstcam_key)

    else:
        raise IOError('File does not contain source-dependent parameters')

    if not isinstance(data.columns, pd.MultiIndex):
        data.columns = pd.MultiIndex.from_tuples(
            [tuple(col[1:-1].replace('\'', '').replace(' ', '').split(",")) for col in data.columns])

    if wobble_angles is not None:
        data = data[wobble_angles]

    return data


def parse_cfg_bytestring(bytestring):
    """
    Parse configuration as read by eventio
    :param bytes bytestring: A ``Bytes`` object with configuration data for one parameter
    :return: Tuple in form ``('parameter_name', 'value')``
    """
    line_decoded = bytestring.decode('utf-8').rstrip()
    if 'ECHO' in line_decoded or '#' in line_decoded:
        return None
    line_list = line_decoded.split('%', 1)[0]  # drop comment
    res = re.sub(' +', ' ', line_list).strip().split(' ', 1)  # remove extra whitespaces and split
    return res[0].upper(), res[1]


def extract_simulation_nsb(filename):
    """
    Get current run NSB from configuration in simtel file
    :param str filename: Input file name
    :return array of `float` by tel_id: NSB rate
    """
    nsb = []
    with EventIOFile(filename) as f:
        for o in yield_all_subobjects(f, [History, HistoryConfig]):
            if hasattr(o, 'parse'):
                try:
                    cfg_element = parse_cfg_bytestring(o.parse()[1])
                    if cfg_element is not None:
                        if cfg_element[0] == 'NIGHTSKY_BACKGROUND':
                            nsb.append(float(cfg_element[1].strip('all:')))
                except Exception as e:
                    print('Unexpected end of %s,\n caught exception %s', filename, e)
    return nsb


def check_mc_type(filename):
    """
    Check MC type ('point_like', 'diffuse', 'ring_wobble') based on the viewcone setting
    Parameters
    ----------
    filename:path (DL1/DL2 hdf file)
    Returns
    -------
    string
    """

    simu_info = read_simu_info_merged_hdf5(filename)
    
    min_viewcone = simu_info.min_viewcone_radius.value
    max_viewcone = simu_info.max_viewcone_radius.value

    if max_viewcone == 0.0:
        mc_type = 'point_like'

    elif min_viewcone == 0.0:
        mc_type = 'diffuse'
        
    elif (max_viewcone - min_viewcone) < 0.1:
        mc_type = 'ring_wobble'

    else:
        raise ValueError('mc type cannot be identified')

    return mc_type
            
