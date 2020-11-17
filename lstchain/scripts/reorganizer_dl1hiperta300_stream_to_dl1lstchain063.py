# !/usr/bin/env python3

# Everything is hardcoded to LST telescopes - because this is just a temporal work that will no longer be used
# in the moment lstchain upgrades to V0.8

import os
import tables
import argparse
import numpy as np
from astropy.table import Table, vstack, join
from astropy.io.misc.hdf5 import write_table_hdf5
# from lst_scripts.reorganize_dl1hiperta_to_dl1lstchain import add_disp_and_mc_type_to_parameters_table
import copy
import pandas as pd
import astropy.units as u
from lstchain.io.io import dl1_params_lstcam_key, dl1_images_lstcam_key, add_column_table
from lstchain.reco.disp import disp
from lstchain.reco.utils import sky_to_camera

parser = argparse.ArgumentParser(description="Re-organize the dl1 `standard` output file from either the "
                                             "hiptecta_r1_to_dl1 or hiperta_r1_dl1 to the lstchain DL1 structure")

parser.add_argument('--infile', '-i',
                    type=str,
                    dest='infile',
                    help='dl1 output file of `hiperta_r0_dl1` to be converted to dl1lstchain_v060',
                    default=None
                    )

parser.add_argument('--outfile', '-o',
                    type=str,
                    dest='outfile',
                    help='Output filename. dl1_reorganized.h5 by default.',
                    default='./dl1v0.6_reorganized.h5'
                    )


def add_disp_and_mc_type_to_parameters_table(dl1_file, table_path):
    """
    HARDCODED function obtained from `lstchain.reco.dl0_to_dl1` because `mc_alt_tel` and `mc_az_tel` are zipped within
    `run_array_direction`.
    1. Reconstruct the disp parameters and source position from a DL1 parameters table and write the result in the file.
    2. Computes mc_type from the name of the file.


    Parameters
    ----------
    dl1_file: HDF5 DL1 file containing the required field in `table_path`:
        - mc_alt
        - mc_az
        - mc_alt_tel
        - mc_az_tel

    table_path: path to the parameters table in the file

    Returns
    -------
        None
    """
    with tables.open_file(dl1_file) as hfile:
        run_array_dir = copy.copy(hfile.root.simulation.run_config.col('run_array_direction')[0])
        # Remember that /telescope has been moved previously
        focal = copy.copy(hfile.root.instrument.telescope.optics.col('equivalent_focal_length')[0])

    df = pd.read_hdf(dl1_file, key=table_path)
    source_pos_in_camera = sky_to_camera(df.mc_alt.values * u.rad,
                                         df.mc_az.values * u.rad,
                                         focal * u.m,
                                         run_array_dir[1] * u.rad,
                                         run_array_dir[0] * u.rad,
                                         )

    disp_parameters = disp(df.x.values * u.m,
                           df.y.values * u.m,
                           source_pos_in_camera.x,
                           source_pos_in_camera.y)

    with tables.open_file(dl1_file, mode="a") as file:
        tab = file.root[table_path]
        add_column_table(tab, tables.Float32Col, 'disp_dx', disp_parameters[0].value)
        tab = file.root[table_path]
        add_column_table(tab, tables.Float32Col, 'disp_dy', disp_parameters[1].value)
        tab = file.root[table_path]
        add_column_table(tab, tables.Float32Col, 'disp_norm', disp_parameters[2].value)
        tab = file.root[table_path]
        add_column_table(tab, tables.Float32Col, 'disp_angle', disp_parameters[3].value)
        tab = file.root[table_path]
        add_column_table(tab, tables.Float32Col, 'disp_sign', disp_parameters[4])
        tab = file.root[table_path]
        add_column_table(tab, tables.Float32Col, 'src_x', source_pos_in_camera.x.value)
        tab = file.root[table_path]
        add_column_table(tab, tables.Float32Col, 'src_y', source_pos_in_camera.y.value)
        tab = file.root[table_path]
        add_column_table(tab, tables.Float32Col, 'mc_alt_tel', np.ones(len(df)) * run_array_dir[1])
        tab = file.root[table_path]
        add_column_table(tab, tables.Float32Col, 'mc_az_tel', np.ones(len(df)) * run_array_dir[0])
        if 'gamma' in dl1_file:
            tab = file.root[table_path]
            add_column_table(tab, tables.Float32Col, 'mc_type', np.zeros(len(df)))
        if 'electron' in dl1_file:
            tab = file.root[table_path]
            add_column_table(tab, tables.Float32Col, 'mc_type', np.ones(len(df)))
        if 'proton' in dl1_file:
            tab = file.root[table_path]
            add_column_table(tab, tables.Float32Col, 'mc_type', 101*np.ones(len(df)))


def stack_and_write_images_table(input_filename, hfile_out, node_dl1_event):
    """
    Stack all the `tel_00X` image tables (in case they exit) and write in the v0.6 file

    Parameters
    input_filename : [ste] input hfile name
    hfile_out : output File pointer
    node_dl1_event : Output hfile (V0.6) dl1.event node pointer
    """
    telescope_node = node_dl1_event.telescope

    imag_per_tels = [Table(table_img.read()) for table_img in telescope_node.images]
    image_table = vstack(imag_per_tels)

    for tab in telescope_node.images:
        hfile_out.remove_node(tab)

    # Todo change names of column `image_mask` to `` ??

    dump_plus_copy_node_to_create_new_table(input_filename,
                                            hfile_out,
                                            image_table,
                                            hfile_out.root.dl1.event.telescope.images,
                                            newname_pointer='LST_LSTCam',
                                            tmp_name='imgsTable')


def stack_and_write_parameters_table(input_filename, hfile_out, node_dl1_event, output_mc_table_pointer):
    """
    Stack all the `tel_00X` parameters tables (of v0.8), change names of the columns and write the table in the
    V0.6 (lstchain like) format

    Parameters
    hfile_out : output File pointer
    node_dl1_event : Output hfile (V0.6) dl1.event node pointer
    output_mc_table_pointer : output subarray node pointer
    """
    telescope_node = node_dl1_event.telescope

    param_per_tels = [Table(table_param.read()) for table_param in telescope_node.parameters]
    parameter_table = vstack(param_per_tels)

    for tab in telescope_node.parameters:
        hfile_out.remove_node(tab)

    parameter_table.rename_column('hillas_intensity', 'intensity')
    parameter_table.rename_column('hillas_x', 'x')
    parameter_table.rename_column('hillas_y', 'y')
    parameter_table.rename_column('hillas_r', 'r')
    parameter_table.rename_column('hillas_phi', 'phi')
    parameter_table.rename_column('hillas_length', 'length')
    parameter_table.rename_column('hillas_width', 'width')
    parameter_table.rename_column('hillas_psi', 'psi')
    parameter_table.rename_column('hillas_skewness', 'skewness')
    parameter_table.rename_column('hillas_kurtosis', 'kurtosis')
    parameter_table.rename_column('timing_slope', 'time_gradient')
    parameter_table.rename_column('timing_intercept', 'intercept')
    parameter_table.rename_column('morphology_num_pixels', 'n_pixels')
    parameter_table.rename_column('morphology_num_islands', 'n_islands')
    parameter_table.add_column(np.log10(parameter_table['intensity']), name='log_intensity')
    parameter_table.add_column(parameter_table['width'] / parameter_table['length'], name='wl')

    # Param table is indeed huge - it contains all the mc_events parameters (from v0.6 !!) too
    if output_mc_table_pointer is not None:
        mc_event_table = Table(output_mc_table_pointer.mc_shower.read())
        mc_event_table.remove_column('obs_id')
        parameter_table = join(parameter_table, mc_event_table, keys='event_id')
        parameter_table.add_column(np.log10(parameter_table['mc_energy']), name='log_mc_energy')

    dump_plus_copy_node_to_create_new_table(input_filename,
                                            hfile_out,
                                            parameter_table,
                                            hfile_out.root.dl1.event.telescope.parameters,
                                            newname_pointer='LST_LSTCam',
                                            tmp_name='paramsTable')


def dump_plus_copy_node_to_create_new_table(input_filename, hfile_out, astropy_table_to_copy, newparent_pointer,
                                            newname_pointer, tmp_name, overwrite=False):
    """
    General function to write an astropy table to a temporal file, and immediately after copy it to the
    output v0.6 hfile.

    Parameters
    input_filename : [ste] input hfile name
    hfile_out : output File pointer
    astropy_table_to_copy : astropy table to be copied
    newparent_pointer : newparent copy_node parameter
    newname_pointer : newname copy_node parameter
    tmp_name : [str] flag to identify the temportal table and make it unique (necessary when simultaneous reorganizers
                are run in the same dir)
    overwrite : overwrite parameter of the copy_node method
    """
    input_filename = input_filename.split('___')[0]
    if tmp_name == '':
        flag_name = 'UNKNOWN'
    else:
        flag_name = tmp_name

    temp_table_name = f'{input_filename}_tmp_table_reoganizer_{flag_name}.h5'
    write_table_hdf5(astropy_table_to_copy, temp_table_name, path='/root')
    temp_table = tables.open_file(temp_table_name, 'r')
    hfile_out.copy_node(temp_table.root.root, newparent=newparent_pointer, newname=newname_pointer, overwrite=overwrite)
    temp_table.close()
    os.remove(temp_table_name)


def rename_mc_shower_colnames(input_filename, hfile_out, event_node, output_mc_table_pointer):
    """
    Rename column names of the `mc_shower` table and dump the table to the v0.6 output hfile.

    Parameters
    input_filename : [ste] input hfile name
    hfile_out : output File pointer
    event_node : root.dl1.event node (of output hfile, so V0.6)
    output_mc_table_pointer : output subarray node pointer
    """
    mc_shower_table = Table(event_node.subarray.mc_shower.read())
    mc_shower_table.rename_column('true_energy', 'mc_energy')
    mc_shower_table.rename_column('true_alt', 'mc_alt')
    mc_shower_table.rename_column('true_az', 'mc_az')
    mc_shower_table.rename_column('true_core_x', 'mc_core_x')
    mc_shower_table.rename_column('true_core_y', 'mc_core_y')
    mc_shower_table.rename_column('true_h_first_int', 'mc_h_first_int')
    mc_shower_table.rename_column('true_x_max', 'mc_x_max')
    mc_shower_table.rename_column('true_shower_primary_id', 'mc_shower_primary_id')

    dump_plus_copy_node_to_create_new_table(input_filename,
                                            hfile_out,
                                            mc_shower_table,
                                            output_mc_table_pointer,
                                            newname_pointer='mc_shower',
                                            tmp_name='mcShowerTable',
                                            overwrite=True)


def create_hfile_out(input_filename, outfile_name, sim_pointer08, config_pointer08, dl1_pointer, filter_pointer):
    """
    Create output hfile (lstchainv0.6 like hdf5 file)

    Parameters
    input_filename : [ste] input hfile name
    outfile_name : [str] output hfile name
    sim_pointer08 : dl1-file_v0.8_simulation pointer
    config_pointer08 : dl1-file_v.08_configuration pointer
    dl1_pointer :  dl1-file_v0.8_dl1 pointer
    filter_pointer : dl1-file_v0.8 filters pointer
    """
    hfile_out = tables.open_file(outfile_name, 'w')
    hfile_out.create_group('/', 'simulation')
    hfile_out.create_group('/', 'dl1')

    if sim_pointer08 is None:
        pass
    else:
        # Simulation node V0.6
        #    /simulation (Group) 'Simulation information of the run'
        #       children := ['mc_event' (Table), 'run_config' (Table), 'thrown_event_distribution' (Table)]
        hfile_out.copy_node(sim_pointer08.service.shower_distribution,
                            newparent=hfile_out.root.simulation,
                            newname='thrown_event_distribution',
                            recursive=True,
                            filters=filter_pointer)
        hfile_out.copy_node(config_pointer08.simulation.run,
                            newparent=hfile_out.root.simulation,
                            newname='run_config',
                            recursive=True,
                            filters=filter_pointer)

    # Instrument node V0.6
    #    --instrument (Group)
    #       +--telescope (Group)
    #       |  +--camera (Group)
    #              +--readout_LSTCam --> copied free, it can be erase.
    #              +--geometry_LSTCAM --> To be renamed to LSTCam
    #       |  `--optics (Table)
    #       `--subarray (Group)
    #          `--layout (Table)
    instrument_node = hfile_out.copy_node(config_pointer08.instrument,
                                          newparent=hfile_out.root,
                                          recursive=True,
                                          filters=filter_pointer)
    hfile_out.rename_node(instrument_node.telescope.camera.geometry_LSTCam, newname='LSTCam')

    # dl1 node V0.6
    #    +--dl1 (Group)
    #       `--event (Group)
    #          +--telescope (Group)
    #             +--image (Group)
    #             `--parameters (Group)
    #          `--subarray (Group)
    #             +--mc_shower (Table)
    #             `--trigger (Table)
    dl1_event_node06 = hfile_out.copy_node(dl1_pointer.event,
                                           newparent=hfile_out.root.dl1,
                                           recursive=True,
                                           filters=filter_pointer)
    # This will only happen on ctapipe, not RTA
    # hfile_out.remove_node(dl1_event_node06.telescope.trigger)  # Table stored twice, remove to avoid problems.

    try:
        subarray_pointer = hfile_out.root.dl1.event.subarray
    except:
        subarray_pointer = None

    if sim_pointer08 is None:
        pass
    else:
        hfile_out.copy_node(sim_pointer08.event.subarray.shower,
                            newparent=subarray_pointer,
                            newname="mc_shower",
                            recursive=True,
                            filters=filter_pointer)
    if subarray_pointer is None and sim_pointer08 is None:
        pass
    else:
        rename_mc_shower_colnames(input_filename,
                                  hfile_out,
                                  dl1_event_node06,
                                  subarray_pointer
                                  )

    stack_and_write_parameters_table(input_filename,
                                     hfile_out,
                                     dl1_event_node06,
                                     subarray_pointer
                                     )

    if 'images' in dl1_event_node06.telescope:
        stack_and_write_images_table(input_filename,
                                     hfile_out,
                                     dl1_event_node06
                                     )

    hfile_out.close()


def main(input_filename, output_filename):
    """
    Conversion from dl1 data model (ctapipe and hiper(CTA)RTA) data model, and convert it to lstchain_v0.6 data mode.

    Parameters
    input_filename : [str] Input filename
    output_filename : [str] Output filename
    """
    hfile = tables.open_file(input_filename, 'r')

    # dl1 v0.8 Pointers
    try:
        simulation_v08 = hfile.root.simulation
    except:
        simulation_v08 = None
    configuration_v08 = hfile.root.configuration
    dl1_v08 = hfile.root.dl1
    filter_v08 = hfile.filters

    create_hfile_out(input_filename, output_filename, simulation_v08, configuration_v08, dl1_v08, filter_v08)

    # Add disp_* and mc_type to the parameters table.
    if simulation_v08 is None:
        pass
    else:
        add_disp_and_mc_type_to_parameters_table(output_filename, 'dl1/event/telescope/parameters/LST_LSTCam')

    hfile.close()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.infile, args.outfile)
