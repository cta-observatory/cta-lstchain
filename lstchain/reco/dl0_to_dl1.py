"""This is a module for extracting data from simtelarray files and
calculate image parameters of the events: Hillas parameters, timing
parameters. They can be stored in HDF5 file. The option of saving the
full camera image is also available.

Usage:

"import calib_dl0_to_dl1"

"""
import os
import logging
import numpy as np
from ctapipe.image import (
    hillas_parameters,
    tailcuts_clean,
    HillasParameterizationError,
)

from ctapipe.utils import get_dataset_path
from ctapipe.io import event_source
from ctapipe.io import HDF5TableWriter
from eventio.simtel.simtelfile import SimTelFile
import math
from . import utils
from .volume_reducer import check_and_apply_volume_reduction
from ..io.lstcontainers import ExtraImageInfo
from ..calib.camera import lst_calibration, load_calibrator_from_config
from ..io import DL1ParametersContainer, standard_config, replace_config
from ctapipe.image.cleaning import number_of_islands

import tables
from functools import partial
from ..io import write_simtel_energy_histogram, write_mcheader, write_array_info, global_metadata
from ..io import add_global_metadata, write_metadata, write_subarray_tables
from ..io.io import add_column_table

import pandas as pd
from . import disp
import astropy.units as u
from .utils import sky_to_camera
from ctapipe.instrument import OpticsDescription
from traitlets.config.loader import Config
from ..calib.camera.calibrator import LSTCameraCalibrator
from ..calib.camera.r0 import LSTR0Corrections
from ..calib.camera.calib import combine_channels
from ..pointing import PointingPosition

__all__ = [
    'get_dl1',
    'r0_to_dl1',
]


cleaning_method = tailcuts_clean


filters = tables.Filters(
    complevel=5,    # enable compression, with level 0=disabled, 9=max
    complib='blosc:zstd',   #  compression using blosc
    fletcher32=True,    # attach a checksum to each chunk for error correction
    bitshuffle=False,   # for BLOSC, shuffle bits for better compression
)


def get_dl1(calibrated_event, telescope_id, dl1_container = None, 
            custom_config = {}, use_main_island = True):
    """
    Return a DL1ParametersContainer of extracted features from a calibrated event.
    The DL1ParametersContainer can be passed to be filled if created outside the function
    (faster for multiple event processing)

    Parameters
    ----------
    calibrated_event: ctapipe event container
    telescope_id: `int`
    dl1_container: DL1ParametersContainer
    custom_config: path to a configuration file
        configuration used for tailcut cleaning
        superseeds the standard configuration
    use_main_island: `bool` Use only the main island 
        to calculate DL1 parameters

    Returns
    -------
    DL1ParametersContainer
    """

    config = replace_config(standard_config, custom_config)
    cleaning_parameters = config["tailcut"]

    dl1_container = DL1ParametersContainer() if dl1_container is None else dl1_container

    tel = calibrated_event.inst.subarray.tels[telescope_id]
    dl1 = calibrated_event.dl1.tel[telescope_id]
    camera = tel.camera

    image = dl1.image
    pulse_time = dl1.pulse_time

    signal_pixels = cleaning_method(camera, image, **cleaning_parameters)

    if image[signal_pixels].sum() > 0:

        # check the number of islands 
        num_islands, island_labels = number_of_islands(camera, signal_pixels)

        if use_main_island:
            n_pixels_on_island = np.zeros(num_islands + 1)

            for iisland in range(1, num_islands + 1):
                n_pixels_on_island[iisland] = np.sum(island_labels == iisland)
              
            max_island_label = np.argmax(n_pixels_on_island)
            signal_pixels[island_labels != max_island_label] = False

        hillas = hillas_parameters(camera[signal_pixels], image[signal_pixels])

        # Fill container
        dl1_container.fill_hillas(hillas)
        dl1_container.fill_event_info(calibrated_event)
        dl1_container.set_mc_core_distance(calibrated_event, telescope_id)
        dl1_container.set_mc_type(calibrated_event)
        dl1_container.set_timing_features(camera[signal_pixels],
                                          image[signal_pixels],
                                          pulse_time[signal_pixels],
                                          hillas)
        dl1_container.set_leakage(camera, image, signal_pixels)
        dl1_container.n_islands = num_islands
        dl1_container.set_telescope_info(calibrated_event, telescope_id)

        return dl1_container

    else:
        return None


def r0_to_dl1(input_filename = get_dataset_path('gamma_test_large.simtel.gz'),
              output_filename = None,
              custom_config = {},
              pedestal_path = None,
              calibration_path = None,
              time_calibration_path = None,
              pointing_file_path = None
              ):
    """
    Chain r0 to dl1
    Save the extracted dl1 parameters in output_filename

    Parameters
    ----------
    input_filename: str
        path to input file, default: `gamma_test_large.simtel.gz`
    output_filename: str
        path to output file, default: `./` + basename(input_filename)
    custom_config: path to a configuration file
    pedestal_path: Path to the DRS4 pedestal file
    calibration_path: Path to the file with calibration constants and
        pedestals
    time_calibration_path: Path to the DRS4 time correction file
    pointing_file_path: path to the Drive log with the pointing information

    Returns
    -------

    """
    if output_filename is None:
        output_filename = (
            'dl1_' + os.path.basename(input_filename).split('.')[0] + '.h5'
        )
    if os.path.exists(output_filename):
        raise AttributeError(output_filename + ' exists, exiting.')

    config = replace_config(standard_config, custom_config)

    custom_calibration = config["custom_calibration"]

    try:
        source = event_source(input_filename, back_seekable=True)
    except:
        # back_seekable might not be available for other sources that eventio
        # TODO for real data: source with calibration file and pointing file
        source = event_source(input_filename)

    is_simu = source.metadata['is_simulation']

    source.allowed_tels = config["allowed_tels"]
    source.max_events = config["max_events"]

    metadata = global_metadata(source)
    write_metadata(metadata, output_filename)

    cal = load_calibrator_from_config(config)

    if not is_simu:
        # TODO : add calibration config in config file, read it and pass it here

        r0_r1_calibrator = LSTR0Corrections(pedestal_path = pedestal_path,
                                            tel_id = 1)

        # all this will be cleaned up in a next PR related to the configuration files
        r1_dl1_calibrator = LSTCameraCalibrator(calibration_path = calibration_path,
                                                time_calibration_path = time_calibration_path,
                                                extractor_product = config['image_extractor'],
                                                gain_threshold = Config(config).gain_selector_config['threshold'],
                                                config = Config(config),
                                                allowed_tels = [1],
                                                )

    dl1_container = DL1ParametersContainer()

    if pointing_file_path:
        # Open drive report
        pointings = PointingPosition()
        pointings.drive_path = pointing_file_path
        drive_data = pointings._read_drive_report()
    
    extra_im = ExtraImageInfo()
    extra_im.prefix = ''  # get rid of the prefix

    event = next(iter(source))

    write_array_info(event, output_filename)
    ### Write extra information to the DL1 file
    if is_simu:
        write_mcheader(event.mcheader, output_filename, obs_id = event.r0.obs_id, 
                       filters = filters, metadata = metadata)
        subarray = event.inst.subarray

    with HDF5TableWriter(filename = output_filename,
                         group_name = 'dl1/event',
                         mode = 'a',
                         filters = filters,
                         add_prefix = True,
                         # overwrite = True,
                         ) as writer:

        print("USING FILTERS: ", writer._h5file.filters)

        if is_simu:
            # build a mapping of tel_id back to tel_index:
            # (note this should be part of SubarrayDescription)
            idx = np.zeros(max(subarray.tel_indices) + 1)
            for key, val in subarray.tel_indices.items():
                idx[key] = val

            # the final transform then needs the mapping and the number of telescopes
            tel_list_transform = partial(utils.expand_tel_list,
                                         max_tels = len(event.inst.subarray.tel) + 1,
                                         )

            writer.add_column_transform(
                table_name = 'subarray/trigger',
                col_name = 'tels_with_trigger',
                transform = tel_list_transform
            )

        ### EVENT LOOP ###
        for i, event in enumerate(source):
            if i % 100 == 0:
                print(i)

            event.dl0.prefix = ''
            event.mc.prefix = 'mc'
            event.trig.prefix = ''

            # write sub tables
            if is_simu:
                write_subarray_tables(writer, event, metadata)

            if not custom_calibration and is_simu:
                cal(event)

            if not is_simu:
                r0_r1_calibrator.calibrate(event)
                r1_dl1_calibrator(event)

            # Temporal volume reducer for lstchain - dl1 level must be filled and dl0 will be overwritten.
            # When the last version of the method is implemented, vol. reduction will be done at dl0
            check_and_apply_volume_reduction(event, config)

            for ii, telescope_id in enumerate(event.r0.tels_with_data):

                tel = event.dl1.tel[telescope_id]
                tel.prefix = ''  # don't really need one
                # remove the first part of the tel_name which is the type 'LST', 'MST' or 'SST'
                tel_name = str(event.inst.subarray.tel[telescope_id])[4:]
                tel_name = tel_name.replace('-003', '')

                if custom_calibration:
                    lst_calibration(event, telescope_id)

                try:
                    dl1_filled = get_dl1(event, telescope_id,
                                         dl1_container = dl1_container,
                                         custom_config = config,
                                         use_main_island = True)

                except HillasParameterizationError:
                    logging.exception(
                        'HillasParameterizationError in get_dl1()'
                    )
                    continue

                if dl1_filled is not None:

                    # Some custom def
                    dl1_container.wl = dl1_container.width / dl1_container.length
                    # Log10(Energy) in GeV
                    if is_simu:
                        dl1_container.mc_energy = event.mc.energy.value
                        dl1_container.log_mc_energy = np.log10(event.mc.energy.value * 1e3)
                        dl1_container.fill_mc(event)

                    dl1_container.log_intensity = np.log10(dl1_container.intensity)
                    dl1_container.gps_time = event.trig.gps_time.value

                    if not is_simu:
                        # GPS + WRS + UCTS is now working in its nominal configuration. 
                        # These TS are stored into ucts_time container.
                        # TS can be alternatively calculated from the TIB/Dragon 
                        # modules counters + NTP time corresponding to the start 
                        # of the run (just for cross-checks). For the time being,
                        # the three TS will be stored in the DL1 files.
                        # This will be deprecated and modified back to uniquely use
                        # gps_time whenever the GPS + WRS + UCTS info reliably and stably
                        # arrives at the EvB side.

                        # gps_time = event.r0.tel[telescope_id].trigger_time

                        ucts_time = event.lst.tel[telescope_id].evt.ucts_timestamp * 1e-9 # nsecs

                        # Get counters from the central Dragon module
                        module_id = 82

                        dragon_time = (
                                event.lst.tel[telescope_id].svc.date +
                                event.lst.tel[telescope_id].evt.pps_counter[module_id] +
                                event.lst.tel[telescope_id].evt.tenMHz_counter[module_id] * 10**(-7))

                        tib_time = (
                                event.lst.tel[telescope_id].svc.date +
                                event.lst.tel[telescope_id].evt.tib_pps_counter +
                                event.lst.tel[telescope_id].evt.tib_tenMHz_counter * 10**(-7))

                        #dl1_container.gps_time = gps_time
                        dl1_container.tib_time = tib_time
                        dl1_container.ucts_time = ucts_time
                        dl1_container.dragon_time = dragon_time

                        # Select the timestamps to be used for pointing interpolation
                        if config['timestamps_pointing'] == "ucts":
                            event_timestamps = ucts_time
                        elif config['timestamps_pointing'] == "dragon":
                            event_timestamps = dragon_time
                        elif config['timestamps_pointing'] == "tib":
                            event_timestamps = tib_time
                        else:
                            raise ValueError("The timestamps_pointing option is not a valid one. \
                                    Try ucts (default), dragon or tib.")

                        if pointing_file_path and event_timestamps > 0:
                            azimuth, altitude = pointings.cal_pointingposition(event_timestamps, drive_data)
                            event.pointing[telescope_id].azimuth = azimuth
                            event.pointing[telescope_id].altitude = altitude
                            dl1_container.az_tel = azimuth
                            dl1_container.alt_tel = altitude
                        else:
                            dl1_container.az_tel = u.Quantity(np.nan, u.rad)
                            dl1_container.alt_tel = u.Quantity(np.nan, u.rad)

                    foclen = event.inst.subarray.tel[telescope_id].optics.equivalent_focal_length
                    width = np.rad2deg(np.arctan2(dl1_container.width, foclen))
                    length = np.rad2deg(np.arctan2(dl1_container.length, foclen))
                    dl1_container.width = width.value
                    dl1_container.length = length.value

                    dl1_container.prefix = tel.prefix

                    extra_im.tel_id = telescope_id
                    extra_im.num_trig_pix = event.r0.tel[telescope_id].num_trig_pix
                    extra_im.trigger_time = event.r0.tel[telescope_id].trigger_time
                    extra_im.trigger_type = event.r0.tel[telescope_id].trigger_type
                    extra_im.trig_pix_id = event.r0.tel[telescope_id].trig_pix_id

                    for container in [extra_im, dl1_container, event.r0, tel]:
                        add_global_metadata(container, metadata)

                    event.r0.prefix = ''

                    writer.write(table_name = f'telescope/image/{tel_name}',
                                 containers = [event.r0, tel, extra_im])
                    writer.write(table_name = f'telescope/parameters/{tel_name}',
                                 containers = [dl1_container, extra_im])

                    # writes mc information per telescope, including photo electron image
                    if is_simu \
                            and (event.mc.tel[telescope_id].photo_electron_image > 0).any() \
                            and config['write_pe_image']:
                        event.mc.tel[telescope_id].prefix = ''
                        writer.write(table_name = f'simulation/{tel_name}',
                                     containers = [event.mc.tel[telescope_id], extra_im]
                                     )

    if is_simu:
        ### Reconstruct source position from disp for all events and write the result in the output file
        for tel_name in ['LST_LSTCam']:
            focal = OpticsDescription.from_name(tel_name.split('_')[0]).equivalent_focal_length
            dl1_params_key = f'dl1/event/telescope/parameters/{tel_name}'
            add_disp_to_parameters_table(output_filename, dl1_params_key, focal)

    # Write energy histogram from simtel file and extra metadata
    if is_simu:
        write_simtel_energy_histogram(source, output_filename, obs_id = event.dl0.obs_id, 
                                      metadata = metadata)


def add_disp_to_parameters_table(dl1_file, table_path, focal):
    """
    Reconstruct the disp parameters and source position from a DL1 parameters table and write the result in the file

    Parameters
    ----------
    dl1_file: HDF5 DL1 file containing the required field in `table_path`:
        - mc_alt
        - mc_az
        - mc_alt_tel
        - mc_az_tel

    table_path: path to the parameters table in the file
    focal: focal of the telescope
    """
    df = pd.read_hdf(dl1_file, key = table_path)
    source_pos_in_camera = sky_to_camera(df.mc_alt.values * u.rad,
                                         df.mc_az.values * u.rad,
                                         focal,
                                         df.mc_alt_tel.values * u.rad,
                                         df.mc_az_tel.values * u.rad,
                                         )
    disp_parameters = disp.disp(df.x.values * u.m,
                                df.y.values * u.m,
                                source_pos_in_camera.x,
                                source_pos_in_camera.y)

    with tables.open_file(dl1_file, mode = "a") as file:
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
