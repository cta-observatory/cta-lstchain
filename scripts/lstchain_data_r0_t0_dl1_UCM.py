'''
 This script process real data from r0 raw files up to
 two DL1a+b h5 separated files. It cannot handle MC data.

 It is just a temporary script that will be deprecated whenever
 the PR #201 (https://github.com/cta-observatory/cta-lstchain/pull/201)
 is merged.
'''

import os
import argparse
import numpy as np
from astropy.time import Time
from datetime import datetime
from traitlets.config.loader import Config
import tables
from ctapipe.image import (
        hillas_parameters,
        tailcuts_clean,
        HillasParameterizationError,
        )
from ctapipe.io import event_source
from lstchain.reco import utils
from ctapipe.image.extractor import *
from ctapipe.io.hdf5tableio import HDF5TableWriter
from lstchain.calib.camera.r0 import LSTR0Corrections
from lstchain.calib.camera.calibrator import LSTCameraCalibrator
from lstchain.io import (
        write_array_info, global_metadata, write_subarray_tables,
        add_global_metadata, write_metadata
        )
from lstchain.io.lstcontainers import ExtraImageInfo
from functools import partial
from lstchain.io import DL1ParametersContainer, standard_config, replace_config
from lstchain.io.config import read_configuration_file
from lstchain.pointing import PointingPosition

parser = argparse.ArgumentParser(description="R0 to DL1a+b")

parser.add_argument("--input_file", '-f', type=str,
                    help="Path to the fitz.fz file.",
                    default="")

parser.add_argument("--outdir", '-o', action='store', type=str,
                    help="Path to the output DL1 file.",
                    default="./dl1_data/")

parser.add_argument("--pedestal_drs4", type=str,
                    help="Path to the DRS4 pedestal file.")

parser.add_argument("--calibration_coeff", type=str,
                    help="Path to the file with calibration coefficients.")

parser.add_argument("--pointing", action='store_true',
                    help="Flag for adding the interpolated pointing")

parser.add_argument("--drive_date", type=str,
                    help="Date to find Drive report as YY_MM_DD")

parser.add_argument("--config", type=str,
                    help="Path to configuration file.")

args = parser.parse_args()


cleaning_method = tailcuts_clean


def get_dl1(calibrated_event, telescope_id, dl1_container=None, custom_config={}):
    """
    Return a DL1ParametersContainer of extracted features from a calibrated event.
    The DL1ParametersContainer can be passed to be filled if created outside the function
    (faster for multiple event processing)

    Parameters
    ----------
    calibrated_event: ctapipe event container
    telescope_id: int
    dl1_container: DL1ParametersContainer
    config_file: path to a configuration file
    configuration used for tailcut cleaning
    superseeds the standard configuration

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
    image = dl1.image[0]  # Get just the first channel, HG (?)
    pulse_time = dl1.pulse_time[0]

    signal_pixels = cleaning_method(camera, image, **cleaning_parameters)

    if image[signal_pixels].sum() > 0:
        hillas = hillas_parameters(camera[signal_pixels], image[signal_pixels])
        # Fill container
        dl1_container.fill_hillas(hillas)
        dl1_container.fill_event_info(calibrated_event)
        dl1_container.set_timing_features(camera[signal_pixels],
                                          image[signal_pixels],
                                          pulse_time[signal_pixels],
                                          hillas)
        dl1_container.set_leakage(camera, image, signal_pixels)
        dl1_container.set_n_islands(camera, signal_pixels)
        dl1_container.set_telescope_info(calibrated_event, telescope_id)

        return dl1_container

    else:
        return None


def r0_to_dl1(input_filename, output_filename=None, custom_config={}):
    """
    Chain r0 to dl1
    Save the extracted dl1a+b parameters into two separated files:
        output_filename_a.h5
        output_filename_b.h5

    Parameters
    ----------
    input_filename: str
        path to input file
    output_filename: str
        path to output file, default:
        `./` + basename(input_filename)
    config_file: path to a configuration file

    Returns
    -------
    """
    if output_filename is None:
        output_filename = (
                'dl1_' + os.path.basename(input_filename).split('.')[2] + '_' +
                         os.path.basename(input_filename).split('.')[3]
                         )

    # Create separate files dl1a and dl1b
    output_filename_dl1a = str(output_filename + "_a.h5")
    output_filename_dl1b = str(output_filename + "_b.h5")

    config = replace_config(standard_config, custom_config)

    source = event_source(input_filename)
    source.allowed_tels = config["allowed_tels"]
    source.max_events = config["max_events"]

    metadata = global_metadata(source)
    write_metadata(metadata, output_filename_dl1a)
    write_metadata(metadata, output_filename_dl1b)

    dl1_container = DL1ParametersContainer()

    # Write extra information to the DL1 file
    first_event = next(iter(source))
    write_array_info(first_event, output_filename_dl1a)
    write_array_info(first_event, output_filename_dl1b)
    extra_im = ExtraImageInfo()
    extra_im.prefix = ''  # get rid of the prefix

    for tel_id in source.allowed_tels:
        start_ntp = first_event.lst.tel[tel_id].svc.date

    subarray = first_event.inst.subarray

    # r0 to r1 calibrator
    r0_r1_calibrator = LSTR0Corrections(
                pedestal_path=args.pedestal_drs4,
                tel_id=1, r1_sample_start=2, r1_sample_end=38)

    # r1 to dl1 calibrator
    charge_config = Config({
       "LocalPeakWindowSum": {
           "window_shift": 4,
           "window_width": 11,
           }
       })

    r1_dl1_calibrator = LSTCameraCalibrator(calibration_path=args.calibration_coeff,
                                  image_extractor="LocalPeakWindowSum",
                                  config=charge_config)

    filters = tables.Filters(
            complevel=5,    # enable compression, with level 0=disabled, 9=max
            complib='blosc:zstd',   # compression using blosc
            fletcher32=True,    # attach a checksum to each chunk for error correction
            bitshuffle=False,   # for BLOSC, shuffle bits for better compression
            )

    if args.pointing:
        pointings = PointingPosition()
        pointings.drive_path = (f'/fefs/home/lapp/DrivePositioning/' +
                               'drive_log_{args.drive_date}.txt')

    # File containing only DL1a data
    with HDF5TableWriter(
            filename=output_filename_dl1a,
        group_name='dl1/event',
        mode='a',
        filters=filters,
        add_prefix=True,
        # overwrite=True
        ) as writer:

        print("USING FILTERS: ", writer._h5file.filters)

        # build a mapping of tel_id back to tel_index:
        # (note this should be part of SubarrayDescription)
        idx = np.zeros(max(subarray.tel_indices) + 1)
        for key, val in subarray.tel_indices.items():
            idx[key] = val

            # the final transform then needs the mapping and the number of telescopes
            tel_list_transform = partial(utils.expand_tel_list,
                    max_tels=len(first_event.inst.subarray.tel) + 1,
                                         )

            writer.add_column_transform(
                        table_name='subarray/trigger',
                        col_name='tels_with_trigger',
                        transform=tel_list_transform
                        )

            for i, event in enumerate(source):
                if i % 100 == 0:
                    print(i)

                event.dl0.prefix = ''
                event.trig.prefix = ''

                write_subarray_tables(writer, event, metadata)

                for ii, telescope_id in enumerate(event.r0.tels_with_data):

                    event.pointing[telescope_id].prefix = ''

                    tel = event.dl1.tel[telescope_id]
                    tel.prefix = ''  # remove the first part of the tel_name
                    tel_name = str(event.inst.subarray.tel[telescope_id])[4:-4]

                    # calibrate r0 --> r1
                    r0_r1_calibrator.calibrate(event)

                    # Get only triggerd event (not pedestal)
                    # if event.r0.tel[telescope_id].trigger_type != 32:
                    '''
                    NB: Given the problems with the TIB for the time being
                    all events are processed regardless its trigger type
                    '''
                    # calibrate r1 --> dl1
                    r1_dl1_calibrator(event)

                    try:
                        dl1_filled = get_dl1(event,
                                telescope_id,
                                dl1_container=dl1_container,
                                custom_config=config)

                    except HillasParameterizationError:
                        logging.exception(
                                'HillasParameterizationError in get_dl1()'
                        )
                        continue

                    if dl1_filled is not None:
                        # Some custom def
                        dl1_container.wl = dl1_container.width / dl1_container.length
                        dl1_container.intensity = np.log10(dl1_container.intensity)

                        foclen = (event.inst.subarray.tel[telescope_id].optics.equivalent_focal_length)
                        width = np.rad2deg(np.arctan2(dl1_container.width, foclen))
                        length = np.rad2deg(np.arctan2(dl1_container.length, foclen))
                        dl1_container.width = width.value
                        dl1_container.length = length.value

                        dl1_container.prefix = tel.prefix

                        extra_im.tel_id = telescope_id

                        for container in [extra_im, dl1_container, event.dl0, tel]:
                            add_global_metadata(container, metadata)

                        writer.write(table_name=f'telescope/image/{tel_name}',
                                containers=[event.dl0, tel, extra_im])

            writer.close()

    # Write DL1b file separately from DL1a
    with HDF5TableWriter(
        filename=output_filename_dl1b,
        group_name='dl1/event',
        mode='a',
        filters=filters,
        add_prefix=True,
        # overwrite=True
        ) as writer:

        print("USING FILTERS: ", writer._h5file.filters)

        # build a mapping of tel_id back to tel_index:
        # (note this should be part of SubarrayDescription)
        idx = np.zeros(max(subarray.tel_indices) + 1)
        for key, val in subarray.tel_indices.items():
            idx[key] = val

            # the final transform then needs the mapping and the number of telescopes
            tel_list_transform = partial(utils.expand_tel_list,
                    max_tels=len(first_event.inst.subarray.tel) + 1,
                                         )

            writer.add_column_transform(
                        table_name='subarray/trigger',
                        col_name='tels_with_trigger',
                        transform=tel_list_transform
                        )

            for i, event in enumerate(source):
                if i % 100 == 0:
                    print(i)

                event.dl0.prefix = ''
                event.trig.prefix = ''

                write_subarray_tables(writer, event, metadata)

                for ii, telescope_id in enumerate(event.r0.tels_with_data):

                    event.pointing[telescope_id].prefix = ''

                    tel = event.dl1.tel[telescope_id]
                    tel.prefix = ''  # remove the first part of the tel_name
                    tel_name = str(event.inst.subarray.tel[telescope_id])[4:-4]

                    # calibrate r0 --> r1
                    r0_r1_calibrator.calibrate(event)

                    # Get only triggerd event (not pedestal)
                    # if event.r0.tel[telescope_id].trigger_type != 32:
                    '''
                    NB: Given the problems with the TIB for the time being
                    all events are processed regardless its trigger type
                    '''

                    # calibrate r1 --> dl1
                    r1_dl1_calibrator(event)

                    try:
                        dl1_filled = get_dl1(event,
                                telescope_id,
                                dl1_container=dl1_container,
                                custom_config=config)

                    except HillasParameterizationError:
                        logging.exception(
                                'HillasParameterizationError in get_dl1()'
                        )
                        continue

                    if dl1_filled is not None:

                        # Some custom def
                        dl1_container.wl = dl1_container.width / dl1_container.length
                        dl1_container.intensity = np.log10(dl1_container.intensity)

                        foclen = (event.inst.subarray.tel[telescope_id].optics.equivalent_focal_length)
                        width = np.rad2deg(np.arctan2(dl1_container.width, foclen))
                        length = np.rad2deg(np.arctan2(dl1_container.length, foclen))
                        dl1_container.width = width.value
                        dl1_container.length = length.value

                        # Get the GPS time from the UCTS. If not available,
                        # it is taken from the TIB and NTP counter
                        ucts_timestamp = event.lst.tel[telescope_id].evt.ucts_timestamp

                        if ucts_timestamp != 0:
                            ns = 1e-9  # to nanosecs
                            utc_time = Time(datetime.utcfromtimestamp(ucts_timestamp * ns))
                        else:
                            ntp_time = start_ntp + event.r0.tel[telescope_id].trigger_time
                            utc_time = Time(datetime.utcfromtimestamp(ntp_time))

                        gps_time = utc_time.gps
                        dl1_container.gps_time = gps_time
                        dl1_container.prefix = tel.prefix
                        extra_im.tel_id = telescope_id

                        if args.pointing:
                            Az, El = pointings.cal_pointingposition(utc_time.unix)
                            event.pointing[telescope_id].azimuth = Az
                            event.pointing[telescope_id].altitude = El

                        for container in [extra_im, dl1_container, event.dl0, tel]:
                            add_global_metadata(container, metadata)

                        if args.pointing:
                            writer.write(table_name=f'telescope/parameters/{tel_name}',
                                         containers=[dl1_container,
                                                     event.pointing[telescope_id]])
                        else:
                            writer.write(table_name=f'telescope/parameters/{tel_name}',
                                         containers=[dl1_container])

            writer.close()


if __name__ == '__main__':
    os.makedirs(args.outdir, exist_ok=True)

    output_filename = (args.outdir +
                       '/dl1_' +
                       os.path.basename(args.input_file).split('.')[2] +
                       '_' +
                       os.path.basename(args.input_file).split('.')[3]
                       )

    config = {}
    if args.config is not None:
        try:
            config = read_configuration_file(args.config)
        except("Custom configuration could not be loaded"):
            pass

    r0_to_dl1(input_filename=args.input_file,
              output_filename=output_filename,
              custom_config=config)
