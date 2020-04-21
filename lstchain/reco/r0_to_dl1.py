"""This is a module for extracting data from simtelarray files and
calculate image parameters of the events: Hillas parameters, timing
parameters. They can be stored in HDF5 file. The option of saving the
full camera image is also available.

"""
import os
import logging
import math
from functools import partial
import numpy as np
from pathlib import Path
import pandas as pd
import tables
import astropy.units as u
from astropy.table import Table
from traitlets.config import Config

from ctapipe.utils import get_dataset_path
from ctapipe.io import event_source, HDF5TableWriter
from ctapipe.instrument import OpticsDescription
from ctapipe.image import (
    hillas_parameters,
    tailcuts_clean,
    HillasParameterizationError,
)
from ctapipe.image.cleaning import number_of_islands
from itertools import chain

from . import utils
from .volume_reducer import apply_volume_reduction
from ..io.lstcontainers import ExtraImageInfo
from ..calib.camera import lst_calibration, load_calibrator_from_config
from ..io import DL1ParametersContainer, standard_config, replace_config
from ..image.muon import analyze_muon_event, tag_pix_thr
from ..image.muon import create_muon_table, fill_muon_event


from ..io import (
    write_simtel_energy_histogram,
    write_mcheader,
    write_array_info,
    global_metadata,
    add_global_metadata,
    write_metadata,
    write_subarray_tables,
)
from ..io.io import add_column_table

from . import disp
from .utils import sky_to_camera
from .utils import unix_tai_to_utc
from ..calib.camera.calibrator import LSTCameraCalibrator
from ..calib.camera.r0 import LSTR0Corrections
from ..pointing import PointingPosition


__all__ = [
    'get_dl1',
    'r0_to_dl1',
    'add_disp_to_parameters_table',
]


cleaning_method = tailcuts_clean


filters = tables.Filters(
    complevel=5,            # enable compression, with level 0=disabled, 9=max
    complib='blosc:zstd',   # compression using blosc
    fletcher32=True,        # attach a checksum to each chunk for error correction
    bitshuffle=False,       # for BLOSC, shuffle bits for better compression
)


def get_dl1(
    calibrated_event,
    telescope_id,
    dl1_container=None,
    custom_config={},
    use_main_island=True,
):
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
    n_pixels = np.count_nonzero(signal_pixels)

    if n_pixels > 0:
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
        dl1_container.set_concentration(camera, image, hillas)
        dl1_container.n_pixels = n_pixels
        dl1_container.n_islands = num_islands
        dl1_container.set_telescope_info(calibrated_event, telescope_id)

        return dl1_container

    else:
        return None


def r0_to_dl1(
    input_filename=get_dataset_path('gamma_test_large.simtel.gz'),
    output_filename=None,
    custom_config={},
    pedestal_path=None,
    calibration_path=None,
    time_calibration_path=None,
    pointing_file_path=None,
    ucts_t0_dragon=math.nan,
    dragon_counter0=math.nan,
    ucts_t0_tib=math.nan,
    tib_counter0=math.nan
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
        Arguments below are just temporal and will be removed whenever UCTS+EvB
        is proved to stably and reliably provide timestamps.
    ucts_t0_dragon: first valid ucts_time
    dragon_counter0: Dragon counter corresponding to ucts_t0_dragon
    ucts_t0_tib: first valid ucts_time for the first valid TIB counter
    tib_counter0: first valid TIB counter

    Returns
    -------

    """
    if output_filename is None:
        if (input_filename.startswith('LST')):
            output_filename = (
                'dl1_' + os.path.basename(input_filename).split('.',5)[0] + '.' 
                + os.path.basename(input_filename).split('.',5)[2] + '.' 
                + os.path.basename(input_filename).split('.',5)[3] + '.h5'
                )
        else:
            p = Path(input_filename)
            output_filename = p.with_name('dl1_' + p.name).with_suffix('.h5')

    if os.path.exists(output_filename):
        raise IOError(output_filename + ' exists, exiting.')

    config = replace_config(standard_config, custom_config)

    custom_calibration = config["custom_calibration"]

    source = event_source(input_filename)

    is_simu = source.metadata['is_simulation']

    source.allowed_tels = config["allowed_tels"]
    if config["max_events"] is not None:
        source.max_events = config["max_events"]

    metadata = global_metadata(source)
    write_metadata(metadata, output_filename)

    cal_mc = load_calibrator_from_config(config)

    # minimum number of pe in a pixel to include it
    # in calculation of muon ring time (peak sample):
    min_pe_for_muon_t_calc = 10.

    # Dictionary to store muon ring parameters
    muon_parameters  = create_muon_table()

    if not is_simu:

        # TODO : add DRS4 calibration config in config file, read it and pass it here
        r0_r1_calibrator = LSTR0Corrections(
            pedestal_path=pedestal_path, tel_id=1,
        )

        # all this will be cleaned up in a next PR related to the configuration files
        r1_dl1_calibrator = LSTCameraCalibrator(
            calibration_path=calibration_path,
            time_calibration_path=time_calibration_path,
            extractor_product=config['image_extractor'],
            gain_threshold=Config(config).gain_selector_config['threshold'],
            config=Config(config),
            allowed_tels=[1],
        )

        # Pulse extractor for muon ring analysis.
        # Same parameters (window_width and _shift) as the one for showers,
        # but using GlobalPeakWindowSum, since the signal for the rings is expected to
        # be very isochronous
        r1_dl1_calibrator_for_muon_rings = LSTCameraCalibrator(
            calibration_path=calibration_path,
            time_calibration_path=time_calibration_path,
            extractor_product=config['image_extractor_for_muons'],
            gain_threshold=Config(config).gain_selector_config['threshold'],
            config=Config(config),
            allowed_tels=[1],
        )

    dl1_container = DL1ParametersContainer()

    if pointing_file_path:
        # Open drive report
        pointings = PointingPosition(drive_path=pointing_file_path)
        drive_data = pointings._read_drive_report()

    extra_im = ExtraImageInfo()
    extra_im.prefix = ''  # get rid of the prefix

    # get the first event to write array info and mc header
    event_iter = iter(source)
    first_event = next(event_iter)

    # Write extra information to the DL1 file
    write_array_info(first_event, output_filename)
    if is_simu:
        write_mcheader(
            first_event.mcheader,
            output_filename,
            obs_id=first_event.r0.obs_id,
            filters=filters,
            metadata=metadata,
        )

    with HDF5TableWriter(
        filename=output_filename,
        group_name='dl1/event',
        mode='a',
        filters=filters,
        add_prefix=True,
        # overwrite=True,
    ) as writer:

        if is_simu:
            subarray = first_event.inst.subarray
            # build a mapping of tel_id back to tel_index:
            # (note this should be part of SubarrayDescription)
            idx = np.zeros(max(subarray.tel_indices) + 1)
            for key, val in subarray.tel_indices.items():
                idx[key] = val

            # the final transform then needs the mapping and the number of telescopes
            tel_list_transform = partial(
                utils.expand_tel_list,
                max_tels=len(first_event.inst.subarray.tel) + 1,
            )

            writer.add_column_transform(
                table_name='subarray/trigger',
                col_name='tels_with_trigger',
                transform=tel_list_transform
            )

        # Forcing filters for the dl1 dataset that are currently read from the pre-existing files
        # This should be fixed in ctapipe and then corrected here
        writer._h5file.filters = filters
        print("USING FILTERS: ", writer._h5file.filters)

        for i, event in enumerate(chain([first_event],  event_iter)):

            if i % 100 == 0:
                print(i)

            event.dl0.prefix = ''
            event.mc.prefix = 'mc'
            event.trig.prefix = ''

            # write sub tables
            if is_simu:
                write_subarray_tables(writer, event, metadata)
                if not custom_calibration:
                    cal_mc(event)

            else:
                r0_r1_calibrator.calibrate(event)
                r1_dl1_calibrator(event)

            # Temporal volume reducer for lstchain - dl1 level must be filled and dl0 will be overwritten.
            # When the last version of the method is implemented, vol. reduction will be done at dl0
            apply_volume_reduction(event, config)
            # FIXME? This should be eventually done after we evaluate whether the image is
            # a candidate muon ring. In that case the full image could be kept, or reduced
            # only after the ring analysis is complete.

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
                    # Log10(Energy) in TeV
                    if is_simu:
                        dl1_container.mc_energy = event.mc.energy.to_value(u.TeV)
                        dl1_container.log_mc_energy = np.log10(event.mc.energy.to_value(u.TeV))
                        dl1_container.fill_mc(event)

                    dl1_container.log_intensity = np.log10(dl1_container.intensity)
                    dl1_container.gps_time = event.trig.gps_time.value

                    if not is_simu:
                        # GPS + WRS + UCTS is now working in its nominal configuration.
                        # These TS are stored into ucts_time container.
                        # TS can be alternatively calculated from the TIB and
                        # Dragon modules counters based on the first valid UCTS TS
                        # as the reference point. For the time being, the three TS
                        # are stored in the DL1 files for checking purposes.

                        ucts_time = event.lst.tel[telescope_id].evt.ucts_timestamp * 1e-9  # secs

                        module_id = 82  # Get counters from the central Dragon module

                        if math.isnan(ucts_t0_dragon) and math.isnan(dragon_counter0) \
                           and math.isnan(ucts_t0_tib) and math.isnan(tib_counter0):
                            # Dragon/TIB timestamps not based on a valid absolute reference timestamp
                            dragon_time = (event.lst.tel[telescope_id].svc.date +
                                event.lst.tel[telescope_id].evt.pps_counter[module_id] +
                                event.lst.tel[telescope_id].evt.tenMHz_counter[module_id] * 10**(-7))

                            tib_time = (event.lst.tel[telescope_id].svc.date +
                                event.lst.tel[telescope_id].evt.tib_pps_counter +
                                event.lst.tel[telescope_id].evt.tib_tenMHz_counter * 10**(-7))

                        else:
                            # Dragon/TIB timestamps based on a valid absolute reference timestamp
                            dragon_time = ((ucts_t0_dragon - dragon_counter0) * 1e-9 +  # secs
                                event.lst.tel[telescope_id].evt.pps_counter[module_id] +
                                event.lst.tel[telescope_id].evt.tenMHz_counter[module_id] * 10**(-7))

                            tib_time = ((ucts_t0_tib - tib_counter0) * 1e-9 +  # secs
                                event.lst.tel[telescope_id].evt.tib_pps_counter +
                                event.lst.tel[telescope_id].evt.tib_tenMHz_counter * 10**(-7))

                        # FIXME: directly use unix_tai format whenever astropy v4.1 is out
                        ucts_time_utc = unix_tai_to_utc(ucts_time)
                        dragon_time_utc = unix_tai_to_utc(dragon_time)
                        tib_time_utc = unix_tai_to_utc(tib_time)

                        dl1_container.ucts_time = ucts_time_utc.unix
                        dl1_container.dragon_time = dragon_time_utc.unix
                        dl1_container.tib_time = tib_time_utc.unix

                        # Select the timestamps to be used for pointing interpolation
                        if config['timestamps_pointing'] == "ucts":
                            event_timestamps = ucts_time_utc.unix
                        elif config['timestamps_pointing'] == "dragon":
                            event_timestamps = dragon_time_utc.unix
                        elif config['timestamps_pointing'] == "tib":
                            event_timestamps = tib_time_utc.unix
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


                        # Until the TIB trigger_type is fully reliable, we also add
                        # the ucts_trigger_type to the data
                        extra_im.ucts_trigger_type = event.lst.tel[telescope_id].evt.ucts_trigger_type


                    # FIXME: no need to read telescope characteristics like foclen for every event!
                    foclen = event.inst.subarray.tel[telescope_id].optics.equivalent_focal_length
                    mirror_area = u.Quantity(event.inst.subarray.tel[telescope_id].optics.mirror_area, u.m**2)
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


                    # Muon ring analysis, for real data only (MC is done starting from DL1 files)
                    if not is_simu:
                        bad_pixels = event.mon.tel[telescope_id].calibration.unusable_pixels[0]
                        # Set to 0 unreliable pixels:
                        image = tel.image*(~bad_pixels)

                        # process only promising events, in terms of # of pixels with large signals:
                        if tag_pix_thr(image):

                            # re-calibrate r1 to obtain new dl1, using a more adequate pulse integrator for muon rings
                            numsamples = event.r1.tel[telescope_id].waveform.shape[2] # not necessarily the same as in r0!
                            bad_pixels_hg = event.mon.tel[telescope_id].calibration.unusable_pixels[0]
                            bad_pixels_lg = event.mon.tel[telescope_id].calibration.unusable_pixels[1]
                            # Now set to 0 all samples in unreliable pixels. Important for global peak
                            # integrator in case of crazy pixels!  TBD: can this be done in a simpler
                            # way?
                            bad_waveform = np.array(([np.transpose(np.array(numsamples*[bad_pixels_hg])),
                                                      np.transpose(np.array(numsamples*[bad_pixels_lg]))]))

                            # print('hg bad pixels:',np.where(bad_pixels_hg))
                            # print('lg bad pixels:',np.where(bad_pixels_lg))

                            event.r1.tel[telescope_id].waveform *= ~bad_waveform
                            r1_dl1_calibrator_for_muon_rings(event)

                            tel = event.dl1.tel[telescope_id]
                            image = tel.image*(~bad_pixels)

                            # Check again: with the extractor for muon rings (most likely GlobalPeakWindowSum)
                            # perhaps the event is no longer promising (e.g. if it has a large time evolution)
                            if not tag_pix_thr(image):
                                good_ring = False
                            else:
                                # read geometry from event.inst. But not needed for every event. FIXME?
                                geom = event.inst.subarray.tel[telescope_id].camera

                                muonintensityparam, size_outside_ring, muonringparam, good_ring, \
                                    radial_distribution, mean_pixel_charge_around_ring = \
                                    analyze_muon_event(event.r0.event_id, image, geom, foclen,
                                                       mirror_area, False, '')
                                #                      mirror_area, True, './') # (test) plot muon rings as png files

                                # Now we want to obtain the waveform sample (in HG and LG) at which the ring light peaks:
                                bright_pixels_waveforms = event.r1.tel[telescope_id].waveform[:,image>min_pe_for_muon_t_calc,:]
                                stacked_waveforms = np.sum(bright_pixels_waveforms, axis=-2)
                                # stacked waveforms from all bright pixels; shape (ngains, nsamples)
                                hg_peak_sample = np.argmax(stacked_waveforms, axis=-1)[0]
                                lg_peak_sample = np.argmax(stacked_waveforms, axis=-1)[1]


                            if good_ring:
                                fill_muon_event(muon_parameters, good_ring, event.r0.event_id, dragon_time,
                                                muonintensityparam, muonringparam, radial_distribution,
                                                size_outside_ring, mean_pixel_charge_around_ring,
                                                hg_peak_sample, lg_peak_sample)


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
        # ONLY of the simtel file has been read until the end, otherwise it seems to hang here forever
        if source.max_events is None:
            write_simtel_energy_histogram(source, output_filename, obs_id = event.dl0.obs_id, 
                                          metadata = metadata)
    else:
        dir = os.path.dirname(output_filename)
        name = os.path.basename(output_filename)
        k = name.find('Run')
        muon_output_filename = name[0:name.find('LST-')+5] + '.' + \
                               name[k:k+13] + '.fits'
    
        muon_output_filename =  dir+'/'+muon_output_filename.replace("dl1", "muons")
        table = Table(muon_parameters)
        table.write(muon_output_filename, format='fits', overwrite=True)


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
