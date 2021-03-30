"""This is a module for extracting data from simtelarray and observed
files and calculate image parameters of the events: Hillas parameters,
timing parameters. They can be stored in HDF5 file. The option of saving the
full camera image is also available.

"""
import logging
import os
from copy import deepcopy
from functools import partial
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
import tables
from astropy.table import Table
from ctapipe.image import (
    HillasParameterizationError,
    hillas_parameters,
    tailcuts_clean,
    apply_time_delta_cleaning,
)
from ctapipe.image.morphology import number_of_islands
from ctapipe.instrument import OpticsDescription
from ctapipe.io import EventSource, HDF5TableWriter
from ctapipe.utils import get_dataset_path
from traitlets.config import Config
from ctapipe.calib.camera import CameraCalibrator
from ctapipe.containers import EventType
from . import disp
from . import utils
from .utils import sky_to_camera
from .volume_reducer import apply_volume_reduction
from ..calib.camera import lst_calibration, load_calibrator_from_config
from ..calib.camera.calibration_calculator import CalibrationCalculator
from ..image.muon import analyze_muon_event, tag_pix_thr
from ..image.muon import create_muon_table, fill_muon_event
from ..io import (
    DL1ParametersContainer,
    replace_config,
    standard_config,
    HDF5_ZSTD_FILTERS,
)
from ..io import (
    add_global_metadata,
    global_metadata,
    write_calibration_data,
    write_mcheader,
    write_metadata,
    write_simtel_energy_histogram,
    write_subarray_tables,
)
from ..io.io import add_column_table
from ..io.lstcontainers import ExtraImageInfo, DL1MonitoringEventIndexContainer
from ..paths import parse_r0_filename, run_to_dl1_filename, r0_to_dl1_filename
from ..io.io import dl1_params_lstcam_key

logger = logging.getLogger(__name__)


__all__ = [
    'add_disp_to_parameters_table',
    'get_dl1',
    'r0_to_dl1',
]


cleaning_method = tailcuts_clean


def get_dl1(
        calibrated_event,
        subarray,
        telescope_id,
        dl1_container=None,
        custom_config={},
):
    """
    Return a DL1ParametersContainer of extracted features from a calibrated event.
    The DL1ParametersContainer can be passed to be filled if created outside the function
    (faster for multiple event processing)

    Parameters
    ----------
    calibrated_event: ctapipe event container
    subarray: `ctapipe.instrument.subarray.SubarrayDescription`
    telescope_id: `int`
    dl1_container: DL1ParametersContainer
    custom_config: path to a configuration file
        configuration used for tailcut cleaning
        supersedes the standard configuration

    Returns
    -------
    DL1ParametersContainer
    """

    config = replace_config(standard_config, custom_config)

    # pop delta_time and use_main_island, so we can cleaning_parameters to tailcuts
    cleaning_parameters = config["tailcut"].copy()
    delta_time = cleaning_parameters.pop("delta_time", None)
    use_main_island = cleaning_parameters.pop("use_only_main_island", True)

    dl1_container = DL1ParametersContainer() if dl1_container is None else dl1_container

    dl1 = calibrated_event.dl1.tel[telescope_id]
    telescope = subarray.tel[telescope_id]
    camera_geometry = telescope.camera.geometry

    image = dl1.image
    peak_time = dl1.peak_time

    signal_pixels = cleaning_method(camera_geometry, image, **cleaning_parameters)
    n_pixels = np.count_nonzero(signal_pixels)

    if n_pixels > 0:
        # check the number of islands
        num_islands, island_labels = number_of_islands(camera_geometry, signal_pixels)

        if use_main_island:
            n_pixels_on_island = np.bincount(island_labels)
            n_pixels_on_island[0] = 0  # first island is no-island and should not be considered
            max_island_label = np.argmax(n_pixels_on_island)
            signal_pixels[island_labels != max_island_label] = False

        if delta_time is not None:
            cleaned_pixel_times = peak_time
            # makes sure only signal pixels are used in the time
            # check:
            cleaned_pixel_times[~signal_pixels] = np.nan

            signal_pixels = apply_time_delta_cleaning(
                camera_geometry,
                signal_pixels,
                cleaned_pixel_times,
                min_number_neighbors=1,
                time_limit=delta_time
            )

        # count surviving pixels
        n_pixels = np.count_nonzero(signal_pixels)

        if n_pixels > 0:
            hillas = hillas_parameters(camera_geometry[signal_pixels], image[signal_pixels])

            # Fill container
            dl1_container.fill_hillas(hillas)

            # convert ctapipe's width and length (in m) to deg:
            foclen = subarray.tel[telescope_id].optics.equivalent_focal_length
            width = np.rad2deg(np.arctan2(dl1_container.width, foclen))
            length = np.rad2deg(np.arctan2(dl1_container.length, foclen))
            dl1_container.width = width
            dl1_container.length = length
            dl1_container.wl = dl1_container.width / dl1_container.length

            dl1_container.set_timing_features(camera_geometry[signal_pixels],
                                              image[signal_pixels],
                                              peak_time[signal_pixels],
                                              hillas)
            dl1_container.set_leakage(camera_geometry, image, signal_pixels)
            dl1_container.set_concentration(camera_geometry, image, hillas)
            dl1_container.n_pixels = n_pixels
            dl1_container.n_islands = num_islands
            dl1_container.log_intensity = np.log10(dl1_container.intensity)

    # We set other fields which still make sense for a non-parametrized
    # image:
    dl1_container.set_telescope_info(subarray, telescope_id)

    return dl1_container

def r0_to_dl1(
    input_filename=get_dataset_path('gamma_test_large.simtel.gz'),
    output_filename=None,
    custom_config={},
):
    """
    Chain r0 to dl1
    Save the extracted dl1 parameters in output_filename

    Parameters
    ----------
    input_filename: str
        path to input file, default: `gamma_test_large.simtel.gz`
    output_filename: str or None
        path to output file, defaults to writing dl1 into the current directory
    custom_config: path to a configuration file

    Returns
    -------

    """

    if output_filename is None:
        try:
            run = parse_r0_filename(input_filename)
            output_filename = run_to_dl1_filename(run.tel_id, run.run, run.subrun)
        except ValueError:
            output_filename = r0_to_dl1_filename(Path(input_filename).name)

    if os.path.exists(output_filename):
        raise IOError(str(output_filename) + ' exists, exiting.')

    config = replace_config(standard_config, custom_config)

    custom_calibration = config["custom_calibration"]

    source = EventSource(input_url=input_filename,
                         config=Config(config["source_config"]))
    subarray = source.subarray
    is_simu = source.is_simulation

    metadata = global_metadata(source)
    write_metadata(metadata, output_filename)

    cal_mc = load_calibrator_from_config(config, subarray)

    # minimum number of pe in a pixel to include it
    # in calculation of muon ring time (peak sample):
    min_pe_for_muon_t_calc = 10.

    # Dictionary to store muon ring parameters
    muon_parameters = create_muon_table()

    # all this will be cleaned up in a next PR related to the configuration files
    r1_dl1_calibrator = CameraCalibrator(
        image_extractor_type=config['image_extractor'],
        config=Config(config),
        subarray=subarray
    )

    if not is_simu:

        # Pulse extractor for muon ring analysis. Same parameters (window_width and _shift) as the one for showers, but
        # using GlobalPeakWindowSum, since the signal for the rings is expected to be very isochronous
        r1_dl1_calibrator_for_muon_rings = CameraCalibrator(image_extractor_type = config['image_extractor_for_muons'],
                                                            config=Config(config),subarray = subarray)

        # Component to process interleaved pedestal and flat-fields
        calib_config = Config(config[config['calibration_product']])

        calibration_calculator = CalibrationCalculator.from_name(
            config['calibration_product'],
            config=calib_config,
            subarray=source.subarray
        )

    calibration_index = DL1MonitoringEventIndexContainer()

    dl1_container = DL1ParametersContainer()

    extra_im = ExtraImageInfo()
    extra_im.prefix = ''  # get rid of the prefix

    # Write extra information to the DL1 file
    subarray.to_hdf(output_filename)

    if is_simu:
        write_mcheader(
            source.simulation_config,
            output_filename,
            obs_id=source.obs_ids[0],
            filters=HDF5_ZSTD_FILTERS,
            metadata=metadata,
        )

    with HDF5TableWriter(
        filename=output_filename,
        group_name='dl1/event',
        mode='a',
        filters=HDF5_ZSTD_FILTERS,
        add_prefix=True,
        # overwrite=True,
    ) as writer:

        if is_simu:
            subarray = subarray
            # build a mapping of tel_id back to tel_index:
            # (note this should be part of SubarrayDescription)
            idx = np.zeros(max(subarray.tel_indices) + 1)
            for key, val in subarray.tel_indices.items():
                idx[key] = val

            # the final transform then needs the mapping and the number of telescopes
            tel_list_transform = partial(
                utils.expand_tel_list,
                max_tels=max(subarray.tel) + 1,
            )

            writer.add_column_transform(
                table_name='subarray/trigger',
                col_name='tels_with_trigger',
                transform=tel_list_transform
            )

        # Forcing filters for the dl1 dataset that are currently read from the pre-existing files
        # This should be fixed in ctapipe and then corrected here
        writer._h5file.filters = HDF5_ZSTD_FILTERS
        logger.info(f"USING FILTERS: {writer._h5file.filters}")

        for i, event in enumerate(source):

            if i % 100 == 0:
                logger.info(i)

            event.dl0.prefix = ''
            event.trigger.prefix = ''
            if event.simulation is not None:
                event.simulation.prefix = 'mc'

            dl1_container.reset()

            # write sub tables
            if is_simu:
                write_subarray_tables(writer, event, metadata)
                if not custom_calibration:
                    cal_mc(event)
                if config['mc_image_scaling_factor'] != 1:
                    rescale_dl1_charge(event, config['mc_image_scaling_factor'])

            else:
                if i == 0:
                    # initialize the telescope
                    # FIXME? LST calibrator is only for one telescope
                    # it should be inside the telescope loop (?)

                    tel_id = calibration_calculator.tel_id


                    #initialize the event monitoring data
                    event.mon = deepcopy(source.r0_r1_calibrator.mon_data)

                    # write the first calibration event (initialized from calibration h5 file)
                    write_calibration_data(writer,
                                           calibration_index,
                                           event.mon.tel[tel_id],
                                           new_ped=True, new_ff=True)

                # flat-field or pedestal:
                if (event.trigger.event_type == EventType.FLATFIELD or
                        event.trigger.event_type == EventType.SKY_PEDESTAL):

                    # process interleaved events (pedestals, ff, calibration)
                    new_ped_event, new_ff_event = calibration_calculator.process_interleaved(event)

                    # write monitoring containers if updated
                    if new_ped_event or new_ff_event:
                        write_calibration_data(writer,
                                           calibration_index,
                                           event.mon.tel[tel_id],
                                           new_ped=new_ped_event, new_ff=new_ff_event)

                    # calibrate and gain select the event by hand for DL1
                    source.r0_r1_calibrator.calibrate(event)

            # create image for all events
            r1_dl1_calibrator(event)

            # Temporal volume reducer for lstchain - dl1 level must be filled and dl0 will be overwritten.
            # When the last version of the method is implemented, vol. reduction will be done at dl0
            apply_volume_reduction(event, subarray, config)

            # FIXME? This should be eventually done after we evaluate whether the image is
            # a candidate muon ring. In that case the full image could be kept, or reduced
            # only after the ring analysis is complete.

            for ii, telescope_id in enumerate(event.dl1.tel.keys()):

                focal_length = subarray.tel[telescope_id].optics.equivalent_focal_length
                mirror_area = subarray.tel[telescope_id].optics.mirror_area

                dl1_container.reset()

                # update the calibration index in the dl1 event container
                dl1_container.calibration_id = calibration_index.calibration_id

                dl1_container.fill_event_info(event)

                tel = event.dl1.tel[telescope_id]
                tel.prefix = ''  # don't really need one
                # remove the first part of the tel_name which is the type 'LST', 'MST' or 'SST'
                tel_name = str(subarray.tel[telescope_id])[4:]

                if custom_calibration:
                    lst_calibration(event, telescope_id)

                write_event = True
                # Will determine whether this event has to be written to the
                # DL1 output or not.

                if is_simu:
                    dl1_container.fill_mc(event, subarray.positions[telescope_id])

                assert event.dl1.tel[telescope_id].image is not None

                try:
                    get_dl1(
                        event,
                        subarray,
                        telescope_id,
                        dl1_container=dl1_container,
                        custom_config=config,
                    )

                except HillasParameterizationError:
                    logging.exception(
                        'HillasParameterizationError in get_dl1()'
                    )

                if not is_simu:
                    dl1_container.ucts_time = 0
                    # convert Time to unix timestamp in (UTC) to keep compatibility
                    # with older lstchain
                    # FIXME: just keep it as time, table writer and reader handle it
                    dl1_container.dragon_time = event.trigger.time.unix
                    dl1_container.tib_time = 0

                    dl1_container.ucts_trigger_type = event.lst.tel[telescope_id].evt.ucts_trigger_type
                    dl1_container.trigger_type = event.lst.tel[telescope_id].evt.tib_masked_trigger
                else:
                    dl1_container.trigger_type = event.trigger.event_type

                dl1_container.az_tel = event.pointing.tel[telescope_id].azimuth
                dl1_container.alt_tel = event.pointing.tel[telescope_id].altitude

                dl1_container.trigger_time = event.trigger.time.unix
                dl1_container.event_type = event.trigger.event_type

                dl1_container.prefix = tel.prefix

                # extra info for the image table
                extra_im.tel_id = telescope_id
                extra_im.selected_gain_channel = event.r1.tel[telescope_id].selected_gain_channel

                for container in [extra_im, dl1_container, event.r0, tel]:
                    add_global_metadata(container, metadata)

                event.r0.prefix = ''

                writer.write(table_name=f'telescope/image/{tel_name}',
                             containers=[event.index, tel, extra_im])
                writer.write(table_name=f'telescope/parameters/{tel_name}',
                             containers=[event.index, dl1_container])

                # Muon ring analysis, for real data only (MC is done starting from DL1 files)
                if not is_simu:
                    bad_pixels = event.mon.tel[telescope_id].calibration.unusable_pixels[0]
                    # Set to 0 unreliable pixels:
                    image = tel.image*(~bad_pixels)

                    # process only promising events, in terms of # of pixels with large signals:
                    if tag_pix_thr(image):

                        # re-calibrate r1 to obtain new dl1, using a more adequate pulse integrator for muon rings
                        numsamples = event.r1.tel[telescope_id].waveform.shape[1]  # not necessarily the same as in r0!
                        bad_pixels_hg = event.mon.tel[telescope_id].calibration.unusable_pixels[0]
                        bad_pixels_lg = event.mon.tel[telescope_id].calibration.unusable_pixels[1]
                        # Now set to 0 all samples in unreliable pixels. Important for global peak
                        # integrator in case of crazy pixels!  TBD: can this be done in a simpler
                        # way?
                        bad_pixels = bad_pixels_hg | bad_pixels_lg
                        bad_waveform = np.transpose(np.array(numsamples*[bad_pixels]))

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
                            geom = subarray.tel[telescope_id].\
                                camera.geometry

                            muonintensityparam, dist_mask, \
                            ring_size, size_outside_ring, muonringparam, \
                            good_ring, radial_distribution, \
                            mean_pixel_charge_around_ring,\
                            muonpars = \
                                analyze_muon_event(subarray,
                                                   event.index.event_id,
                                                   image, geom, focal_length,
                                                   mirror_area, False, '')
                            #                      mirror_area, True, './')
                            #           (test) plot muon rings as png files

                            # Now we want to obtain the waveform sample (in HG & LG) at which the ring light peaks:
                            bright_pixels = image > min_pe_for_muon_t_calc
                            selected_gain = event.r1.tel[telescope_id].selected_gain_channel
                            mask_hg = bright_pixels & (selected_gain == 0)
                            mask_lg = bright_pixels & (selected_gain == 1)

                            bright_pixels_waveforms_hg = event.r1.tel[telescope_id].waveform[mask_hg, :]
                            bright_pixels_waveforms_lg = event.r1.tel[telescope_id].waveform[mask_lg, :]
                            stacked_waveforms_hg = np.sum(bright_pixels_waveforms_hg, axis=0)
                            stacked_waveforms_lg = np.sum(bright_pixels_waveforms_lg, axis=0)

                            # stacked waveforms from all bright pixels; shape (ngains, nsamples)
                            hg_peak_sample = np.argmax(stacked_waveforms_hg, axis=-1)
                            lg_peak_sample = np.argmax(stacked_waveforms_lg, axis=-1)

                        if good_ring:
                            fill_muon_event(-1,
                                            muon_parameters,
                                            good_ring,
                                            event.index.event_id,
                                            dl1_container.dragon_time,
                                            muonintensityparam,
                                            dist_mask,
                                            muonringparam,
                                            radial_distribution,
                                            ring_size,
                                            size_outside_ring,
                                            mean_pixel_charge_around_ring,
                                            muonpars,
                                            hg_peak_sample, lg_peak_sample)

                # writes mc information per telescope, including photo electron image
                if (
                    is_simu
                    and config['write_pe_image']
                    and event.simulation.tel[telescope_id].true_image is not None
                    and event.simulation.tel[telescope_id].true_image.any()
                ):
                    event.simulation.tel[telescope_id].prefix = ''
                    writer.write(
                        table_name=f'simulation/{tel_name}',
                        containers=[event.simulation.tel[telescope_id], extra_im]
                    )

        if not is_simu:
            # at the end of event loop ask calculation of remaining interleaved statistics
            new_ped, new_ff = calibration_calculator.output_interleaved_results(event)
            # write monitoring events
            write_calibration_data(writer,
                                   calibration_index,
                                   event.mon.tel[tel_id],
                                   new_ped=new_ped, new_ff=new_ff)

    if is_simu:
        # Reconstruct source position from disp for all events and write the result in the output file
        add_disp_to_parameters_table(output_filename, dl1_params_lstcam_key, focal_length)

        # Write energy histogram from simtel file and extra metadata
        # ONLY of the simtel file has been read until the end, otherwise it seems to hang here forever
        if source.max_events is None:
            write_simtel_energy_histogram(source, output_filename, obs_id=event.index.obs_id,
                                          metadata=metadata)
    else:
        dir, name = os.path.split(output_filename)
        name = name.replace('dl1', 'muons').replace('LST-1.1', 'LST-1')
        # Consider the possibilities of DL1 files with .fits.h5 & .h5 ending:
        name = name.replace('.fits.h5', '.fits').replace('.h5', '.fits')
        muon_output_filename = Path(dir, name)
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
    df = pd.read_hdf(dl1_file, key=table_path)

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


def rescale_dl1_charge(event, scaling_factor):
    """
    Rescale the charges (images) by a given scaling factor.
    The images in dl1.tel[tel_id].image is directly multiplied in place by `scaling_factor`.

    Parameters
    ----------
    event: `ctapipe.containers.ArrayEventContainer`
    scaling_factor: float
    """

    for tel_id, tel in event.dl1.tel.items():
        tel.image *= scaling_factor
