"""This is a module for extracting data from simtelarray and observed
files and calculate image parameters of the events: Hillas parameters,
timing parameters. They can be stored in HDF5 file. The option of saving the
full camera image is also available.

"""
import logging
import os
from copy import deepcopy
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
import tables
from scipy.interpolate import interp1d
from astropy.table import Table
from ctapipe.calib.camera import CameraCalibrator
from ctapipe.containers import EventType
from ctapipe.image import (
    HillasParameterizationError,
    hillas_parameters,
    tailcuts_clean,
)
from ctapipe.image import number_of_islands, apply_time_delta_cleaning
from ctapipe.io import EventSource, HDF5TableWriter, DataWriter 
from ctapipe.utils import get_dataset_path
from traitlets.config import Config
from ctapipe_io_lst.constants import (
    PIXEL_INDEX
)

from . import disp
from .utils import sky_to_camera
from .volume_reducer import apply_volume_reduction
from ..data import NormalizedPulseTemplate
from ..calib.camera import load_calibrator_from_config
from ..calib.camera.calibration_calculator import CalibrationCalculator
from ..image.cleaning import apply_dynamic_cleaning
from ..image.modifier import tune_nsb_on_waveform, calculate_required_additional_nsb
from .reconstructor import TimeWaveformFitter
from ..image.muon import analyze_muon_event, tag_pix_thr
from ..image.muon import create_muon_table, fill_muon_event
from ..io import (
    DL1ParametersContainer,
    DL1LikelihoodParametersContainer,
    replace_config,
    standard_config,
    HDF5_ZSTD_FILTERS,
)
from ..io import (
    add_global_metadata,
    add_config_metadata,
    global_metadata,
    write_calibration_data,
    write_mcheader,
    write_metadata,
    write_simtel_energy_histogram,
    write_subarray_tables,
)

from ..io.io import add_column_table, extract_simulation_nsb, dl1_params_lstcam_key
from ..io.lstcontainers import ExtraImageInfo, DL1MonitoringEventIndexContainer
from ..paths import parse_r0_filename, run_to_dl1_filename, r0_to_dl1_filename
from ..visualization.plot_reconstructor import plot_debug

logger = logging.getLogger(__name__)


__all__ = [
    'add_disp_to_parameters_table',
    'get_dl1',
    'apply_lh_fit',
    'r0_to_dl1',
]


cleaning_method = tailcuts_clean


def setup_writer(writer, subarray, is_simulation):
    '''Setup column transforms and exlusions for the hdf5 writer for dl1'''
    writer.add_column_transform(
        table_name="subarray/trigger",
        col_name="tels_with_trigger",
        transform=subarray.tel_ids_to_mask,
    )
    writer.exclude('subarray/trigger', 'tel')

    # Forcing filters for the dl1 dataset that are currently read from the pre-existing files
    writer.h5file.filters = HDF5_ZSTD_FILTERS
    logger.info(f"USING FILTERS: {writer.h5file.filters}")

    tel_names = {str(tel)[4:] for tel in subarray.telescope_types}

    for tel_name in tel_names:
        # None values in telescope/image table
        writer.exclude(f'telescope/image/{tel_name}', 'parameters')
        # None values in telescope/parameters table
        writer.exclude(f'telescope/parameters/{tel_name}', 'hadroness')
        writer.exclude(f'telescope/parameters/{tel_name}', 'disp_norm')
        writer.exclude(f'telescope/parameters/{tel_name}', 'disp_dx')
        writer.exclude(f'telescope/parameters/{tel_name}', 'disp_dy')
        writer.exclude(f'telescope/parameters/{tel_name}', 'disp_angle')
        writer.exclude(f'telescope/parameters/{tel_name}', 'disp_sign')
        writer.exclude(f'telescope/parameters/{tel_name}', 'disp_miss')
        writer.exclude(f'telescope/parameters/{tel_name}', 'src_x')
        writer.exclude(f'telescope/parameters/{tel_name}', 'src_y')

        if is_simulation:
            writer.exclude(f'telescope/parameters/{tel_name}', 'dragon_time')
            writer.exclude(f'telescope/parameters/{tel_name}', 'ucts_time')
            writer.exclude(f'telescope/parameters/{tel_name}', 'tib_time')
            writer.exclude(f'telescope/parameters/{tel_name}', 'ucts_jump')
            writer.exclude(f'telescope/parameters/{tel_name}', 'ucts_trigger_type')
        else:
            writer.exclude(f'telescope/parameters/{tel_name}', 'mc_energy')
            writer.exclude(f'telescope/parameters/{tel_name}', 'log_mc_energy')
            writer.exclude(f'telescope/parameters/{tel_name}', 'mc_alt')
            writer.exclude(f'telescope/parameters/{tel_name}', 'mc_az')
            writer.exclude(f'telescope/parameters/{tel_name}', 'mc_core_x')
            writer.exclude(f'telescope/parameters/{tel_name}', 'mc_core_y')
            writer.exclude(f'telescope/parameters/{tel_name}', 'mc_h_first_int')
            writer.exclude(f'telescope/parameters/{tel_name}', 'mc_alt_tel')
            writer.exclude(f'telescope/parameters/{tel_name}', 'mc_az_tel')
            writer.exclude(f'telescope/parameters/{tel_name}', 'mc_x_max')
            writer.exclude(f'telescope/parameters/{tel_name}', 'mc_core_distance')


def _camera_distance_to_angle(distance, focal_length):
    '''Convert a distance in the camera plane into an angle

    This should be replaced by calculating image parameters in
    telescope frame.
    '''
    return np.rad2deg(np.arctan(distance / focal_length))


def parametrize_image(image, peak_time, signal_pixels, camera_geometry, focal_length, dl1_container):
    '''
    Calculate image parameters and fill them into ``dl1_container``
    '''

    geom_selected = camera_geometry[signal_pixels]
    image_selectecd = image[signal_pixels]
    hillas = hillas_parameters(geom_selected, image_selectecd)

    # Fill container
    dl1_container.fill_hillas(hillas)

    # convert ctapipe's width and length (in m) to deg:

    for key in ['width', 'width_uncertainty', 'length', 'length_uncertainty']:
        value = getattr(dl1_container, key)
        setattr(dl1_container, key, _camera_distance_to_angle(value, focal_length))

    dl1_container.wl = dl1_container.width / dl1_container.length

    dl1_container.set_timing_features(
        geom_selected,
        image_selectecd,
        peak_time[signal_pixels],
        hillas,
    )
    dl1_container.set_leakage(camera_geometry, image, signal_pixels)
    dl1_container.set_concentration(geom_selected, image_selectecd, hillas)
    dl1_container.log_intensity = np.log10(dl1_container.intensity)


def get_dl1(
    calibrated_event,
    subarray,
    telescope_id,
    dl1_container=None,
    custom_config={}
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

    # pop delta_time and use_main_island, so we can pass cleaning_parameters to
    # tailcuts
    cleaning_parameters = config["tailcut"].copy()
    delta_time = cleaning_parameters.pop("delta_time", None)
    use_main_island = cleaning_parameters.pop("use_only_main_island", True)

    use_dynamic_cleaning = False
    if "apply" in config["dynamic_cleaning"]:
        use_dynamic_cleaning = config["dynamic_cleaning"]["apply"]

    dl1_container = DL1ParametersContainer() if dl1_container is None else dl1_container

    dl1 = calibrated_event.dl1.tel[telescope_id]
    telescope = subarray.tel[telescope_id]
    camera_geometry = telescope.camera.geometry
    optics = telescope.optics

    image = dl1.image
    peak_time = dl1.peak_time

    signal_pixels = cleaning_method(camera_geometry, image, **cleaning_parameters)
    n_pixels = np.count_nonzero(signal_pixels)

    if n_pixels > 0:

        if delta_time is not None:
            signal_pixels = apply_time_delta_cleaning(
                camera_geometry,
                signal_pixels,
                peak_time,
                min_number_neighbors=1,
                time_limit=delta_time
            )

        if use_dynamic_cleaning:
            threshold_dynamic = config['dynamic_cleaning']['threshold']
            fraction_dynamic = config['dynamic_cleaning']['fraction_cleaning_intensity']
            signal_pixels = apply_dynamic_cleaning(image,
                                                   signal_pixels,
                                                   threshold_dynamic,
                                                   fraction_dynamic)

        # check the number of islands
        num_islands, island_labels = number_of_islands(camera_geometry, signal_pixels)
        dl1_container.n_islands = num_islands

        if use_main_island:
            n_pixels_on_island = np.bincount(island_labels)
            n_pixels_on_island[0] = 0  # first island is no-island and should not be considered
            max_island_label = np.argmax(n_pixels_on_island)
            signal_pixels[island_labels != max_island_label] = False

        # count surviving pixels
        n_pixels = np.count_nonzero(signal_pixels)
        dl1_container.n_pixels = n_pixels

        if n_pixels > 0:
            parametrize_image(
                image=image,
                peak_time=peak_time,
                signal_pixels=signal_pixels,
                camera_geometry=camera_geometry,
                focal_length=optics.equivalent_focal_length,
                dl1_container=dl1_container,
            )

    # We set other fields which still make sense for a non-parametrized
    # image:
    dl1_container.set_telescope_info(subarray, telescope_id)

    # save the applied cleaning mask
    calibrated_event.dl1.tel[telescope_id].image_mask = signal_pixels

    return dl1_container


def apply_lh_fit(
        event,
        telescope_id,
        dl1_container,
        fitter
):
    """
    Prepare and performs the extraction of DL1 parameters using a likelihood
    based reconstruction method.

    Parameters
    ----------
    event: ctapipe event container
    telescope_id: int
    dl1_container: DL1ParametersContainer
    fitter: TimeWaveformFitter

    Returns
    -------
    DL1LikelihoodParametersContainer

    """
    # Applied to all physic events
    if event.trigger.event_type == EventType.SUBARRAY:
        # Don't fit if the cleaning used in the seed parametrisation didn't select any pixels
        if dl1_container.n_pixels <= 0:
            lhfit_container = DL1LikelihoodParametersContainer(lhfit_call_status=0)
        else:
            try:
                lhfit_container = fitter(event=event, telescope_id=telescope_id, dl1_container=dl1_container)
            except Exception:
                logger.error("Unexpected error encountered in likelihood reconstruction.")
                raise
    else:
        lhfit_container = DL1LikelihoodParametersContainer(lhfit_call_status=-10)
    return lhfit_container


def r0_to_dl1(
    input_filename=None,
    output_filename=None,
    custom_config={},
):
    """
    Chain r0 to dl1
    Save the extracted dl1 parameters in output_filename

    Parameters
    ----------
    input_filename: str
        path to input file, default is an example simulation file
    output_filename: str or None
        path to output file, defaults to writing dl1 into the current directory
    custom_config: path to a configuration file

    Returns
    -------

    """

    # using None as default and using `get_dataset_path` only inside the function
    # prevents downloading at import time.
    if input_filename is None:
        get_dataset_path('gamma_test_large.simtel.gz')

    if output_filename is None:
        try:
            run = parse_r0_filename(input_filename)
            output_filename = run_to_dl1_filename(run.tel_id, run.run, run.subrun)
        except ValueError:
            output_filename = r0_to_dl1_filename(Path(input_filename).name)

    if os.path.exists(output_filename):
        raise IOError(str(output_filename) + ' exists, exiting.')

    config = replace_config(standard_config, custom_config)

    source = EventSource(input_url=input_filename,
                         config=Config(config["source_config"]))
    subarray = source.subarray
    is_simu = source.is_simulation

    metadata = global_metadata()
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
        calib_config = Config({config['calibration_product']: config[config['calibration_product']]})
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
            source.simulation_config[source.obs_ids[0]],
            output_filename,
            obs_id=source.obs_ids[0],
            filters=HDF5_ZSTD_FILTERS,
            metadata=metadata,
        )
    nsb_tuning = False
    nsb_tuning_args = None
    if 'waveform_nsb_tuning' in config.keys():
        nsb_tuning = config['waveform_nsb_tuning']['nsb_tuning']
        if nsb_tuning:
            if is_simu:
                nsb_original = extract_simulation_nsb(input_filename)
                pulse_template = NormalizedPulseTemplate.load_from_eventsource(
                    subarray.tel[1].camera.readout, resample=True
                )
                if 'nsb_tuning_ratio' in config['waveform_nsb_tuning'].keys():
                    # get value from config to possibly extract it beforehand on multiple files for averaging purposes
                    # or gain time
                    nsb_tuning_ratio = config['waveform_nsb_tuning']['nsb_tuning_ratio']
                else:
                    # extract the pedestal variance difference between the current MC file and the target data
                    # FIXME? fails for multiple telescopes
                    nsb_tuning_ratio = calculate_required_additional_nsb(input_filename,
                                                                         config['waveform_nsb_tuning']['target_data'],
                                                                         config=config)[0]
                spe = np.loadtxt(config['waveform_nsb_tuning']['spe_location']).T
                spe_integral = np.cumsum(spe[1])
                charge_spe_cumulative_pdf = interp1d(spe_integral, spe[0], kind='cubic',
                                                     bounds_error=False, fill_value=0.,
                                                     assume_sorted=True)
                allowed_tel = np.zeros(len(nsb_original), dtype=bool)
                allowed_tel[np.array(config['source_config']['LSTEventSource']['allowed_tels'])] = True
                logger.info('Tuning NSB on MC waveform from '
                            + str(np.asarray(nsb_original)[allowed_tel])
                            + 'GHz to {0:d}%'.format(int(nsb_tuning_ratio * 100 + 100.5))
                            + ' for telescopes ids ' + str(config['source_config']['LSTEventSource']['allowed_tels']))
                nsb_tuning_args = [nsb_tuning_ratio, nsb_original, pulse_template, charge_spe_cumulative_pdf]
            else:
                logger.warning('NSB tuning on waveform active in config but file is real data, option will be ignored')
                nsb_tuning = False

    lhfit_fitter = None
    if 'lh_fit_config' in config.keys():
        lhfit_fitter_config = {'TimeWaveformFitter': config['lh_fit_config']}
        lhfit_fitter = TimeWaveformFitter(subarray=subarray, config=Config(lhfit_fitter_config))
        if lhfit_fitter_config['TimeWaveformFitter']['use_interleaved']:
            tmp_source = EventSource(input_url=input_filename,
                                     config=Config(config["source_config"]))
            if is_simu:
                lhfit_fitter.get_ped_from_true_signal_less(tmp_source, nsb_tuning_args)
            else:
                lhfit_fitter.get_ped_from_interleaved(tmp_source)
            del tmp_source

    # initialize the writer of the interleaved events 
    interleaved_writer = None
    if 'write_interleaved_events' in config and not is_simu:
        interleaved_writer_config = Config(config['write_interleaved_events'])
        dir, name = os.path.split(output_filename)

        # create output dir in the data-tree if necessary
        dir = f"{dir}/interleaved"
        os.makedirs(dir, exist_ok=True)
        if 'dl1' in name: 
            name = name.replace('dl1', 'interleaved').replace('LST-1.1', 'LST-1')
        else:
            name = f"interleaved_{name}"
        interleaved_output_file = Path(dir, name)
        interleaved_writer = DataWriter(event_source=source,output_path=interleaved_output_file,config=interleaved_writer_config)
        interleaved_writer._writer.exclude("/r1/event/telescope/.*", "selected_gain_channel")

    with HDF5TableWriter( 
        filename=output_filename,
        group_name='dl1/event',
        mode='a',
        filters=HDF5_ZSTD_FILTERS,
        add_prefix=True,
        # overwrite=True,
    ) as writer:

        setup_writer(writer, source.subarray, is_simulation=is_simu)

        event = None
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
                    for container in [event.mon.tel[tel_id].pedestal, event.mon.tel[tel_id].flatfield, event.mon.tel[tel_id].calibration]:
                        add_global_metadata(container, metadata)
                        add_config_metadata(container, config)

                    # write the first calibration event (initialized from calibration h5 file)
                    # TODO: these data are supposed to change table_path with "dl1/monitoring/telescope/CatA" in short future
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
                    # these data a supposed to be replaced by the Cat_B data in a short future
                    if new_ped_event or new_ff_event:
                        write_calibration_data(writer,
                                           calibration_index,
                                           event.mon.tel[tel_id],
                                           new_ped=new_ped_event, new_ff=new_ff_event)
                    
                    # write the calibrated R1 waveform without gain selection
                    source.r0_r1_calibrator.select_gain = False
                    source.r0_r1_calibrator.calibrate(event)
                
                    if interleaved_writer is not None:
                        interleaved_writer(event)

                    # gain select the events
                    source.r0_r1_calibrator.select_gain = True

                    r1 = event.r1.tel[tel_id]                   
                    r1.selected_gain_channel = source.r0_r1_calibrator.gain_selector(event.r0.tel[tel_id].waveform)
                    r1.waveform = r1.waveform[r1.selected_gain_channel, PIXEL_INDEX]

                    event.calibration.tel[tel_id].dl1.time_shift = \
                    event.calibration.tel[tel_id].dl1.time_shift[r1.selected_gain_channel, PIXEL_INDEX]
                    
                    event.calibration.tel[tel_id].dl1.relative_factor = \
                    event.calibration.tel[tel_id].dl1.relative_factor[r1.selected_gain_channel, PIXEL_INDEX]
                    
            # Option to add nsb in waveforms
            if nsb_tuning:
                # FIXME? assumes same correction ratio for all telescopes
                for tel_id in config['source_config']['LSTEventSource']['allowed_tels']:
                    waveform = event.r1.tel[tel_id].waveform
                    readout = subarray.tel[tel_id].camera.readout
                    sampling_rate = readout.sampling_rate.to_value(u.GHz)
                    dt = (1.0 / sampling_rate)
                    selected_gains = event.r1.tel[tel_id].selected_gain_channel
                    mask_high = (selected_gains == 0)
                    tune_nsb_on_waveform(waveform, nsb_tuning_ratio, nsb_original[tel_id] * u.GHz,
                                         dt * u.ns, pulse_template, mask_high, charge_spe_cumulative_pdf)

            # create image for all events
            r1_dl1_calibrator(event)

            # Temporal volume reducer for lstchain - dl1 level must be filled and dl0 will be overwritten.
            # When the last version of the method is implemented, vol. reduction will be done at dl0
            apply_volume_reduction(event, subarray, config)

            # FIXME? This should be eventually done after we evaluate whether the image is
            # a candidate muon ring. In that case the full image could be kept, or reduced
            # only after the ring analysis is complete.

            for telescope_id, dl1_tel in event.dl1.tel.items():
                dl1_tel.prefix = ''  # don't really need one
                tel_name = str(subarray.tel[telescope_id])[4:]

                # extra info for the image table
                extra_im.tel_id = telescope_id
                extra_im.selected_gain_channel = event.r1.tel[telescope_id].selected_gain_channel
                add_global_metadata(extra_im, metadata)
                add_config_metadata(extra_im, config)

                focal_length = subarray.tel[telescope_id].optics.equivalent_focal_length

                dl1_container.reset()

                # update the calibration index in the dl1 event container
                dl1_container.calibration_id = calibration_index.calibration_id

                dl1_container.fill_event_info(event)


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

                    calibration_mon = source.r0_r1_calibrator.mon_data.tel[telescope_id].calibration
 
                    dl1_container.ucts_time = 0
                    # convert Time to unix timestamp in (UTC) to keep compatibility
                    # with older lstchain
                    # FIXME: just keep it as time, table writer and reader handle it
                    dl1_container.dragon_time = event.trigger.time.unix
                    dl1_container.tib_time = 0
                    if 'ucts_jump' in vars(event.lst.tel[
                                               telescope_id].evt.__class__):
                        dl1_container.ucts_jump = event.lst.tel[telescope_id].evt.ucts_jump
                    dl1_container.ucts_trigger_type = event.lst.tel[telescope_id].evt.ucts_trigger_type
                    dl1_container.trigger_type = event.lst.tel[telescope_id].evt.tib_masked_trigger
                else:
                    dl1_container.trigger_type = event.trigger.event_type


                dl1_container.az_tel = event.pointing.tel[telescope_id].azimuth
                dl1_container.alt_tel = event.pointing.tel[telescope_id].altitude
                dl1_container.sin_az_tel = np.sin(dl1_container.az_tel)

                dl1_container.trigger_time = event.trigger.time.unix
                dl1_container.event_type = event.trigger.event_type

                dl1_container.prefix = dl1_tel.prefix

                for container in [extra_im, dl1_container, event.r0, dl1_tel]:
                    add_global_metadata(container, metadata)
                    add_config_metadata(container, config)

                writer.write(table_name=f'telescope/parameters/{tel_name}',
                             containers=[event.index, dl1_container])

                writer.write(
                    table_name=f'telescope/image/{tel_name}',
                    containers=[event.index, dl1_tel, extra_im]
                )

                if lhfit_fitter is not None:
                    lhfit_container = apply_lh_fit(event, telescope_id, dl1_container, lhfit_fitter)
                    # Plotting code for development purpose only, will disappear in final release
                    if lhfit_fitter.verbose >= 2 and lhfit_container["lhfit_call_status"] == 1:
                        plot_debug(lhfit_fitter, event, telescope_id, dl1_container, str(event.index.event_id))
                    lhfit_container.prefix = dl1_tel.prefix
                    add_global_metadata(lhfit_container, metadata)
                    add_config_metadata(lhfit_container, config)
                    writer.write(table_name=f'telescope/likelihood_parameters/{tel_name}',
                                 containers=[event.index, lhfit_container])

                # Muon ring analysis, for real data only (MC is done starting from DL1 files)
                if not is_simu:
                    bad_pixels = calibration_mon.unusable_pixels[0]

                    # Set to 0 unreliable pixels:
                    image = dl1_tel.image*(~bad_pixels)

                    # process only promising events, in terms of # of pixels with large signals:
                    if tag_pix_thr(image):

                        # re-calibrate r1 to obtain new dl1, using a more adequate pulse integrator for muon rings
                        numsamples = event.r1.tel[telescope_id].waveform.shape[1]  # not necessarily the same as in r0!
                        bad_pixels_hg = calibration_mon.unusable_pixels[0]
                        bad_pixels_lg = calibration_mon.unusable_pixels[1]

                        # Now set to 0 all samples in unreliable pixels. Important for global peak
                        # integrator in case of crazy pixels!  TBD: can this be done in a simpler
                        # way?
                        bad_pixels = bad_pixels_hg | bad_pixels_lg
                        bad_waveform = np.transpose(np.array(numsamples*[bad_pixels]))

                        # print('hg bad pixels:',np.where(bad_pixels_hg))
                        # print('lg bad pixels:',np.where(bad_pixels_lg))

                        event.r1.tel[telescope_id].waveform *= ~bad_waveform
                        r1_dl1_calibrator_for_muon_rings(event)
                        # since ctapipe 0.17,  the calibrator overwrites the full dl1 container
                        # instead of overwriting the image in the existing container
                        # so we need to get the image again
                        image = event.dl1.tel[telescope_id].image * (~bad_pixels)

                        # Check again: with the extractor for muon rings (most likely GlobalPeakWindowSum)
                        # perhaps the event is no longer promising (e.g. if it has a large time evolution)
                        if not tag_pix_thr(image):
                            good_ring = False
                        else:
                            muonintensityparam, dist_mask, \
                            ring_size, size_outside_ring, muonringparam, \
                            good_ring, radial_distribution, \
                            mean_pixel_charge_around_ring,\
                            muonpars = \
                                analyze_muon_event(subarray,
                                                   tel_id, event.index.event_id,
                                                   image, good_ring_config=None,
                                                   plot_rings=False, plots_path='')
                            #                      plot_rings=True, plots_path='./')
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

        if event is None:
            logger.warning('No events in file')

        if not is_simu and event is not None:
            # at the end of event loop ask calculation of remaining interleaved statistics
            new_ped, new_ff = calibration_calculator.output_interleaved_results(event)
            # write monitoring events
            # these data a supposed to be replaced by the Cat_B data in a short future
            write_calibration_data(writer,
                                   calibration_index,
                                   event.mon.tel[tel_id],
                                   new_ped=new_ped, new_ff=new_ff)

    if is_simu and event is not None:
        # Reconstruct source position from disp for all events and write the result in the output file
        add_disp_to_parameters_table(output_filename, dl1_params_lstcam_key, focal_length)

        # Write energy histogram from simtel file and extra metadata
        # ONLY of the simtel file has been read until the end, otherwise it seems to hang here forever
        if source.max_events is None:
            write_simtel_energy_histogram(source, output_filename, obs_id=event.index.obs_id,
                                          metadata=metadata)
    if not is_simu:
        dir, name = os.path.split(output_filename)
        name = name.replace('dl1', 'muons').replace('LST-1.1', 'LST-1')
        # Consider the possibilities of DL1 files with .fits.h5 & .h5 ending:
        name = name.replace('.fits.h5', '.fits').replace('.h5', '.fits')
        muon_output_filename = Path(dir, name)
        table = Table(muon_parameters)
        table.write(muon_output_filename, format='fits', overwrite=True)
        
        # close the interleaved output file and write metadata
        if interleaved_writer is not None:
            interleaved_writer.finish()



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
                                source_pos_in_camera.y,
                                df.psi.values * u.rad)

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
