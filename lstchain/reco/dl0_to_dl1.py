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

from ..calib.camera import lst_calibration, load_calibrator_from_config
from ..io import DL1ParametersContainer, standard_config, replace_config

import pandas as pd
from . import disp
import astropy.units as u
from .utils import sky_to_camera
from ctapipe.instrument import OpticsDescription

__all__ = [
    'get_dl1',
    'r0_to_dl1',
]



cleaning_method = tailcuts_clean


def get_dl1(calibrated_event, telescope_id, dl1_container=None, custom_config={}):
    """
    Return a DL1ParametersContainer of extracted features from a calibrated event.
    The DL1ParametersContainer can be passed to be filled if created outside the function
    (faster for multiple event processing)

    Parameters
    ----------
    event: ctapipe event container
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

    image = dl1.image
    pulse_time = dl1.pulse_time

    signal_pixels = cleaning_method(camera, image, **cleaning_parameters)

    if image[signal_pixels].sum() > 0:
        hillas = hillas_parameters(camera[signal_pixels], image[signal_pixels])
        # Fill container
        dl1_container.fill_mc(calibrated_event)
        dl1_container.fill_hillas(hillas)
        dl1_container.fill_event_info(calibrated_event)
        dl1_container.set_mc_core_distance(calibrated_event, telescope_id)
        dl1_container.set_mc_type(calibrated_event)
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


def r0_to_dl1(input_filename=get_dataset_path('gamma_test_large.simtel.gz'), output_filename=None, custom_config={}):
    """
    Chain r0 to dl1
    Save the extracted dl1 parameters in output_filename

    Parameters
    ----------
    input_filename: str
        path to input file, default: `gamma_test_large.simtel.gz`
    output_filename: str
        path to output file, default: `./` + basename(input_filename)
    config_file: path to a configuration file

    Returns
    -------

    """
    if output_filename is None:
        output_filename = (
            'dl1_' + os.path.basename(input_filename).split('.')[0] + '.h5'
        )


    config = replace_config(standard_config, custom_config)

    custom_calibration = config["custom_calibration"]

    source = event_source(input_filename)
    source.allowed_tels = config["allowed_tels"]
    source.max_events = config["max_events"]

    cal = load_calibrator_from_config(config)

    dl1_container = DL1ParametersContainer()

    with HDF5TableWriter(
        filename=output_filename,
        group_name='events',
        overwrite=True
    ) as writer:

        for i, event in enumerate(source):
            if i % 100 == 0:
                print(i)
            if not custom_calibration:
                cal(event)
                # for telescope_id, dl1 in event.dl1.tel.items():
            for ii, telescope_id in enumerate(event.r0.tels_with_data):
                if custom_calibration:
                    lst_calibration(event, telescope_id)

                try:
                    dl1_filled = get_dl1(event, telescope_id, dl1_container=dl1_container, custom_config=config)
                except HillasParameterizationError:
                    logging.exception(
                        'HillasParameterizationError in get_dl1()'
                    )
                    continue

                if dl1_filled is not None:

                    # Some custom def
                    dl1_container.wl = dl1_container.width / dl1_container.length
                    # Log10(Energy) in GeV
                    dl1_container.mc_energy = np.log10(event.mc.energy.value * 1e3)
                    dl1_container.intensity = np.log10(dl1_container.intensity)
                    dl1_container.gps_time = event.trig.gps_time.value

                    foclen = (
                        event.inst.subarray.tel[telescope_id]
                        .optics.equivalent_focal_length
                    )
                    width = np.rad2deg(np.arctan2(dl1_container.width, foclen))
                    length = np.rad2deg(np.arctan2(dl1_container.length, foclen))
                    dl1_container.width = width.value
                    dl1_container.length = length.value

                    if width >= 0:
                        # Camera geometry
                        camera = event.inst.subarray.tel[telescope_id].camera
                        writer.write(camera.cam_id, [dl1_container])

    lst_focal = OpticsDescription.from_name('LST').equivalent_focal_length

    with pd.HDFStore(output_filename) as store:

        df = store['events/LSTCam']

        source_pos_in_camera = sky_to_camera(df.mc_alt.values * u.rad,
                                             df.mc_az.values * u.rad,
                                             lst_focal,
                                             df.mc_alt_tel.values * u.rad,
                                             df.mc_az_tel.values * u.rad,
                                             )
        disp_parameters = disp.disp(df.x.values * u.m,
                                    df.y.values * u.m,
                                    source_pos_in_camera.x,
                                    source_pos_in_camera.y)

        disp_df = pd.DataFrame(np.transpose(disp_parameters),
                               columns=['disp_dx', 'disp_dy', 'disp_norm', 'disp_angle', 'disp_sign'])
        disp_df['src_x'] = source_pos_in_camera.x.value
        disp_df['src_y'] = source_pos_in_camera.y.value

        store['events/LSTCam'] = pd.concat([store['events/LSTCam'], disp_df], axis=1)

    with HDF5TableWriter(filename=output_filename, group_name="simulation", mode="a") as writer:
        writer.write("run_config", [event.mcheader])



def get_spectral_w_pars(filename):
    """
    Return parameters required to calculate spectral weighting of an event

    Parameters
    ----------
    filename: string, simtelarray file

    Returns
    -------
    array of parameters
    """

    particle = utils.guess_type(filename)

    source = SimTelFile(filename)

    emin,emax = source.mc_run_headers[0]['E_range']*1e3 #GeV
    spectral_index = source.mc_run_headers[0]['spectral_index']
    num_showers = source.mc_run_headers[0]['num_showers']
    num_use = source.mc_run_headers[0]['num_use']
    Simulated_Events = num_showers * num_use
    Max_impact = source.mc_run_headers[0]['core_range'][1]*1e2 #cm
    Area_sim = np.pi * math.pow(Max_impact,2)
    cone = source.mc_run_headers[0]['viewcone'][1]

    cone = cone * np.pi/180
    if(cone == 0):
        Omega = 1
    else:
        Omega = 2*np.pi*(1-np.cos(cone))

    if particle=='proton':
        K_w = 9.6e-11 # GeV^-1 cm^-2 s^-1
        index_w = -2.7
        E0 = 1000. # GeV
    if particle=='gamma':
        K_w = 2.83e-11 # GeV^-1 cm^-2 s^-1
        index_w = -2.62
        E0 = 1000. # GeV

    K = Simulated_Events*(1+spectral_index)/(emax**(1+spectral_index)-emin**(1+spectral_index))
    Int_e1_e2 = K*E0**spectral_index
    N_ = Int_e1_e2*(emax**(index_w+1)-emin**(index_w+1))/(E0**index_w)/(index_w+1)
    R = K_w*Area_sim*Omega*(emax**(index_w+1)-emin**(index_w+1))/(E0**index_w)/(index_w+1)

    return E0,spectral_index,index_w,R,N_

def get_spectral_w(w_pars, energy):
    """
    Return spectral weight of an event

    Parameters
    ----------
    w_pars: parameters obtained with get_spectral_w_pars() function
    energy: energy of the event in GeV

    Returns
    -------
    float w
    """

    E0 = w_pars[0]
    index = w_pars[1]
    index_w = w_pars[2]
    R = w_pars[3]
    N_ = w_pars[4]

    w = ((energy/E0)**(index_w-index))*R/N_

    return w
