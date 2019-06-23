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
from ctapipe.image.extractor import LocalPeakWindowSum
from ctapipe.image import timing_parameters as time
from ctapipe.instrument import OpticsDescription
from ctapipe.utils import get_dataset_path
from ctapipe.calib import CameraCalibrator
from ctapipe.io import event_source
from ctapipe.io import HDF5TableWriter
from eventio.simtel.simtelfile import SimTelFile
import pandas as pd
import astropy.units as units
import h5py
import math
from . import utils
from ..calib.camera import lst_calibration, gain_selection
from ..io.lstcontainers import DL1ParametersContainer
from ctapipe.image.extractor import NeighborPeakWindowSum

__all__ = [
    'get_dl1',
    'r0_to_dl1',
    'get_events',
]


### PARAMETERS - TODO: use a yaml config file


allowed_tels = {1, 2, 3, 4}  # select LST1 only
max_events = None  # limit the number of events to analyse in files - None if no limit

threshold = 4094

# Add option to use custom calibration

custom = False

cal = CameraCalibrator(image_extractor=NeighborPeakWindowSum())

cleaning_method = tailcuts_clean
cleaning_parameters = {'boundary_thresh': 3,
                       'picture_thresh': 6,
                       'keep_isolated_pixels': False,
                       'min_number_picture_neighbors': 1
                       }


def get_dl1(calibrated_event, telescope_id, dl1_container=None):
    """
    Return a DL1ParametersContainer of extracted features from a calibrated event.
    The DL1ParametersContainer can be passed to be filled if created outside the function
    (faster for multiple event processing)

    Parameters
    ----------
    event: ctapipe event container
    telescope_id: int
    dl1_container: DL1ParametersContainer

    Returns
    -------
    DL1ParametersContainer
    """
    dl1_container = DL1ParametersContainer() if dl1_container is None else dl1_container

    tel = calibrated_event.inst.subarray.tels[telescope_id]
    dl1 = calibrated_event.dl1.tel[telescope_id]
    camera = tel.camera

    waveform = calibrated_event.r0.tel[telescope_id].waveform
    image = dl1.image
    pulse_time = dl1.pulse_time

    image, pulse_time = gain_selection(waveform, image, pulse_time, threshold)

    signal_pixels = cleaning_method(camera, image,
                                    **cleaning_parameters)
    image[~signal_pixels] = 0

    if image.sum() > 0:
        hillas = hillas_parameters(camera, image)
        # Fill container
        dl1_container.fill_mc(calibrated_event)
        dl1_container.fill_hillas(hillas)
        dl1_container.fill_event_info(calibrated_event)
        dl1_container.set_mc_core_distance(calibrated_event, telescope_id)
        dl1_container.set_mc_type(calibrated_event)
        dl1_container.set_timing_features(camera, image, pulse_time, hillas)
        dl1_container.set_leakage(camera, image, signal_pixels)
        dl1_container.set_n_islands(camera, signal_pixels)
        dl1_container.set_source_camera_position(
            calibrated_event, telescope_id)
        dl1_container.set_disp(
            [dl1_container.src_x, dl1_container.src_y],
            hillas
        )
        dl1_container.set_telescope_info(calibrated_event, telescope_id)

        return dl1_container

    else:
        return None


def r0_to_dl1(
    input_filename=get_dataset_path('gamma_test_large.simtel.gz'),
    output_filename=None
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

    Returns
    -------

    """
    if output_filename is None:
        output_filename = (
            'dl1_' + os.path.basename(input_filename).split('.')[0] + '.h5'
        )

    source = event_source(input_filename)
    source.allowed_tels = allowed_tels
    source.max_events = max_events

    dl1_container = DL1ParametersContainer()

    with HDF5TableWriter(
        filename=output_filename,
        group_name='events',
        overwrite=True
    ) as writer:

        for i, event in enumerate(source):
            if i % 100 == 0:
                print(i)
            if not custom:
                cal(event)
                # for telescope_id, dl1 in event.dl1.tel.items():
            for ii, telescope_id in enumerate(event.r0.tels_with_data):
                if custom:
                    lst_calibration(event, telescope_id)

                try:
                    dl1_filled = get_dl1(event, telescope_id, dl1_container=dl1_container)
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

    with HDF5TableWriter(filename=output_filename, group_name="simulation", mode="a") as writer:
        writer.write("run_config", [event.mcheader])


def get_events(filename, storedata=False, test=False,
               concatenate=False, storeimg=False, outdir='./results/'):
    """
    Depreciated, use r0_to_dl1.

    Read a Simtelarray file, extract pixels charge, calculate image
    parameters and timing parameters and store the result in an hdf5
    file.

    Parameters:
    -----------
    filename: str
    Name of the simtelarray file.

    storedata: boolean
    True: store extracted data in a hdf5 file

    concatenate: boolean
    True: store the extracted data at the end of an existing file

    storeimg: boolean
    True: store also pixel data

    outdir: srt
    Output directory

    Returns:
    --------
    pandas DataFrame: output
    """
    from warnings import warn
    warn("Deprecated: use r0_to_dl1")

    #Particle type:
    particle_type = utils.guess_type(filename)

    #Create data frame where DL2 data will be stored:

    features = ['obs_id',
                'event_id',
                'mc_energy',
                'mc_alt',
                'mc_az',
                'mc_core_x',
                'mc_core_y',
                'mc_h_first_int',
                'mc_type',
                'gps_time',
                'width',
                'length',
                'wl',
                'phi',
                'psi',
                'r',
                'x',
                'y',
                'intensity',
                'skewness',
                'kurtosis',
                'mc_alt_tel',
                'mc_az_tel',
                'mc_core_distance',
                'mc_x_max',
                'time_gradient',
                'intercept',
                'src_x',
                'src_y',
                'disp_norm',
                ]

    output = pd.DataFrame(columns=features)

    #Read LST1 events:
    source = event_source(
        input_url=filename,
        allowed_tels={1}) #Open Simtelarray file

    #Cleaning levels:

    level1 = {'LSTCam' : 6.}
    level2 = level1.copy()
    # We use as second cleaning level just half of the first cleaning level

    for key in level2:
        level2[key] *= 0.5


    log10pixelHGsignal = {}
    survived = {}

    imagedata = np.array([])

    for key in level1:

        log10pixelHGsignal[key] = []
        survived[key] = []
    i=0
    for event in source:
        if i%100==0:
            print("EVENT_ID: ", event.r0.event_id, "TELS: ",
                  event.r0.tels_with_data,
                  "MC Energy:", event.mc.energy )

        i=i+1

        ntels = len(event.r0.tels_with_data)


        if test==True and i > 1000:   # for quick tests
            break

        for ii, tel_id in enumerate(event.r0.tels_with_data):

            geom = event.inst.subarray.tel[tel_id].camera     #Camera geometry
            tel_coords = event.inst.subarray.tel_coords[
                event.inst.subarray.tel_indices[tel_id]
            ]

            data = event.r0.tel[tel_id].waveform

            ped = event.mc.tel[tel_id].pedestal    # the pedestal is the
            #average (for pedestal events) of the *sum* of all samples,
            #from sim_telarray

            nsamples = data.shape[2]  # total number of samples

            # Subtract pedestal baseline. atleast_3d converts 2D to 3D matrix

            pedcorrectedsamples = data - np.atleast_3d(ped)/nsamples

            integrator = LocalPeakWindowSum()
            integration, pulse_time = integrator(pedcorrectedsamples) # these are 2D matrices num_gains * num_pixels

            chan = 0  # high gain used for now...
            signals = integration[chan].astype(float)

            dc2pe = event.mc.tel[tel_id].dc_to_pe   # numgains * numpixels
            signals *= dc2pe[chan]

            # Add all individual pixel signals to the numpy array of the
            # corresponding camera inside the log10pixelsignal dictionary

            log10pixelHGsignal[str(geom)].extend(np.log10(signals))

            # Apply image cleaning

            cleanmask = tailcuts_clean(geom, signals,
                                       picture_thresh=level1[str(geom)],
                                       boundary_thresh=level2[str(geom)],
                                       keep_isolated_pixels=False,
                                       min_number_picture_neighbors=1)

            survived[str(geom)].extend(cleanmask)

            clean = signals.copy()
            clean[~cleanmask] = 0.0   # set to 0 pixels which did not
            # survive cleaning

            if np.max(clean) < 1.e-6:  # skip images with no pixels
                continue

            # Calculate image parameters

            hillas = hillas_parameters(geom, clean)
            foclen = event.inst.subarray.tel[tel_id].optics.equivalent_focal_length

            w = np.rad2deg(np.arctan2(hillas.width, foclen))
            l = np.rad2deg(np.arctan2(hillas.length, foclen))

            #Calculate Timing parameters

            peak_time = units.Quantity(pulse_time[chan])*units.Unit("ns")
            timepars = time.timing_parameters(geom,clean,peak_time,hillas)

            if w >= 0:
                if storeimg==True:
                    if imagedata.size == 0:
                        imagedata = clean
                    else:
                        imagedata = np.vstack([imagedata,clean]) #Pixel content

                #Hillas parameters
                width = w.value
                length = l.value
                phi = hillas.phi.value
                psi = hillas.psi.value
                r = hillas.r.value
                x = hillas.x.value
                y = hillas.y.value
                intensity = np.log10(hillas.intensity)
                skewness = hillas.skewness
                kurtosis = hillas.kurtosis

                #MC data:
                obs_id = event.r0.obs_id
                event_id = event.r0.event_id

                mc_energy = np.log10(event.mc.energy.value*1e3) #Log10(Energy) in GeV
                mc_alt = event.mc.alt.value
                mc_az = event.mc.az.value
                mc_core_x = event.mc.core_x.value
                mc_core_y = event.mc.core_y.value
                mc_h_first_int = event.mc.h_first_int.value
                mc_type = event.mc.shower_primary_id
                mc_az_tel = event.mcheader.run_array_direction[0].value
                mc_alt_tel = event.mcheader.run_array_direction[1].value
                mc_x_max = event.mc.x_max.value
                gps_time = event.trig.gps_time.value

                #Calculate mc_core_distance parameters

                mc_core_distance = np.sqrt((
                    tel_coords.x.value-event.mc.core_x.value)**2
                    +(tel_coords.y.value-event.mc.core_y.value)**2)

                #Timing parameters

                time_gradient = timepars['slope'].value
                intercept = timepars['intercept']

                #Calculate disp_ and Source position in camera coordinates

                tel = OpticsDescription.from_name('LST') #Telescope description
                focal_length = tel.equivalent_focal_length.value
                sourcepos = utils.cal_cam_source_pos(mc_alt,mc_az,
                                                              mc_alt_tel,mc_az_tel,
                                                              focal_length)
                src_x = sourcepos[0]
                src_y = sourcepos[1]
                disp = utils.disp_norm(sourcepos[0], sourcepos[1], x, y)

                eventdf = pd.DataFrame([[obs_id, event_id, mc_energy, mc_alt, mc_az,
                                         mc_core_x, mc_core_y, mc_h_first_int, mc_type,
                                         gps_time, width, length, width / length, phi,
                                         psi, r, x, y, intensity, skewness, kurtosis,
                                         mc_alt_tel, mc_az_tel, mc_core_distance, mc_x_max,
                                         time_gradient, intercept, src_x, src_y,
                                         disp, mc_type]],
                                       columns=features)

                output = output.append(eventdf,
                                       ignore_index=True)

    outfile = outdir + particle_type + '_events.hdf5'

    if storedata==True:
        if (concatenate==False or
            (concatenate==True and
             np.DataSource().exists(outfile)==False)):
            output.to_hdf(outfile,
                          key=particle_type+"_events",mode="w")
            if storeimg==True:
                f = h5py.File(outfile,'r+')
                f.create_dataset('images',data=imagedata)
                f.close()
        else:
            if storeimg==True:
                f = h5py.File(outfile,'r')
                images = f['images']
                del f['images']
                images = np.vstack([images,imagedata])
                f.close()
                saved = pd.read_hdf(outfile, key=particle_type+'_events')
                output = saved.append(output, ignore_index=True)
                output.to_hdf(outfile,key=particle_type+"_events",mode="w")
                f = h5py.File(outfile,'r+')
                f.create_dataset('images',data=images)
                f.close()
            else:
                saved = pd.read_hdf(outfile, key=particle_type+'_events')
                output = saved.append(output, ignore_index=True)
                output.to_hdf(outfile, key=particle_type+"_events",mode="w")
    del source
    return output

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

def get_spectral_w(w_pars,energy):
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
