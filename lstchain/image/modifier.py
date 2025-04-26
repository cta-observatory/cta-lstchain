import astropy.units as u
import logging
import numpy as np
from ctapipe.calib.camera import CameraCalibrator
from ctapipe.io import (
    EventSource,
    read_table,
)
from ctapipe.containers import EventType
from numba import njit
from scipy.interpolate import interp1d
from traitlets.config import Config

from lstchain.io import standard_config, read_configuration_file
from lstchain.reco.reconstructorCC import nsb_only_waveforms
from lstchain.data import NormalizedPulseTemplate
from lstchain.io.io import get_resource_path
from scipy.optimize import curve_fit

__all__ = [
    'add_noise_in_pixels',
    'calculate_required_additional_nsb',
    'calculate_noise_parameters',
    'random_psf_smearer',
    'set_numba_seed',
    'WaveformNsbTuner',
]

log = logging.getLogger(__name__)

# number of neighbors of completely surrounded pixels of hexagonal cameras:
N_PIXEL_NEIGHBORS = 6
SMEAR_PROBABILITIES = np.full(N_PIXEL_NEIGHBORS, 1 / N_PIXEL_NEIGHBORS)


def add_noise_in_pixels(rng, image, extra_noise_in_dim_pixels,
                        extra_bias_in_dim_pixels, transition_charge,
                        extra_noise_in_bright_pixels):
    """
    Addition of Poissonian noise to the pixels

    Parameters
    ----------
    rng : `numpy.random.default_rng`
        Random number generator
    image: `np.ndarray`
        Charges (p.e.) in the camera
    extra_noise_in_dim_pixels: `float`
        Mean additional number of p.e. to be added (Poisson noise) to
        pixels with charge below transition_charge. To be tuned by
        comparing the starting MC and data
    extra_bias_in_dim_pixels: `float`
        Mean bias (w.r.t. original charge) of the new charge in pixels.
        Should be 0 for non-peak-search pulse integrators. To be tuned by
        comparing the starting MC and data
    transition_charge: `float`
        Border between "dim" and "bright" pixels. To be tuned by
        comparing the starting MC and data
    extra_noise_in_bright_pixels: `float`
        Mean additional number of p.e. to be added (Poisson noise) to
        pixels with charge above transition_charge. This is unbiased,
        i.e. Poisson noise is introduced, and its average subtracted,
        so that the mean charge in bright pixels remains unaltered.
        This is because we assume that above transition_charge the
        integration window is determined by the Cherenkov light, and
        would not be modified by the additional NSB noise (presumably
        small compared to the C-light). To be tuned by
        comparing the starting MC and data

    Returns
    -------
    image: `np.ndarray`
        Modified (noisier) image

    """

    bright_pixels = image > transition_charge
    noise = np.where(bright_pixels, extra_noise_in_bright_pixels,
                     extra_noise_in_dim_pixels)
    bias = np.where(bright_pixels, -extra_noise_in_bright_pixels,
                    extra_bias_in_dim_pixels - extra_noise_in_dim_pixels)

    image = image + rng.poisson(noise) + bias

    return image


@njit(cache=True)
def set_numba_seed(seed):
    np.random.seed(seed)


@njit(cache=True)
def random_psf_smearer(image, fraction, indices, indptr):
    """
    Random PSF smearer

    Parameters
    ----------
    image: `np.ndarray`
        Charges (p.e.) in the camera
    indices : `camera_geometry.neighbor_matrix_sparse.indices`
        Pixel indices.
    indptr : camera_geometry.neighbor_matrix_sparse.indptr
    fraction: `float`
        Fraction of the light in a pixel that will be distributed among its
        immediate surroundings, i.e. immediate neighboring pixels, according
        to Poisson statistics. Some light is lost for pixels  which are at
        the camera edge and hence don't have all possible neighbors

    Returns
    -------
    new_image: `np.ndarray`
        Modified (smeared) image

    """

    new_image = image.copy()
    for pixel in range(len(image)):

        if image[pixel] <= 0:
            continue

        to_smear = np.random.poisson(image[pixel] * fraction)

        if to_smear == 0:
            continue

        # remove light from current pixel
        new_image[pixel] -= to_smear

        # add light to neighbor pixels
        neighbors = indices[indptr[pixel]: indptr[pixel + 1]]
        n_neighbors = len(neighbors)

        # all neighbors are equally likely to receive the charge
        # we always distribute the charge into 6 neighbors, so that charge
        # on the edges of the camera is lost
        neighbor_charges = np.random.multinomial(to_smear, SMEAR_PROBABILITIES)

        for n in range(n_neighbors):
            neighbor = neighbors[n]
            new_image[neighbor] += neighbor_charges[n]

    return new_image


def calculate_noise_parameters(simtel_filename, data_dl1_filename,
                               config_filename=None):
    """
    Calculates the parameters needed to increase the noise in an MC DL1 file
    to match the noise in a real data DL1 file, using add_noise_in_pixels
    The returned parameters are those needed by the function add_noise_in_pixels (see
    description in its documentation above).

    Parameters
    ----------
    simtel_filename: `str`
        a simtel file containing showers, from the same
        production (same NSB and telescope settings) as the MC DL1 file below. It
        must contain pixel-wise info on true number of p.e.'s from C-photons (
        will be used to identify pixels which only contain noise).

    data_dl1_filename: `str`
        a real data DL1 file (processed with calibration
        settings corresponding to those with which the MC is to be processed).
        It must contain calibrated images, i.e. "DL1a" data. This file has the
        "target" noise which we want to have in the MC files, for better
        agreement of data and simulations.

    config_filename: `str`
        configuration file containing the calibration
        settings used for processing both the data and the MC files above

    Returns
    -------
    extra_noise_in_dim_pixels: `float`
        Extra noise of dim pixels (number of NSB photoelectrons).
    extra_bias_in_dim_pixels: `float`
        Extra bias of dim pixels  (direct shift in photoelectrons).
    extra_noise_in_bright_pixels: `float`
        Extra noise of bright pixels  (number of NSB photoelectrons).

    """

    log.setLevel(logging.INFO)

    if config_filename is None:
        config = standard_config
    else:
        config = read_configuration_file(config_filename)

    # Real data DL1 tables:
    data_dl1_calibration = read_table(data_dl1_filename,
                                      '/dl1/event/telescope/monitoring/calibration')
    data_dl1_pedestal = read_table(data_dl1_filename,
                                   '/dl1/event/telescope/monitoring/pedestal')
    data_dl1_flatfield = read_table(data_dl1_filename,
                                    '/dl1/event/telescope/monitoring/flatfield')
    data_dl1_parameters = read_table(data_dl1_filename,
                                     '/dl1/event/telescope/parameters/LST_LSTCam')
    data_dl1_image = read_table(data_dl1_filename,
                                '/dl1/event/telescope/image/LST_LSTCam')

    unusable = data_dl1_calibration['unusable_pixels']
    # Locate pixels with HG declared unusable either in original calibration or
    # in interleaved events:
    bad_pixels = unusable[0][0]  # original calibration
    if unusable.shape[0] > 1:
        for tf in unusable[1:][0]:  # calibrations with interleaveds
            bad_pixels = np.logical_or(bad_pixels, tf)
    good_pixels = ~bad_pixels

    # First index:  1,2,... = values from interleaveds (0 is for original
    # calibration run)
    # Second index: 0 = high gain
    # Third index: pixels

    # HG adc to pe conversion factors from interleaved calibrations:
    data_HG_dc_to_pe = data_dl1_calibration['dc_to_pe'][:, 0, :]

    if data_dl1_flatfield['charge_mean'].shape[0] < 2:
        logging.error('\nCould not find interleaved FF calibrations in '
                      'monitoring table!')
        return None, None, None

    if data_dl1_pedestal['charge_std'].shape[0] < 2:
        logging.error('\nCould not find interleaved pedestal info in '
                      'monitoring table!')
        return None, None, None

    # Mean HG charge in interleaved FF events, to spot possible issues:
    data_HG_FF_mean = data_dl1_flatfield['charge_mean'][1:, 0, :]
    dummy = []
    # indices which connect each FF calculation to a given calibration:
    calibration_id = data_dl1_flatfield['calibration_id'][1:]

    for i, x in enumerate(data_HG_FF_mean[:, ]):
        dummy.append(x * data_HG_dc_to_pe[calibration_id[i],])
    dummy = np.array(dummy)
    # Average for all interleaved calibrations (in case there are more than one)
    data_HG_FF_mean_pe = np.mean(dummy, axis=0)  # one value per pixel

    # Pixel-wise pedestal standard deviation (for an unbiased extractor),
    # in adc counts:
    data_HG_ped_std = data_dl1_pedestal['charge_std'][1:, 0, :]
    # indices which connect each pedestal calculation to a given calibration:
    calibration_id = data_dl1_pedestal['calibration_id'][1:]
    # convert pedestal st deviations to p.e.
    dummy = []
    for i, x in enumerate(data_HG_ped_std[:, ]):
        dummy.append(x * data_HG_dc_to_pe[calibration_id[i],])
    dummy = np.array(dummy)
    # Average for all interleaved calibrations (in case there are more than one)
    data_HG_ped_std_pe = np.mean(dummy, axis=0)  # one value per pixel

    # Identify noisy pixels, likely containing stars - we want to adjust MC to
    # the average diffuse NSB across the camera

    data_median_std_ped_pe = np.nanmedian(data_HG_ped_std_pe[good_pixels])
    data_std_std_ped_pe = np.nanstd(data_HG_ped_std_pe[good_pixels])
    log.info('\nReal data:')
    log.info(f'   Number of bad pixels (from calibration): {bad_pixels.sum()}')
    log.info(f'   Median of FF pixel charge: '
             f'{np.nanmedian(data_HG_FF_mean_pe[good_pixels]):.3f} p.e.')
    log.info(f'   Median across camera of good pixels\' pedestal std '

             f'{data_median_std_ped_pe:.3f} p.e.')
    brightness_limit = data_median_std_ped_pe + 3 * data_std_std_ped_pe
    too_bright_pixels = (data_HG_ped_std_pe > brightness_limit)
    log.info(f'   Number of pixels beyond 3 std dev of median: '
             f'   {too_bright_pixels.sum()}, (above {brightness_limit:.2f} '
             f'p.e.)')

    ped_mask = data_dl1_parameters['event_type'] == 2
    # The charges in the images below are obtained with the extractor for
    # showers, usually a biased one, like e.g. LocalPeakWindowSum
    data_ped_charges = data_dl1_image['image'][ped_mask]

    # Exclude too bright pixels, besides those with unusable calibration:
    good_pixels &= ~too_bright_pixels
    # recalculate the median of the pixels' std dev, with good_pixels:
    data_median_std_ped_pe = np.nanmedian(data_HG_ped_std_pe[good_pixels])

    log.info(f'Good and not too bright pixels: {good_pixels.sum()}')

    # all_good is an events*pixels boolean array of valid signals:
    all_good = np.reshape(np.tile(good_pixels, data_ped_charges.shape[0]),
                          data_ped_charges.shape)

    # histogram of pedestal charges (biased extractor) from good and not noisy
    # pixels:
    qbins = 100
    qrange = (-10, 15)
    dataq = np.histogram(data_ped_charges[all_good].flatten(), bins=qbins,
                         range=qrange, density=True)

    # Find the peak of the pedestal biased charge distribution of real data.
    # Use an interpolated version of the histogram, for robustness:
    func = interp1d(0.5 * (dataq[1][1:] + dataq[1][:-1]), dataq[0],
                    kind='quadratic', fill_value='extrapolate')
    xx = np.linspace(qrange[0], qrange[1], 100 * qbins)
    mode_data = xx[np.argmax(func(xx))]

    # Event reader for simtel file:
    with EventSource(input_url=simtel_filename, config=Config(config)) as mc_reader:
        # Obtain the configuration with which the pedestal calculations were
        # performed:
        ped_config = config['LSTCalibrationCalculator']['PedestalIntegrator']
        tel_id = ped_config['tel_id']
        # Obtain the (unbiased) extractor used for pedestal calculations:
        pedestal_extractor_type = ped_config['charge_product']
        pedestal_calibrator = CameraCalibrator(
            image_extractor_type=pedestal_extractor_type,
            config=Config(ped_config),
            subarray=mc_reader.subarray
        )

        # Obtain the (usually biased) extractor used for shower images:
        shower_extractor_type = config['image_extractor']
        shower_calibrator = CameraCalibrator(
            image_extractor_type=shower_extractor_type,
            config=Config(config),
            subarray=mc_reader.subarray
        )
    
        # Since these extractors are now for use on MC, we have to apply the pulse
        # integration correction (in data that is currently, as of
        # lstchain v0.7.5, replaced by an empirical (hard-coded) correction of the
        # adc to pe conversion factors )
        pedestal_calibrator.image_extractors[ped_config['charge_product']].apply_integration_correction = True
        shower_calibrator.image_extractors[shower_extractor_type].apply_integration_correction = True

        # Pulse integration window width of the (biased) extractor for showers:
        shower_extractor_window_width = config[config['image_extractor']]['window_width']
    
        # Pulse integration window width for the pedestal estimation:
        pedestal_extractor_config = ped_config[pedestal_extractor_type]
        pedestal_extractor_window_width = pedestal_extractor_config['window_width']
    
        # MC pedestals integrated with the unbiased pedestal extractor
        mc_ped_charges = []
        # MC pedestals integrated with the biased shower extractor
        mc_ped_charges_biased = []

        for event in mc_reader:
            if tel_id not in event.trigger.tels_with_trigger:
                continue
            # Extract the signals as we do for pedestals (unbiased fixed window
            # extractor):
            pedestal_calibrator(event)
            charges = event.dl1.tel[tel_id].image
    
            # True number of pe's from Cherenkov photons (to identify noise-only pixels)
            true_image = event.simulation.tel[tel_id].true_image
            mc_ped_charges.append(charges[true_image == 0])
    
            # Now extract the signal as we would do for shower events (usually
            # with a biased extractor, e.g. LocalPeakWindowSum):
            shower_calibrator(event)
            charges_biased = event.dl1.tel[tel_id].image
            mc_ped_charges_biased.append(charges_biased[true_image == 0])

    # All pixels behave (for now) in the same way in MC, just put them together
    mc_ped_charges = np.concatenate(mc_ped_charges)
    mc_ped_charges_biased = np.concatenate(mc_ped_charges_biased)

    mcq = np.histogram(mc_ped_charges_biased, bins=qbins, range=qrange,
                       density=True)
    # Find the peak of the pedestal biased charge distribution of MC. Use
    # an interpolated version of the histogram, for robustness:
    func = interp1d(0.5 * (mcq[1][1:] + mcq[1][:-1]), mcq[0],
                    kind='quadratic', fill_value='extrapolate')
    xx = np.linspace(qrange[0], qrange[1], 100 * qbins)
    mode_mc = xx[np.argmax(func(xx))]

    mc_unbiased_std_ped_pe = np.std(mc_ped_charges)

    # Find the additional noise (in data w.r.t. MC) for the unbiased extractor,
    # and scale it to the width of the window for integration of shower images.
    # The idea is that when a strong signal is present, the biased extractor
    # will integrate around it, and the additional noise is unbiased because
    # it won't modify the integration range.
    # The noise is defined as the number of NSB photoelectrons, i.e. the extra
    # variance, rather than standard deviation, of the distribution
    extra_noise_in_bright_pixels = \
        ((data_median_std_ped_pe ** 2 - mc_unbiased_std_ped_pe ** 2) *
         shower_extractor_window_width / pedestal_extractor_window_width)

    # Just in case, makes sure we just add noise if the MC noise is smaller
    # than the real data's:
    extra_noise_in_bright_pixels = max(0., extra_noise_in_bright_pixels)

    bias = mode_data - mode_mc
    extra_bias_in_dim_pixels = max(bias, 0)

    # differences of values to peak charge:
    dq = data_ped_charges[all_good].flatten() - mode_data
    dqmc = mc_ped_charges_biased - mode_mc
    # maximum distance (in pe) from peak, to avoid strong impact of outliers:
    maxq = 10
    # calculate widening of the noise bump:
    added_noise = (np.sum(dq[dq < maxq] ** 2) / len(dq[dq < maxq]) -
                   np.sum(dqmc[dqmc < maxq] ** 2) / len(dqmc[dqmc < maxq]))
    extra_noise_in_dim_pixels = max(0., added_noise)

    return extra_noise_in_dim_pixels, extra_bias_in_dim_pixels, \
        extra_noise_in_bright_pixels


def get_pix_median_charges(data_dl1_filename, event_type):
    """
    Obtain from a DL1 real data file (containing camera images, i.e. DL1a)
    the median (across camera) of mean (and std dev) of pixel charge for
    events of type event_type. The medians are obtained using only healthy
    pixels and excluding too bright outliers (we want to get the typical
    pixel values, unaffected by the few stars in the FoV)

    Parameters
    ----------
    data_dl1_filename: DL1 file name
    event_type (ctapipe.containers.EventType): type of events, e.g. pedestals

    Returns
    -------
    median_ped_meanpixq (p.e.)
    median_ped_stdpixq  (p.e.)  Median values for healthy and not too
    noisy pixels
    tel_id : telescope id to which the events in the file correspond (we assume
    for now that a given DL1 file contains data from just one telescope)
    """

    # Real data DL1 tables:
    camera_images = read_table(data_dl1_filename,
                               '/dl1/event/telescope/image/LST_LSTCam')
    # parameters:
    image_params = read_table(data_dl1_filename,
                              '/dl1/event/telescope/parameters/LST_LSTCam')

    # All events in a DL1 file should correspond to the same telescope,
    # just take it from the first event:
    tel_id = image_params['tel_id'][0]

    data_dl1_calibration = read_table(data_dl1_filename,
                                      '/dl1/event/telescope/monitoring/calibration')
    unusable = data_dl1_calibration['unusable_pixels']

    # Locate pixels with HG declared unusable either in original calibration or
    # in interleaved events:
    bad_pixels = unusable[0][0]  # original calibration
    for tf in unusable[1:][0]:  # calibrations with interleaved
        bad_pixels = np.logical_or(bad_pixels, tf)
    good_pixels = ~bad_pixels
    # First index:  1,2,... = values from interleaved (0 is for original
    # calibration run)
    # Second index: 0 = high gain
    # Third index: pixels

    # Now compute the average pixel charge in real pedestal events, for good
    # and "not too bright" pixels (the idea is to exclude stars, we want to
    # estimate the overall "diffuse" NSB level)
    interleaved_ped = image_params['event_type'] == event_type.value
    ped_pixq = camera_images['image'][interleaved_ped]
    ped_meanpixq = np.mean(ped_pixq, axis=0)
    ped_stdpixq = np.std(ped_pixq, axis=0)
    median_ped_meanpixq = np.median(ped_meanpixq[good_pixels])
    # Exclude the brightest pixels, which may be affected by stars:
    too_bright = (ped_meanpixq > median_ped_meanpixq + 3 * np.std(ped_meanpixq))
    good_pixels &= ~too_bright
    log.info(f'Good and not too bright pixels: {good_pixels.sum()}')
    # Recompute median:
    median_ped_meanpixq = np.median(ped_meanpixq[good_pixels])
    median_ped_stdpixq = np.median(ped_stdpixq[good_pixels])

    return median_ped_meanpixq, median_ped_stdpixq, tel_id



def calculate_required_additional_nsb(simtel_filename, data_dl1_filename, config=None):
    """
    Calculates the additional NSB needed in the MC waveforms
    to match a real data DL1 file

    Parameters
    ----------
    simtel_filename: a simtel file containing showers, from the production
    (same NSB and telescope settings) as the one on which the correction will
    be applied. It must contain pixel-wise info on true number of p.e.'s from
    C-photons (will be used to identify pixels which only contain noise).
    data_dl1_filename: a real data DL1 file (processed with calibration
    settings corresponding to those with which the MC is to be processed).
    It must contain calibrated images, i.e. "DL1a" data. This file has the
    "target" NSB which we want to have in the MC files, for better
    agreement of data and simulations.
    config: configuration containing the calibration
    settings used for processing both the data and the MC files above

    Returns
    -------
    extra_nsb: additional NSB rate in absolute units, p.e./ns (a.k.a. "GHz"),
    that has to be added in the MC to match the real data
    data_ped_variance: Pedestal variance from data
    mc_ped_variance: Pedestal variance from MC, AFTER THE TUNING!

    """

    log.setLevel(logging.INFO)

    if config is None:
        config = standard_config

    # Obtain the median (across camera) of mean (and std dev of) pixel charge
    # taking into account only healthy pixels, and excluding outliers.
    # We also get the tel_id to which the DL1 file corresponds:
    median_ped_meanpixq, median_ped_stdpixq, tel_id = get_pix_median_charges(
            data_dl1_filename, EventType.SKY_PEDESTAL)

    # Now we process the Monte Carlo:
    # Event reader for simtel file:
    with EventSource(input_url=simtel_filename, config=Config(config)) as mc_reader:
        subarray = mc_reader.subarray
    
        # Get the single-pe response fluctuations:
        spe_location = (config['waveform_nsb_tuning']['spe_location']
                        if 'spe_location' in config['waveform_nsb_tuning']
                           and config['waveform_nsb_tuning']['spe_location']
                           is not None
                        else get_resource_path(
                "data/SinglePhE_ResponseInPhE_expo2Gaus.dat"))
        spe = np.loadtxt(spe_location).T
        spe_integral = np.cumsum(spe[1])
        charge_spe_cumulative_pdf = interp1d(spe_integral, spe[0], kind='cubic',
                                             bounds_error=False, fill_value=0.,
                                             assume_sorted=True)
        # Pulse template shape for a single p.e.:
        pulse_templates = {tel_id: NormalizedPulseTemplate.load_from_eventsource(
                subarray.tel[tel_id].camera.readout, resample=True) for tel_id in
            config['source_config']['LSTEventSource']['allowed_tels']}
    
        # Since the pulse integrator will now be used on MC, we have to apply the
        # pulse integration correction (in data that is currently, as of
        # lstchain v0.10, replaced by an empirical (hard-coded) correction of the
        # adc to pe conversion factors )
        config['LocalPeakWindowSum']['apply_integration_correction'] = True
        r1_dl1_calibrator = CameraCalibrator(image_extractor_type=config[
            'image_extractor'], config=Config(config), subarray=subarray)
    
        numevents = 0
        maxmcevents = 200 # Enough statistics to determine the right NSB level
    
        # Simulated levels of additional NSB, rate in p.e./ns (a.k.a. GHz):
        total_added_nsb = np.array([0, 0.125, 0.25, 0.5, 1, 2, 4])
    
        # Just the product of added_nsb and original_nsb (first two arguments
        # below) is relevant! Units "GHz" (p.e./ns)
        nsb_tuner = [None]
        # We now create the instances of WaveformNsbTuner to add the different
        # levels of noise to the MC waveforms.
        # NOTE: the waveform gets updated every time, the noise addition is
        # cumulative (hence the np.diff):
    
        for nsb_rate in np.diff(total_added_nsb):
            # Create a dict to put the NSB value for each tel_id. It is an array for
            # the future case in which there are more telescopes, but for now it
            # just creates it with a single value, for tel_id
            nsb = {tel_id: nsb_rate * u.GHz for tel_id in
                   config['source_config']['LSTEventSource']['allowed_tels']}
            nsb_tuner.append(WaveformNsbTuner(nsb, pulse_templates,
                                              charge_spe_cumulative_pdf,
                                              pre_computed_multiplicity=10))
        # last argument means it will precompute 10 * 1855 (pixels) * 2 (gains)
        # noise waveforms to pick from during the actual introduction of the noise
    
        # List of lists to keep the integrated pixel charges for different NSB
        # levels:
        modified_integrated_charge = [[] for i in range(len(nsb_tuner))]

        for event in mc_reader:
            if tel_id not in event.trigger.tels_with_trigger:
                continue
            numevents += 1
            if numevents > maxmcevents:
                break

            # Calibrate the event to get the integrated charges (DL1a):
            r1_dl1_calibrator(event)

            selected_gains = event.r1.tel[tel_id].selected_gain_channel
            mask_high = (selected_gains == 0)
            true_image = event.simulation.tel[tel_id].true_image
            # Use only pixels with no Cherenkov signal, just noise:
            pedmask = mask_high & (true_image == 0)
        
            # First store the charges with no added NSB:
            modified_integrated_charge[0].extend(event.dl1.tel[tel_id].image[
                                                     pedmask])
        
            # Now add the different levels of NSB and recompute charges:
            for ii, tuner in enumerate(nsb_tuner[1:]):
                waveform = event.r1.tel[tel_id].waveform

                # NOTE!! The line below modifies the waveform in event.r1
                tuner.tune_nsb_on_waveform(waveform, tel_id, mask_high, subarray)
                r1_dl1_calibrator(event)
                modified_integrated_charge[ii + 1].extend(
                    event.dl1.tel[1].image[pedmask])

    modified_integrated_charge = np.array(modified_integrated_charge)
    # Fit the total added NSB rate vs. the average pixel charge:
    params, _ = curve_fit(custom_function,
                          np.mean(modified_integrated_charge, axis=1),
                          total_added_nsb, p0=[-0.2, 0.2, 2])

    # Obtain the right rate of NSB to be added to MC so that it matches the
    # data:
    extra_nsb = custom_function(median_ped_meanpixq,
                                params[0], params[1], params[2])

    # Since the level of noise in MC is low, it should hardly ever happen
    # that extra_nsb is negative, but just in case:
    extra_nsb = max(extra_nsb, 0)

    # Now open the MC file again and test that the tuning was successful:
    with EventSource(input_url=simtel_filename, config=Config(config)) as mc_reader:
        nsb = {tel_id: extra_nsb * u.GHz for tel_id in
               config['source_config']['LSTEventSource']['allowed_tels']}
        tuner = WaveformNsbTuner(nsb, pulse_templates,
                                 charge_spe_cumulative_pdf,
                                 pre_computed_multiplicity=10)
        final_mc_qped = []
        numevents = 0
        for event in mc_reader:
            if tel_id not in event.trigger.tels_with_trigger:
                continue
            numevents += 1
            if numevents > maxmcevents:
                break
            selected_gains = event.r1.tel[tel_id].selected_gain_channel
            mask_high = (selected_gains == 0)
            true_image = event.simulation.tel[tel_id].true_image
            pedmask = mask_high & (true_image == 0)
    
            waveform = event.r1.tel[tel_id].waveform
            tuner.tune_nsb_on_waveform(waveform, tel_id, mask_high, subarray)
            r1_dl1_calibrator(event)
            final_mc_qped.extend(event.dl1.tel[tel_id].image[pedmask])

    data_ped_variance = median_ped_stdpixq**2
    mc_ped_variance = np.std(final_mc_qped)**2

    return extra_nsb, data_ped_variance, mc_ped_variance


def custom_function(x, a, b, c):
    """
    Custom function to fit the "mean pix charge in pedestal events" vs. the
    added level of NSB in waveforms
    """
    return a + b * x ** c

class WaveformNsbTuner:
    """
    Handles the injection of additional NSB pulses in waveforms.
    """
    def __init__(self, added_nsb, pulse_template, charge_spe_cumulative_pdf,
                 pre_computed_multiplicity=10):
        """
        Parameters
        ----------
        added_nsb: dict [`float`,`astropy.units.Quantity`]
            NSB frequency (GHz) to be added to the waveforms (per tel_id)
        pulse_template: `dict`[int,`lstchain.data.NormalizedPulseTemplate`]
            Single photo-electron pulse template per telescope
        charge_spe_cumulative_pdf: `scipy.interpolate.interp1d`
            Cumulative amplitude probability density function for single photo-electrons
        pre_computed_multiplicity: `int`
            Multiplicative factor on the number of pixels used to determine the number of pre-generated, nsb-only,
            waveforms. Later used during event modification. Set to 0 to always compute the correction on the fly.

        """
        self.added_nsb = added_nsb
        self.pulse_template = pulse_template
        self.charge_spe_cumulative_pdf = charge_spe_cumulative_pdf
        self.multiplicity = pre_computed_multiplicity

        # Number of extra time sample added at the start of the time window used to inject NSB pulses
        # It should be large enough to account for pulses created by NSB hits before the recorded time window
        self.extra_samples = 25

        self.nsb_waveforms = {}
        self.nb_simulated = {}
        self.rng = np.random.default_rng()
        if self.multiplicity == 0:
            self.tune_nsb_on_waveform = self._tune_nsb_on_waveform_direct
        else:
            self.tune_nsb_on_waveform = self._tune_nsb_on_waveform_precomputed

    def initialise_waveforms(self, waveform, dt, tel_id):
        """
        Creates an array of nsb only waveforms later used to inject additional nsb in events.
        Called once for each telescope id at first encounter in `self._tune_nsb_on_waveform_precomputed`.

        Parameters
        ----------
        waveform: ndarray
            Waveform used to know the number of pixels and time samples for a given telescope id
        dt: `astropy.units.Quantity`
            Time between samples
        tel_id: `int`
            Telescope identifier for which nsb waveforms are generated

        """
        log.info(f"Pre-generating nsb waveforms for nsb tuning and telescope id {tel_id}.")
        _, n_pixels, n_samples = waveform.shape
        baseline_correction = -(self.added_nsb[tel_id] * dt).to_value("")
        nsb_waveforms = np.full((self.multiplicity * n_pixels, 2, n_samples), baseline_correction)
        duration = (self.extra_samples + n_samples) * dt
        t = np.arange(-self.extra_samples, n_samples) * dt.to_value(u.ns)
        mean_added_nsb = (self.added_nsb[tel_id] * duration).to_value("")
        additional_nsb = self.rng.poisson(mean_added_nsb, self.multiplicity * n_pixels)
        added_nsb_time = self.rng.uniform(-self.extra_samples * dt.to_value(u.ns),
                                          -self.extra_samples * dt.to_value(u.ns) + duration.to_value(u.ns),
                                          (self.multiplicity * n_pixels, max(additional_nsb)))
        added_nsb_amp = self.charge_spe_cumulative_pdf(
            self.rng.uniform(size=(self.multiplicity * n_pixels, max(additional_nsb))))
        nsb_waveforms[:, 0, :] += nsb_only_waveforms(
            time=t[self.extra_samples:],
            is_high_gain=np.zeros(self.multiplicity * n_pixels),
            additional_nsb=additional_nsb,
            amplitude=added_nsb_amp,
            t_0=added_nsb_time,
            t0_template=self.pulse_template[tel_id].t0,
            dt_template=self.pulse_template[tel_id].dt,
            a_hg_template=self.pulse_template[tel_id].amplitude_HG,
            a_lg_template=self.pulse_template[tel_id].amplitude_LG
        )
        nsb_waveforms[:, 1, :] += nsb_only_waveforms(
            time=t[self.extra_samples:],
            is_high_gain=np.ones(self.multiplicity * n_pixels),
            additional_nsb=additional_nsb,
            amplitude=added_nsb_amp,
            t_0=added_nsb_time,
            t0_template=self.pulse_template[tel_id].t0,
            dt_template=self.pulse_template[tel_id].dt,
            a_hg_template=self.pulse_template[tel_id].amplitude_HG,
            a_lg_template=self.pulse_template[tel_id].amplitude_LG
        )
        self.nsb_waveforms[tel_id] = nsb_waveforms
        self.nb_simulated[tel_id] = self.multiplicity * n_pixels

    def _tune_nsb_on_waveform_precomputed(self, waveform, tel_id, is_high_gain, subarray):
        """
        Inject single photon pulses in existing R1 waveforms to increase NSB using pre-computed pure nsb waveforms.

        Parameters
        ----------
        waveform: ndarray
            Charge (p.e. / ns) in each pixel and sampled time
        tel_id: `int`
            Telescope id associated to the waveform to tune
        is_high_gain: ndarray of boolean
            Gain channel used per pixel: True=hg, False=lg
        subarray: `ctapipe.instrument.subarray.SubarrayDescription`

        """
        if tel_id not in self.nsb_waveforms:
            readout = subarray.tel[tel_id].camera.readout
            sampling_rate = readout.sampling_rate
            dt = (1.0 / sampling_rate).to(u.s)
            self.initialise_waveforms(waveform, dt, tel_id)
        _, n_pixels, _ = waveform.shape
        # The nsb_waveform array is randomised along the first axis,
        # then the n_pixels first elements with correct gain are used for the injection
        self.rng.shuffle(self.nsb_waveforms[tel_id])
        waveform[0] += ((is_high_gain == 0)[:, np.newaxis] * self.nsb_waveforms[tel_id][:n_pixels, 0] +
                        (is_high_gain == 1)[:, np.newaxis] * self.nsb_waveforms[tel_id][:n_pixels, 1])

    def _tune_nsb_on_waveform_direct(self, waveform, tel_id, is_high_gain, subarray):
        """
        Inject single photon pulses in existing R1 waveforms to increase NSB.

        Parameters
        ----------
        waveform: ndarray
            Charge (p.e. / ns) in each pixel and sampled time
        tel_id: `int`
            Telescope id associated to the waveform to tune
        is_high_gain: ndarray of boolean
            Gain channel used per pixel: True=hg, False=lg
        subarray: `ctapipe.instrument.subarray.SubarrayDescription`

        """
        _, n_pixels, n_samples = waveform.shape
        readout = subarray.tel[tel_id].camera.readout
        sampling_rate = readout.sampling_rate
        dt = (1.0 / sampling_rate).to(u.s)
        duration = (self.extra_samples + n_samples) * dt
        t = np.arange(-self.extra_samples, n_samples) * dt.to_value(u.ns)
        mean_added_nsb = (self.added_nsb[tel_id] * duration).to_value("")
        additional_nsb = self.rng.poisson(mean_added_nsb, n_pixels)
        added_nsb_time = self.rng.uniform(-self.extra_samples * dt.to_value(u.ns),
                                          -self.extra_samples * dt.to_value(u.ns) + duration.to_value(u.ns),
                                          (n_pixels, max(additional_nsb)))
        added_nsb_amp = self.charge_spe_cumulative_pdf(self.rng.uniform(size=(n_pixels, max(additional_nsb))))
        baseline_correction = (self.added_nsb[tel_id] * dt).to_value("")
        waveform -= baseline_correction
        waveform += nsb_only_waveforms(
            time=t[self.extra_samples:],
            is_high_gain=is_high_gain,
            additional_nsb=additional_nsb,
            amplitude=added_nsb_amp,
            t_0=added_nsb_time,
            t0_template=self.pulse_template[tel_id].t0,
            dt_template=self.pulse_template[tel_id].dt,
            a_hg_template=self.pulse_template[tel_id].amplitude_HG,
            a_lg_template=self.pulse_template[tel_id].amplitude_LG
        )
