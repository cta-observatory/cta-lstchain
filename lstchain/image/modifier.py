
import logging

import numpy as np
from ctapipe.calib.camera import CameraCalibrator
from ctapipe.io import (
    EventSource,
    read_table,
)
from numba import njit
from scipy.interpolate import interp1d
from traitlets.config import Config

from lstchain.io import standard_config
from lstchain.io.config import read_configuration_file
from lstchain.reco.reconstructorCC import nsb_only_waveforms

__all__ = [
    'add_noise_in_pixels',
    'calculate_required_additional_nsb',
    'calculate_noise_parameters',
    'random_psf_smearer',
    'set_numba_seed',
    'tune_nsb_on_waveform',
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
    data_dl1_pedestal =  read_table(data_dl1_filename,
                    '/dl1/event/telescope/monitoring/pedestal')
    data_dl1_flatfield =  read_table(data_dl1_filename,
                    '/dl1/event/telescope/monitoring/flatfield')
    data_dl1_parameters =  read_table(data_dl1_filename,
                    '/dl1/event/telescope/parameters/LST_LSTCam')
    data_dl1_image = read_table(data_dl1_filename,
                    '/dl1/event/telescope/image/LST_LSTCam')

    unusable = data_dl1_calibration['unusable_pixels']
    # Locate pixels with HG declared unusable either in original calibration or
    # in interleaved events:
    bad_pixels = unusable[0][0]  # original calibration
    if unusable.shape[0] > 1:
        for tf in unusable[1:][0]:   # calibrations with interleaveds
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

    if data_dl1_pedestal['charge_std'].shape[0] < 2 :
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
    data_HG_FF_mean_pe = np.mean(dummy, axis=0) # one value per pixel

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
    data_HG_ped_std_pe = np.mean(dummy, axis=0) # one value per pixel

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
    func = interp1d(0.5*(dataq[1][1:]+dataq[1][:-1]), dataq[0],
                    kind='quadratic', fill_value='extrapolate')
    xx = np.linspace(qrange[0], qrange[1], 100*qbins)
    mode_data = xx[np.argmax(func(xx))]

    # Event reader for simtel file:
    mc_reader = EventSource(input_url=simtel_filename, config=Config(config))

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
    func = interp1d(0.5*(mcq[1][1:]+mcq[1][:-1]), mcq[0],
                    kind='quadratic', fill_value='extrapolate')
    xx = np.linspace(qrange[0], qrange[1], 100*qbins)
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
        ((data_median_std_ped_pe**2 - mc_unbiased_std_ped_pe**2) *
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
    added_noise = (np.sum(dq[dq<maxq]**2)/len(dq[dq<maxq]) -
                   np.sum(dqmc[dqmc<maxq]**2)/len(dqmc[dqmc < maxq]))
    extra_noise_in_dim_pixels = max(0., added_noise)

    return extra_noise_in_dim_pixels, extra_bias_in_dim_pixels, \
           extra_noise_in_bright_pixels


def tune_nsb_on_waveform(waveform, added_nsb_fraction, original_nsb,
                         dt, pulse_templates, gain, charge_spe_cumulative_pdf):
    """
    Inject single photon pulses in existing R1 waveforms to increase NSB.
    Parameters
    ----------
    waveform: charge (p.e. / ns) in each pixel and sampled time
    added_nsb_fraction: fraction of the original NSB in simulation to be added
    original_nsb: original NSB rate (astropy unit Hz)
    dt: time between waveform samples (astropy unit s)
    pulse_templates: `lstchain.data.NormalizedPulseTemplate` containing
    the single p.e. pulse template used for the injection
    gain: gain channel identifier for each pixel
    charge_spe_cumulative_pdf: `scipy.interpolate.interp1d` Single p.e. gain
    fluctuation cumulative pdf used to randomise the normalisation of
    injected pulses
    """
    n = 25
    n_pixels, n_samples = waveform.shape
    duration = (n + n_samples) * dt  # TODO check needed time window, effect of edges
    t = np.arange(-n, n_samples) * dt.value
    mean_added_nsb = (added_nsb_fraction * original_nsb) * duration
    rng = np.random.default_rng()
    additional_nsb = rng.poisson(mean_added_nsb, n_pixels)
    added_nsb_time = rng.uniform(-n * dt.value, -n * dt.value + duration.value, (n_pixels, max(additional_nsb)))
    added_nsb_amp = charge_spe_cumulative_pdf(rng.uniform(size=(n_pixels, max(additional_nsb))))
    baseline_correction = (added_nsb_fraction * original_nsb * dt).value
    waveform -= baseline_correction
    waveform += nsb_only_waveforms(
        time=t[n:],
        is_high_gain=gain,
        additional_nsb=additional_nsb,
        amplitude=added_nsb_amp,
        t_0=added_nsb_time,
        t0_template=pulse_templates.t0,
        dt_template=pulse_templates.dt,
        a_hg_template=pulse_templates.amplitude_HG,
        a_lg_template=pulse_templates.amplitude_LG
    )

def calculate_required_additional_nsb(simtel_filename, data_dl1_filename, config=None):
    # TODO check if good estimation
    # TODO reduce duplicated code with 'calculate_noise_parameters'
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
    extra_nsb: Fraction of the additional NSB in data compared to MC.
    data_ped_variance: Pedestal variance from data
    mc_ped_variance: Pedestal variance from MC
    """

    log.setLevel(logging.INFO)

    if config is None:
        config = standard_config

    # Real data DL1 tables:
    data_dl1_calibration = read_table(data_dl1_filename,
                                      '/dl1/event/telescope/monitoring/calibration')
    data_dl1_pedestal = read_table(data_dl1_filename,
                                   '/dl1/event/telescope/monitoring/pedestal')
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

    # HG adc to pe conversion factors from interleaved calibrations:
    data_HG_dc_to_pe = data_dl1_calibration['dc_to_pe'][:, 0, :]
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
    data_median_std_ped_pe = np.median(data_HG_ped_std_pe)
    data_std_std_ped_pe = np.std(data_HG_ped_std_pe)
    log.info(f'Real data: median across camera of good pixels\' pedestal std '
             f'{data_median_std_ped_pe:.3f} p.e.')
    brightness_limit = data_median_std_ped_pe + 3 * data_std_std_ped_pe
    too_bright_pixels = (data_HG_ped_std_pe > brightness_limit)
    log.info(f'Number of pixels beyond 3 std dev of median: '
             f'{too_bright_pixels.sum()}, (above {brightness_limit:.2f} p.e.)')

    # Exclude too bright pixels, besides those with unusable calibration:
    good_pixels &= ~too_bright_pixels
    # recalculate the median of the pixels' std dev, with good_pixels:
    data_median_std_ped_pe = np.median(data_HG_ped_std_pe[good_pixels])

    log.info(f'Good and not too bright pixels: {good_pixels.sum()}')

    # Event reader for simtel file:
    mc_reader = EventSource(input_url=simtel_filename, config=Config(config))

    # Obtain the configuration with which the pedestal calculations were
    # performed:
    ped_config = config['LSTCalibrationCalculator']['PedestalIntegrator']
    tel_id = ped_config['tel_id']
    # Obtain the (unbiased) extractor used for pedestal calculations:
    pedestal_calibrator = CameraCalibrator(
        image_extractor_type=ped_config['charge_product'],
        config=Config(config['LSTCalibrationCalculator']),
        subarray=mc_reader.subarray)

    # Since these extractors are now for use on MC, we have to apply the pulse
    # integration correction (in data that is currently, as of
    # lstchain v0.7.5, replaced by an empirical (hard-coded) correction of the
    # adc to pe conversion factors )
    pedestal_calibrator.image_extractors[ped_config['charge_product']].apply_integration_correction = True

    # MC pedestals integrated with the unbiased pedestal extractor
    mc_ped_charges = []

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

    # All pixels behave (for now) in the same way in MC, just put them together
    mc_ped_charges = np.concatenate(mc_ped_charges)
    mc_unbiased_std_ped_pe = np.std(mc_ped_charges)

    # Find the additional noise (in data w.r.t. MC) for the unbiased extractor
    # The idea is that pedestal variance scales with NSB

    data_ped_variance = data_median_std_ped_pe ** 2
    mc_ped_variance = mc_unbiased_std_ped_pe ** 2
    extra_nsb = ((data_ped_variance - mc_ped_variance)
                 / mc_ped_variance)
    return extra_nsb, data_ped_variance, mc_ped_variance
