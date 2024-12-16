import glob
import logging
import numpy as np
from pathlib import Path
from lstchain.paths import run_to_dl1_filename
from lstchain.io.io import dl1_params_lstcam_key, dl1_images_lstcam_key
from lstchain.io.io import dl1_params_tel_mon_cal_key
from lstchain.io.config import get_standard_config
from ctapipe.containers import EventType
from scipy.stats import median_abs_deviation

from ctapipe.io import read_table
__all__ = ['apply_dynamic_cleaning',
           'find_tailcuts',
           'pic_th']

log = logging.getLogger(__name__)


def apply_dynamic_cleaning(image, signal_pixels, threshold, fraction):
    """
    Application of the dynamic cleaning

    Parameters
    ----------
    image: `np.ndarray`
          Pixel charges
    signal_pixels
    threshold: `float`
        Minimum average charge in the 3 brightest pixels to apply
        the dynamic cleaning (else nothing is done)
    fraction: `float`
        Pixels below fraction * (average charge in the 3 brightest pixels)
        will be removed from the cleaned image

    Returns
    -------
    mask_dynamic_cleaning: `np.ndarray`
        Mask with the selected pixels after the dynamic cleaning

    """

    max_3_value_index = np.argsort(image)[-3:]
    mean_3_max_signal = np.mean(image[max_3_value_index])

    if mean_3_max_signal < threshold:
        return signal_pixels

    dynamic_threshold = fraction * mean_3_max_signal
    mask_dynamic_cleaning = (image >= dynamic_threshold) & signal_pixels

    return mask_dynamic_cleaning


def find_tailcuts(input_dir, run_number):
    """
    This functions uses DL1 files to determine tailcuts which are adequate
    for the bulk of the pixels in a given run. It does so simply based on the
    median (for the whole camera) of the median pixel charge for pedestal
    events.
    For reasons of stability & simplicity of analysis, we cannot decide the
    cleaning levels on a subrun-by-subrun basis. We select values which are
    valid for the whole run.
    The script also returns the suggested NSB adjustment needed in the
    "dark-sky" MC to match the data.
    The script will process a subset of the subruns (~10, hardcoded) of the run,
    distributed uniformly through it.

    Parameters
    ----------
    input_dir: `Path`
        directory where the DL1 files (subrun-wise, i,e, including
        DL1a) are stored

    run_number: ìnt`
        run number to be processed

    Returns
    -------
        additional_nsb_rate: p.e./ns rate of NSB to be added to "dark MC" to
        match the noise in the data
        newconfig (dict): cleaning configuration for running the DL1ab stage
    """

    log.setLevel(logging.INFO)

    # subrun-wise dl1 file names:
    dl1_filenames = Path(input_dir,
                         run_to_dl1_filename(1, run_number, 0).replace(
                                 '.0000.h5', '.????.h5'))
    all_dl1_files = glob.glob(str(dl1_filenames))
    all_dl1_files.sort()

    # Number of median absolute deviations (mad) away from the median that a
    # value has to be to be considered an outlier:
    mad_max = 5  # would exclude 8e-4 of the pdf for a gaussian

    # Aprox. maximum number of subruns (uniformly distributed through the
    # run) to be processed:
    max_number_of_processed_subruns = 10
    # Keep only ~max_number_of_processed_subruns subruns, distributed
    # along the run:
    dl1_files = all_dl1_files[::int(1+len(all_dl1_files) /
                                    max_number_of_processed_subruns)]

    number_of_pedestals = []
    usable_pixels = []
    median_ped_median_pix_charge = []

    for dl1_file in dl1_files:
        log.info('\nInput file: %s', dl1_file)

        data_parameters = read_table(dl1_file, dl1_params_lstcam_key)
        event_type_data = data_parameters['event_type'].data
        pedestal_mask = event_type_data == EventType.SKY_PEDESTAL.value

        number_of_pedestals.append(pedestal_mask.sum())
        data_images = read_table(dl1_file, dl1_images_lstcam_key)
        data_calib = read_table(dl1_file, dl1_params_tel_mon_cal_key)
        # data_calib['unusable_pixels'] , indices: (Gain  Calib_id  Pixel)

        # Get the "unusable" flags from the pedcal file:
        unusable_hg = data_calib['unusable_pixels'][0][0]
        unusable_lg = data_calib['unusable_pixels'][0][1]

        reliable_pixels = ~(unusable_hg | unusable_lg)
        usable_pixels.append(reliable_pixels)

        charges_data = data_images['image']
        charges_pedestals = charges_data[pedestal_mask]
        # pixel-wise median charge through the subrun:
        median_pix_charge = np.nanmedian(charges_pedestals, axis=0)
        median_pix_charge_dev = median_abs_deviation(charges_pedestals,
                                                     axis=0,
                                                     nan_policy='omit')
        # Just a cut to remove outliers:
        outliers = (np.abs(charges_pedestals - median_pix_charge) /
                    median_pix_charge_dev) > mad_max
        if outliers.sum() > 0:
            removed_fraction = outliers.sum() / outliers.size
            log.info(f'    Removed {removed_fraction:.2%} of pixels (outliers) '
                     f'from pedestal median calculation')
            # Replace outliers by nans:
            charges_pedestals = np.where(outliers, np.nan, charges_pedestals)
            # Recompute the pixel-wise medians ignoring the outliers:
            median_pix_charge = np.nanmedian(charges_pedestals, axis=0)

        # Now compute the median (for the whole camera) of the medians (for
        # each pixel) of the charges in pedestal events. Use only reliable
        # pixels for this:
        median_ped_median_pix_charge.append(np.nanmedian(median_pix_charge[
                                                             reliable_pixels]))

    # convert to ndarray:
    median_ped_median_pix_charge = np.array(median_ped_median_pix_charge)
    number_of_pedestals = np.array(number_of_pedestals)

    # Now compute the median for all processed subruns, which is more robust
    # against e.g. subruns affected by car flashes. We exclude subruns
    # which have less than half of the median statistics per subrun.
    good_stats = number_of_pedestals > 0.5 * np.median(number_of_pedestals)
    qped = np.nanmedian(median_ped_median_pix_charge[good_stats])
    # Now we also remove outliers if any:
    qped_dev = median_abs_deviation(median_ped_median_pix_charge[good_stats])
    not_outlier = (np.abs(median_ped_median_pix_charge - qped) /
                   qped_dev) < mad_max

    if (~not_outlier).sum() > 0:
        log.info(f'    Removed {(~not_outlier).sum()} outlier subruns '
                 f'(out of {not_outlier.size}) from pedestal median '
                 f'calculation')
        # recompute without outliers:
        qped = np.nanmedian(median_ped_median_pix_charge[good_stats & not_outlier])

    picture_threshold = pic_th(qped)
    boundary_threshold = picture_threshold / 2

    # We now create a .json files with recommended image cleaning
    # settings for lstchain_dl1ab.
    newconfig = get_standard_config()['tailcuts_clean_with_pedestal_threshold']
    # casts below are needed, json does not like numpy's int64:
    newconfig['picture_thresh'] = int(picture_threshold)
    newconfig['boundary_thresh'] = int(boundary_threshold)

    additional_nsb_rate = get_nsb(qped)

    return additional_nsb_rate, newconfig


def pic_th(mean_ped):
    """
    Parameters
    ----------
    mean_ped: `float`
        mean pixel charge in pedestal events (for the standard
        LocalPeakWindowSearch algo & settings in lstchain)

    Returns
    -------
        `int`
        recommended picture threshold for image cleaning (from a table)

    """
    mp_edges = np.array([2.4, 3.1, 3.8, 4.5, 5.2])
    picture_threshold = np.array([8, 10, 12, 14, 16, 18])

    if mean_ped >= mp_edges[-1]:
        return picture_threshold[-1]
    return picture_threshold[np.where(mp_edges > mean_ped)[0][0]]


def get_nsb(median_ped):
    """
    Parameters
    ----------
    median_ped: `float`
        median pixel charge in pedestal events

    Returns
    -------
        `float`
        (from a parabolic parametrization) the recommended additional NSB
        (in p.e. / ns) that has to be added to the "dark MC" waveforms in
        order to match the data for which the median pedestal charge is
        median_ped

    """
    params = [1.63991237, 0.3130943, 0.07311759]
    return (params[1] * (median_ped - params[0]) +
            params[2] * (median_ped - params[0]) ** 2)
