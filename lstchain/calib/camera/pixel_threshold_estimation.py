import tables
import numpy as np
from ctapipe_io_lst.constants import HIGH_GAIN
from lstchain.io.io import dl1_params_tel_mon_ped_key, dl1_params_tel_mon_cal_key

from lstchain.io.config import get_standard_config
from lstchain.io.config import read_configuration_file, replace_config

ORIGINAL_CALIBRATION_ID = 0
INTERLEAVED_CALIBRATION_ID = 1

def get_bias_and_std(dl1_file):
    """
    Function to extract bias and std of pedestal from interleaved events from dl1 file.
    Parameters
    ----------
    input_filename: str
        path to dl1 file
    Returns
    -------
    bias, std: np.ndarray, np.ndarray
        bias and std in p.e.
    """
    with tables.open_file(dl1_file) as f:
        ped = f.root[dl1_params_tel_mon_ped_key]
        ped_charge_mean = ped.col('charge_mean')
        ped_charge_std = ped.col('charge_std')
        calib = f.root[dl1_params_tel_mon_cal_key]
        dc_to_pe = calib.col('dc_to_pe')[ORIGINAL_CALIBRATION_ID]
        ped_charge_mean_pe = ped_charge_mean * dc_to_pe
        ped_charge_std_pe = ped_charge_std * dc_to_pe

    return ped_charge_mean_pe, ped_charge_std_pe

def get_threshold_from_dl1_file(dl1_path, sigma_clean):
    """
    Function to get picture threshold from dl1 from interleaved pedestal events.
    Return modified picture threshold for tailcut cleaning method.
    Allow cleaning the most noisy pixels (for example around the star location).
    Threshold for each pixel is defined as:
        threshold = pedestal_bias + sigma * pedestal_std.
    Recommended threshold for cleaning:
        galactic source: picture_thresh=8, boundary_thresh=4, sigma=2.5
        extragalactic source: picture_thresh=6, boundary_thresh=3, sigma=2.5

    Parameters
    ----------
    input_filename: str
        path to dl1 file
    sigma_clean: float
        cleaning level parameter
    Returns
    -------
    picture_thresh: np.ndarray
        picture threshold calculated using interleaved pedestal events
    """

    ped_mean_pe, ped_std_pe = get_bias_and_std(dl1_path)

    # If problem with interleaved pedestal std values occur, take pedestal
    # std values from calibration run.
    # Correct interleaved pedestal std array should have shape (2,2,1855)
    if ped_std_pe.shape[0] > 1:
        pedestal_id = INTERLEAVED_CALIBRATION_ID
    else:
        pedestal_id = ORIGINAL_CALIBRATION_ID
    threshold_clean_pe = ped_mean_pe + sigma_clean*ped_std_pe
    # find pixels with std = 0 and mean = 0 <=> dead pixels in interleaved
    # pedestal event likely due to stars
    unusable_pixels = get_unusable_pixels(dl1_path, pedestal_id)
    # for dead pixels set max value of threshold
    threshold_clean_pe[pedestal_id, HIGH_GAIN, unusable_pixels] = \
        max(threshold_clean_pe[pedestal_id, HIGH_GAIN, :])
    # return pedestal interleaved threshold from data run for high gain
    return threshold_clean_pe[pedestal_id, HIGH_GAIN, :]

def get_unusable_pixels(dl1_path, pedestal_id):
    with tables.open_file(dl1_path) as f:
        calibration_id = f.root.dl1.event.telescope.monitoring.pedestal.col('calibration_id')
        unusable_pixels = np.where(f.root[dl1_params_tel_mon_cal_key].col(
                                   'unusable_pixels')[calibration_id[pedestal_id],
                                                      HIGH_GAIN,
                                                      :] == True)
    return unusable_pixels


def find_safe_threshold_from_dl1_file(dl1_path, config_file=None,
                                      max_fraction=0.04):
    """
    Function to obtain an integer value for the picture threshold such that
    at most a fraction max_fraction of pixels have a higher value resulting
    from the "clean_with_pedestal_threshold" cleaning approach. That approach
    increases the cleaning picture threshold of a pixel to, say, 2.5 standard
    deviations (or the number in "sigma" below) above its pedestal mean,
    hence pixels illuminated by stars get a higher threshold, and so we avoid
    too many spurious signals from the starlight fluctuations. The downside of
    the method is that it introduces non-uniformities in the camera response
    for the real data, and therefore data-MC discrepancies (there are no stars
    in MC and all pixels have the same cleaning settings).

    Here we try to calculate what should be the "base" picture threshold (to
    use both in MC and data) so that at most a given fraction "max_fraction"
    of the camera gets an increased threshold via the
    clean_with_pedestal_threshold condition. In this way the number /
    extension of camera inhomogeneities in the real data is limited. By
    default the max_fraction is 0.04, which e.g. gives a picture threshold
    around 8 for the Crab field (in no-moon conditions).

    Note: we want cleaning settings for a whole run, so we will have to run
    the function over sub-runs and average the values, or make the function
    able to read multiple DL1 files

    Parameters
    ----------
    dl1_path: real data DL1 file, to have access to the monitoring table
    where we can read the pedesta bias & std dev estimated with interleaved
    pedestal events

    config_file: must be the one used for the analysis of the real data,
    in particular, it has to contain the tailcuts_clean_with_pedestal_threshold
    setting (sigma) that one wants to use

    max_fraction: maximum fraction of camera pixels that are allowed to get a
    picture threshold above the base one. That is, we calculate here the base
    picture threshold that will ensure that the condition is fulfilled

    Returns
    -------
    (scalar) the value of the picture threshold that has to be used in data and
    MC to ensure that no more than max_fraction of the camera gets an
    increased value via tailcuts_clean_with_pedestal_threshold

    """
    std_config = get_standard_config()
    if config_file is not None:
        config = replace_config(std_config,
                                read_configuration_file(config_file))
    else:
        config = std_config

    cleaning_method = 'tailcuts_clean_with_pedestal_threshold'
    sigma = config[cleaning_method]['sigma']

    # Obtain the picture thresholds of pixels based on the "clean with
    # pedestal threshold" method:
    pic_threshold = get_threshold_from_dl1_file(dl1_path, sigma)
    threshold_sorted = np.sort(pic_threshold)

    # find the value new_threshold above which a fraction max_fraction of
    # pixels lies:
    index = int(len(threshold_sorted) * (1 - max_fraction))
    new_threshold = threshold_sorted[index]

    # Return the first integer value above new_threshold (to avoid too many
    # different cleaning settings in different runs):

    return np.ceil(new_threshold)
