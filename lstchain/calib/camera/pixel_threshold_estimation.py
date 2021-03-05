import tables
import numpy as np
from ctapipe_io_lst.constants import HIGH_GAIN
from lstchain.io.io import dl1_params_tel_mon_ped_key, dl1_params_tel_mon_cal_key


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
        ped_charge_mean = np.array(ped.cols.charge_mean)
        ped_charge_std = np.array(ped.cols.charge_std)
        calib = f.root[dl1_params_tel_mon_cal_key]
        dc_to_pe = np.array(calib.cols.dc_to_pe)
        ped_charge_mean_pe = ped_charge_mean * dc_to_pe
        ped_charge_std_pe = ped_charge_std * dc_to_pe

    return ped_charge_mean_pe, ped_charge_std_pe

def get_threshold_from_dl1_file(dl1_path, sigma_clean):
    """
    Function to get picture threshold from dl1 from interleaved pedestal events.
    Return modified picture threshold for tailcut cleaning method.
    Allow cleaning the most noisy pixels (for example around the star location).
    Threshold for each pixel is define as:
        threshold = pedestal_bias + sigma * pedestal_std.
    Recommended threshold for cleaning:
        galactic source: picture_thresh=8, boundary_thresh=4, sigma=3
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
    if ped_std_pe.shape[0] == 2:
        interleaved_events_id = 1
    else:
        interleaved_events_id = 0
    threshold_clean_pe = ped_mean_pe + sigma_clean*ped_std_pe
    # find pixels with std = 0 and mean = 0 <=> dead pixels in interleaved
    # pedestal event likely due to stars
    unusable_pixels = get_unusable_pixels(dl1_path, interleaved_events_id)
    # for dead pixels set max value of threshold
    threshold_clean_pe[interleaved_events_id, HIGH_GAIN, unusable_pixels] = \
        max(threshold_clean_pe[interleaved_events_id, HIGH_GAIN, :])
    # return pedestal interleaved threshold from data run for high gain
    return threshold_clean_pe[interleaved_events_id, HIGH_GAIN, :]

def get_unusable_pixels(dl1_path, interleaved_events_id):
    with tables.open_file(dl1_path) as f:
        unusable_pixels = np.where(f.root[dl1_params_tel_mon_cal_key].col(
                                   'unusable_pixels')[interleaved_events_id,
                                                      HIGH_GAIN,
                                                      :] == True)
    return unusable_pixels
