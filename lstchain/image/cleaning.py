import glob
import logging
import numpy as np
from pathlib import Path
from lstchain.paths import run_to_dl1_filename
from lstchain.io.io import dl1_params_lstcam_key, dl1_images_lstcam_key
from lstchain.io.io import dl1_params_tel_mon_cal_key
from lstchain.io.config import get_standard_config
from ctapipe.containers import EventType

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

    # subrun-wise dl1 file names:
    dl1_filenames = Path(input_dir,
                         run_to_dl1_filename(1, run_number, 0).replace(
                                 '.0000.h5', '.????.h5'))
    all_dl1_files = glob.glob(str(dl1_filenames))
    all_dl1_files.sort()

    # Aprox. maximum number of subruns (uniformly distributed through the
    # run) to be processed:
    max_number_of_processed_subruns = 10
    # Keep only ~max_number_of_processed_subruns subruns, distributed
    # along the run:
    dl1_files = all_dl1_files[::int(1+len(all_dl1_files) /
                                    max_number_of_processed_subruns)]

    number_of_pedestals = []
    usable_pixels = []
    median_ped_mean_pix_charge = []

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
        mean_ped_charge = np.mean(charges_pedestals, axis=0)
        median_ped_mean_pix_charge.append(np.median(mean_ped_charge[
                                                        reliable_pixels]))

    median_ped_mean_pix_charge = np.array(median_ped_mean_pix_charge)
    number_of_pedestals = np.array(number_of_pedestals)

    # Now compute the median for all processed subruns, which is more robust
    # against e.g. subruns affected by car flashes. We also exclude subruns
    # which have less than half of the median statistics per subrun.
    good_stats = number_of_pedestals > 0.5 * np.median(number_of_pedestals)
    qped = np.median(median_ped_mean_pix_charge[good_stats])

    picture_threshold = pic_th(qped)
    boundary_threshold = picture_threshold / 2

    # We now create a .json files with recommended image cleaning
    # settings for lstchain_dl1ab.
    newconfig = get_standard_config()['tailcuts_clean_with_pedestal_threshold']
    # casts below are needed, json does not like numpy's int64:
    newconfig['picture_thresh'] = int(picture_threshold)
    newconfig['boundary_thresh'] = int(boundary_threshold)

    return newconfig


def pic_th(mean_ped):
    """
    mean_ped: mean pixel charge in pedestal events (for the standard
    LocalPeakWindowSearch algo & settings in lstchain)

    Returns:
        recommended picture threshold for image cleaning (from a table)
    """
    mp_edges = np.array([2.4, 3.1, 3.8, 4.5, 5.2])
    picture_threshold = np.array([8, 10, 12, 14, 16, 18])

    if mean_ped >= mp_edges[-1]:
        return picture_threshold[-1]
    return picture_threshold[np.where(mp_edges > mean_ped)[0][0]]
