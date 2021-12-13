import numpy as np

__all__ = ['apply_dynamic_cleaning']



def apply_dynamic_cleaning(image, signal_pixels, threshold, fraction):
    """

    Parameters
    ----------
    image      pixel charges
    mask       pixels selected by previous cleaning
    threshold  minimum average charge in the 3 brightest pixels to apply
               the dynamic cleaning (else nothing is done)
    fraction   pixels below fraction * (average charge in the 3 brightest
               pixels) will be removed from the cleaned image

    Returns    a mask with the selected pixels after the dynamic cleaning
    -------

    """
    max_3_value_index = np.argsort(image)[-3:]
    mean_3_max_signal = np.mean(image[max_3_value_index])

    if mean_3_max_signal < threshold:
        return signal_pixels

    dynamic_threshold = fraction * mean_3_max_signal
    mask_dynamic_cleaning = (image >= dynamic_threshold) & signal_pixels

    return mask_dynamic_cleaning
