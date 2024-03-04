import numpy as np

from ctapipe.containers import ArrayEventContainer
from ctapipe.core.traits import (
    FloatTelescopeParameter,
    IntTelescopeParameter,
    BoolTelescopeParameter,
)
from ctapipe.image import ImageCleaner

__all__ = ["apply_dynamic_cleaning", "LSTImageCleaner"]


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


def lst_image_cleaning():
    pass


class LSTImageCleaner(ImageCleaner):
    """
    Clean images basically in 3 steps:
    1) Get picture threshold for `tailcuts_clean` in step 2) from interleaved
    pedestal events if `use_pedestal_cleaning` is set to `true`
    2) Apply tailcuts image cleaning algorithm - `ctapipe.image.tailcuts_clean`
    3) Apply `lst_image_cleaning` - `lstchain.image.cleaning.lst_image_cleaning`
    """

    picture_threshold_pe = FloatTelescopeParameter(
        default_value=8.0,
        help="top-level threshold in photoelectrons for `tailcuts_clean`",
    ).tag(config=True)

    boundary_threshold_pe = FloatTelescopeParameter(
        default_value=4.0,
        help="second-level threshold in photoelectrons for `tailcuts_clean`",
    ).tag(config=True)

    min_picture_neighbors = IntTelescopeParameter(
        default_value=2,
        help="Minimum number of neighbors above threshold to "
        "consider for `tailcuts_clean`",
    ).tag(config=True)

    keep_isolated_pixels = BoolTelescopeParameter(
        default_value=False,
        help="If False, pixels with less neighbors than ``min_picture_neighbors`` are"
        "removed for `tailcuts_clean`.",
    ).tag(config=True)

    delta_time = FloatTelescopeParameter(
        default_value=2,
        help="Time limit for the `time_delta_cleaning`. Set to None if no"
        "`time_delta_cleaning` should be applied",
    ).tag(config=True)

    use_dynamic_cleaning = BoolTelescopeParameter(
        default_value=True, help="Set to true if dynamic cleaning should be applied"
    ).tag(config=True)

    fraction_dynamic = FloatTelescopeParameter(
        default_value=0.03, help="`fraction` parameter for `apply_dynamic_cleaning`"
    ).tag(config=True)

    threshold_dynamic = FloatTelescopeParameter(
        default_value=267,
        help="`threshold` parameter for `apply_dynamic_cleaning`",
    ).tag(config=True)

    use_only_largest_island = BoolTelescopeParameter(
        default_value=False, help="Set to true to get only main island"
    ).tag(config=True)

    use_pedestal_cleaning = BoolTelescopeParameter(
        default_value=False,
        help="Set to true to apply pedestal cleaning. Just works if mean and std values "
        "for interleaved pedestal events are available",
    ).tag(config=True)

    sigma = FloatTelescopeParameter(
        default_value=2.5,
        help="`sigma` parameter for interleaved pedestal cleaning",
    ).tag(config=True)

    def __call__(self, event: ArrayEventContainer, tel_id: int) -> np.ndarray:
        # get pedestal thresholds
        # tailcuts_clean
        # lst_cleaning
        # return mask
        pass
