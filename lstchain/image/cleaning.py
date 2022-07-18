import numpy as np
from ctapipe.core.traits import (
    FloatTelescopeParameter,
    IntTelescopeParameter,
    BoolTelescopeParameter,
    create_class_enum_trait
)
from ctapipe.image import (
    ImageCleaner,
    number_of_islands,
    apply_time_delta_cleaning,
    tailcuts_clean
)
from ctapipe_io_lst.constants import HIGH_GAIN
from lstchain.calib.camera.pixel_threshold_estimation import get_ped_thresh

__all__ = [
    'apply_dynamic_cleaning',
    'get_only_main_island',
    'lst_image_cleaning',
    'LSTImageCleaner'
]

ORIGINAL_CALIBRATION_ID = 0
INTERLEAVED_CALIBRATION_ID = 1

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


def get_only_main_island(geom, signal_pixels):
    """
    Reduce the selected image mask to return only the main island

    Parameters
    ----------
    geom: `ctapipe.instrument.CameraGeometry`
        Camera geometry information
    signal_pixels: `np.ndarray`
        Boolean mask with the selected pixels

    Returns
    -------
    signal_pixels: `np.ndarray`
        Boolean mask with the selected pixels after selecting only the main island
    num_islands: `int`
        Number of islands before it was reduced to one

    """

    num_islands, island_labels = number_of_islands(geom, signal_pixels)
    n_pixels_on_island = np.bincount(island_labels)
    n_pixels_on_island[0] = 0  # first island is no-island and should not be considered
    max_island_label = np.argmax(n_pixels_on_island)
    signal_pixels[island_labels != max_island_label] = False

    return signal_pixels, num_islands


class LSTImageCleaner(ImageCleaner):
    """
    Clean images basically in 5 steps:
    1) Get picture threshold for `tailcuts_clean` in step 2) from interleaved
       pedestal events if `use_pedestal_cleaning` is set to `true`
    2) Apply tailcuts image cleaning algorithm - `ctapipe.image.tailcuts_clean`
    3) Apply time_delta_cleaning - `ctapipe.image.apply_time_delta_cleaning` if
       `delta_time` is not None
    4) Apply dynamic_cleaning - `lstchain.image.cleaning.apply_dynamic_cleaning` if
       `use_dynamic_cleaning` is set to `true`
    5) Get only main island - `lstchain.image.cleaning.get_only_main_island` if
       `use_only_main_island` is set to `true`

    Returns
    -------
    signal_pixels: `np.ndarray`
        Boolean mask with the selected pixels after all the cleaning steps
    num_islands: `int`
        Number of islands before it was reduced to one in step 5)
    n_pixels: `int`
        Number of pixels which survived the `tailcuts_clean` in step 2)
    """

    picture_threshold_pe = FloatTelescopeParameter(
        default_value=8.0,
        help="top-level threshold in photoelectrons for `tailcuts_clean`"
    ).tag(config=True)

    boundary_threshold_pe = FloatTelescopeParameter(
        default_value=4.0, help="second-level threshold in photoelectrons for `tailcuts_clean`"
    ).tag(config=True)

    min_picture_neighbors = IntTelescopeParameter(
        default_value=2, help="Minimum number of neighbors above threshold to "
        "consider for `tailcuts_clean`"
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
        default_value=True,
        help="Set to true if dynamic cleaning should be applied"
    ).tag(config=True)

    fraction_dynamic = FloatTelescopeParameter(
        default_value=0.03,
        help="`fraction` parameter for `apply_dynamic_cleaning`"
    ).tag(config=True)

    threshold_dynamic = FloatTelescopeParameter(
        default_value=267,
        help="`threshold` parameter for `apply_dynamic_cleaning`",
    ).tag(config=True)  

    use_only_main_island = BoolTelescopeParameter(
        default_value=False,
        help="Set to true if only get main island"
    ).tag(config=True)
    
    use_pedestal_cleaning = BoolTelescopeParameter(
        default_value=False,
        help="Set to true to apply pedestal cleaning. Just works if mean and std values "
        "for interleaved pedestal events are available"
    ).tag(config=True)
    
    sigma = FloatTelescopeParameter(
        default_value=2.5,
        help="`sigma` parameter for interleaved pedestal cleaning",
    ).tag(config=True)  

    def __call__(self, tel_id: int, event): 

        geom = self.subarray.tel[tel_id].camera.geometry
        image = event.dl1.tel[tel_id].image
        arrival_times = event.dl1.tel[tel_id].peak_time
        pic_thresh = self.picture_threshold_pe.tel[tel_id]

        if self.use_pedestal_cleaning:
            ped_thresh = get_ped_thresh(tel_id=tel_id, event=event, sigma=self.sigma)
            pic_thresh = np.clip(ped_thresh, pic_thresh, None)

        signal_pixels = tailcuts_clean(
            geom=geom,
            image=image,
            picture_thresh=pic_thresh,
            boundary_thresh=self.boundary_threshold_pe.tel[tel_id],
            min_number_picture_neighbors=self.min_picture_neighbors.tel[tel_id],
            keep_isolated_pixels=self.keep_isolated_pixels.tel[tel_id],
        )

        n_pixels = np.count_nonzero(signal_pixels)
        num_islands = 0

        if n_pixels > 0:
            if self.delta_time.tel[tel_id] is not None:
                signal_pixels = apply_time_delta_cleaning(
                    geom,
                    signal_pixels,
                    arrival_times,
                    min_number_neighbors=1,
                    time_limit=self.delta_time.tel[tel_id]
                )
            if self.use_dynamic_cleaning:
                signal_pixels = apply_dynamic_cleaning(
                    image,
                    signal_pixels,
                    self.threshold_dynamic.tel[tel_id],
                    self.fraction_dynamic.tel[tel_id]
                )
            if self.use_only_main_island:
                signal_pixels, num_islands = get_only_main_island(geom, signal_pixels)
            else:
                num_islands, _ = number_of_islands(geom, signal_pixels)

        event.dl1.tel[tel_id].image_mask = signal_pixels

        return signal_pixels, n_pixels, num_islands
