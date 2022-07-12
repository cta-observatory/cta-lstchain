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
)

__all__ = [
    'apply_dynamic_cleaning',
    'get_only_main_island',
    'lst_image_cleaning',
    'LSTImageCleaner'
]


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


def get_only_main_island(island_labels, signal_pixels):
    """
    Reduce the selected image mask to return only the main island

    Parameters
    ----------
    island_labels: `np.ndarray`
        Returned island labels from `ctapipe.image.number_of_islands`
    signal_pixels: `np.ndarray`
        Boolean mask with the selected pixels

    Returns
    -------
    signal_pixels: `np.ndarray`
        Boolean mask with the selected pixels after selecting only the main island

    """

    n_pixels_on_island = np.bincount(island_labels)
    n_pixels_on_island[0] = 0  # first island is no-island and should not be considered
    max_island_label = np.argmax(n_pixels_on_island)
    signal_pixels[island_labels != max_island_label] = False

    return signal_pixels


def lst_image_cleaning(
    geom,
    image,
    arrival_times,
    signal_pixels,
    delta_time,
    use_dynamic_cleaning,
    fraction_dynamic,
    threshold_dynamic,
    use_only_main_island
):
    """
    Clean an already selected image of signal pixels (e.g. with TailcutsImageCleaner)
    in 3 further steps:
    1) Apply time_delta_cleaning - `ctapipe.image.apply_time_delta_cleaning`
    2) Apply dynamic_cleaning - `lstchain.image.cleaning.apply_dynamic_cleaning`
    3) Get only main island - `lstchain.image.cleaning.get_only_main_island`

    Parameters
    ----------
    geom: `ctapipe.instrument.CameraGeometry`
        Camera geometry information
    image: `np.ndarray`
        pixel values
    arrival_times: `np.ndarray`
        pixel timing information
    signal_pixels: `np.ndarray`
        boolean mask of cleaned pixels after e.g. `TailcutsImageCleaner`
    delta_time: `Float`
        Time limit for the `apply_time_delta_cleaning` in step 2).
        Set to `None` if no time_delta_cleaning should be applied
    use_dynamic_cleaning: `Bool`
        Set to True, if dynamic cleaning (Step 3) should be applied
    fraction_dynamic: `Float`
        Fraction parameter for `apply_dynamic_cleaning`
    threshold_dynamic: `Float`
        Threshold parameter for `apply_dynamic_cleaning`
    use_only_main_island: `Bool`
        Set to True if considering only the main island in step 4)

    Returns
    -------
    signal_pixels: `np.ndarray`
        Boolean mask with the selected pixels after all the cleaning steps
    num_islands: `int`
        Number of islands before it was reduced to one in step 3)
    n_pixels: `int`
        Number of pixels which survived the cleaning in the previous step
        (e.g. after the TailcutsImageCleaner)
    """

    n_pixels = np.count_nonzero(signal_pixels)
    num_islands = 0

    if n_pixels > 0:
        if delta_time is not None:
            signal_pixels = apply_time_delta_cleaning(
                geom,
                signal_pixels,
                arrival_times,
                min_number_neighbors=1,
                time_limit=delta_time
            )
        if use_dynamic_cleaning:
            signal_pixels = apply_dynamic_cleaning(
                image,
                signal_pixels,
                threshold_dynamic,
                fraction_dynamic
            )
        num_islands, island_labels = number_of_islands(geom, signal_pixels)
        if use_only_main_island:
            signal_pixels = get_only_main_island(island_labels, signal_pixels)

    return signal_pixels, num_islands, n_pixels

class LSTImageCleaner(ImageCleaner):
    """
    Clean images in two steps:
    1) Apply first image cleaning algorithm
       Default: TailcutsImageCleaner - `ctapipe.image.TailcutsImageCleaner`
    2) Apply `lst_image_cleaning`-algorithm on signal_pixels returned from step 1)

    """
    image_cleaner_type = create_class_enum_trait(
        base_class=ImageCleaner, default_value="TailcutsImageCleaner"
    )   

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

    def __call__(self, tel_id: int, image: np.ndarray, arrival_times=None): 

        cleaner = ImageCleaner.from_name(
            self.image_cleaner_type, subarray=self.subarray, parent=self
        )
        signal_pixels = cleaner(tel_id=tel_id, image=image, arrival_times=arrival_times)

        return lst_image_cleaning(
            geom=self.subarray.tel[tel_id].camera.geometry,
            image=image,
            arrival_times=arrival_times,
            signal_pixels=signal_pixels,
            delta_time=self.delta_time.tel[tel_id],
            use_dynamic_cleaning=self.use_dynamic_cleaning.tel[tel_id],
            fraction_dynamic=self.fraction_dynamic.tel[tel_id],
            threshold_dynamic=self.threshold_dynamic.tel[tel_id],
            use_only_main_island=self.use_only_main_island.tel[tel_id]
        )
