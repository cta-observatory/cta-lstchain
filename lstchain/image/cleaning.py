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
        Mask with the selected pixels

    Returns
    -------
    signal_pixels: `np.ndarray`
        Mask with the selected pixels after selecting only the main island

    """

    n_pixels_on_island = np.bincount(island_labels)
    n_pixels_on_island[0] = 0  # first island is no-island and should not be considered
    max_island_label = np.argmax(n_pixels_on_island)
    signal_pixels[island_labels != max_island_label] = False

    return signal_pixels


class LSTImageCleaner(ImageCleaner):
    """
    Clean images in four variable steps:
    1) Apply first image cleaning 
       Default: TailcutsImageCleaner - `ctapipe.image.TailcutsImageCleaner`
    2) Apply time_delta_cleaning - `ctapipe.image.apply_time_delta_cleaning`
    3) Apply dynamic_cleaning - `lstchain.image.cleaning.apply_dynamic_cleaning`
    4) Get only main island - `lstchain.image.cleaning.get_only_main_island`

    Attributes
    ----------
    image_cleaner_type: `String`
        Name of the image cleaner to be used in step 1).
        Default: TailcutsImageCleaner
    delta_time: `FloatTelescopeParameter`
        Time limit for the `apply_time_delta_cleaning` in step 2).
        Set to `None` if no time_delta_cleaning should be applied
    use_dynamic_cleaning: `BoolTelescopeParameter`
        Set to True, if dynamic cleaning (Step 3) should be applied
    fraction_dynamic: `FloatTelescopeParameter`
        Fraction parameter for `apply_dynamic_cleaning`
    threshold_dynamic: `FloatTelescopeParameter`
        Threshold parameter for `apply_dynamic_cleaning`
    use_only_main_island: `BoolTelescopeParameter`
        Set to True if considering only the main island in step 4)

    Returns
    -------
    signal_pixels: `np.ndarray`
        Mask with the selected pixels after all the cleaning steps
    num_islands: `int`
        Number of islands before it was reduced to one in step 4)
    n_pixels: `int`
        Number of pixels surviving the the cleaning in step 1)
    """
    image_cleaner_type = create_class_enum_trait(
        base_class=ImageCleaner, default_value="TailcutsImageCleaner"
    )   

    delta_time = FloatTelescopeParameter(
        default_value=2, 
        help="Time limit for the ``time_delta_cleaning``. Set to None if no" 
        "``time_delta_cleaning`` should be applied",
    ).tag(config=True)

    use_dynamic_cleaning = BoolTelescopeParameter(
        default_value=True,
        help="Set to true if dynamic cleaning should be applied"
    ).tag(config=True)

    fraction_dynamic = FloatTelescopeParameter(
        default_value=0.03,
        help="``fraction`` parameter for ``apply_dynamic_cleaning``"
    ).tag(config=True)

    threshold_dynamic = FloatTelescopeParameter(
        default_value=267,
        help="``threshold`` parameter for ``apply_dynamic_cleaning``",
    ).tag(config=True)  

    use_only_main_island = BoolTelescopeParameter(
        default_value=False,
        help="Set to true if only use main island"
    ).tag(config=True)

    def __call__(self, tel_id: int, image: np.ndarray, arrival_times=None): 

        cleaner = ImageCleaner.from_name(
            self.image_cleaner_type, subarray=self.subarray, parent=self
        )

        camera_geometry = self.subarray.tel[tel_id].camera.geometry
        signal_pixels = cleaner(tel_id=tel_id, image=image, arrival_times=arrival_times)
        n_pixels = np.count_nonzero(signal_pixels)
        num_islands = 0

        if n_pixels > 0:

            if self.delta_time.tel[tel_id] is not None:
                signal_pixels = apply_time_delta_cleaning(
                    camera_geometry,
                    signal_pixels,
                    arrival_times,
                    min_number_neighbors=1,
                    time_limit=self.delta_time.tel[tel_id]
                )
            if self.use_dynamic_cleaning.tel[tel_id]:
                signal_pixels = apply_dynamic_cleaning(image,
                                                       signal_pixels,
                                                       self.threshold_dynamic.tel[tel_id],
                                                       self.fraction_dynamic.tel[tel_id])

            num_islands, island_labels = number_of_islands(camera_geometry, signal_pixels)
            if self.use_only_main_island.tel[tel_id]:
                signal_pixels = get_only_main_island(island_labels, signal_pixels)

        return signal_pixels, num_islands, n_pixels
