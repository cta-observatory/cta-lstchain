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


class LSTImageCleaner(ImageCleaner):
    """

    """
    image_cleaner_type = create_class_enum_trait(
        base_class=ImageCleaner, default_value="TailcutsImageCleaner"
    )   

    delta_time = FloatTelescopeParameter(
        default_value=2, 
        help="Time limit for the ``delta_time_cleaning``. Set to None if no" 
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
        num_islands = {}

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

            # check the number of islands
            num_islands, island_labels = number_of_islands(camera_geometry, signal_pixels)
            if self.use_only_main_island.tel[tel_id]:
                n_pixels_on_island = np.bincount(island_labels)
                n_pixels_on_island[0] = 0  # first island is no-island and should not be considered
                max_island_label = np.argmax(n_pixels_on_island)
                signal_pixels[island_labels != max_island_label] = False

        return signal_pixels, num_islands, n_pixels
