import numpy as np
from ctapipe.image import ImageCleaner
from ctapipe.core.traits import Bool, Float, Int

__all__ = [
    'apply_dynamic_cleaning',
    'LSTImageCleaner'
]

class LSTImageCleaner(ImageCleaner):
    """

    """
    def __call__():
        
    signal_pixels = cleaning_method(camera_geometry, image, **cleaning_parameters)
    n_pixels = np.count_nonzero(signal_pixels)

    if n_pixels > 0:

        if delta_time is not None:
            signal_pixels = apply_time_delta_cleaning(
                camera_geometry,
                signal_pixels,
                peak_time,
                min_number_neighbors=1,
                time_limit=delta_time
            )

        if use_dynamic_cleaning:
            threshold_dynamic = config['dynamic_cleaning']['threshold']
            fraction_dynamic = config['dynamic_cleaning']['fraction_cleaning_intensity']
            signal_pixels = apply_dynamic_cleaning(image,
                                                   signal_pixels,
                                                   threshold_dynamic,
                                                   fraction_dynamic)

        # check the number of islands
        num_islands, island_labels = number_of_islands(camera_geometry, signal_pixels)
        dl1_container.n_islands = num_islands

        if use_main_island:
            n_pixels_on_island = np.bincount(island_labels)
            n_pixels_on_island[0] = 0  # first island is no-island and should not be considered
            max_island_label = np.argmax(n_pixels_on_island)
            signal_pixels[island_labels != max_island_label] = False



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
