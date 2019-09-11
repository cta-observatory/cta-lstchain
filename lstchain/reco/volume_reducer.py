"""
Module with functions to perform a volume reduction to the dl1 files:
NOTE : The volume reduction is indeed applied in the process
between R1 to dl1.

Usage:

"import volume_reducer"
"""
from ctapipe.image.cleaning import tailcuts_clean, dilate

__all__ = ['apply_volume_reduction',
           'zero_suppression_tailcut_dilation',
           'dilate_pixel_mask'
           ]


def apply_volume_reduction(event, tel_id, config):
    """
    Function that looks into the loaded configuration and applies to the data the specified volume
    reduction method.

    Parameters
    ----------
    event: 'ctapipe.io.containers.DataContainer'
    tel_id: int
        A telescope id wih data: an `event.r0.tels_with_data` object.
    config: dict
        Loaded configuration: configuration used to select the volume reducer method and its
        corresponding parameters.

    Returns
    -------
    none
        Modifies the event container by applying the computed mask to the image, the waveform
        and the pulse_time objects, as:
        image[~mask] = 0, ...

    """
    if config['volume_reducer']['algorithm'] is None:
        pass
    else:
        camera = event.inst.subarray.tel[tel_id].camera
        image = event.dl1.tel[tel_id].image
        pulse_time = event.dl1.tel[tel_id].pulse_time
        waveform = event.r1.tel[tel_id].waveform

        algorithm = globals()[config['volume_reducer']['algorithm']]
        parameters = config['volume_reducer']['parameters']

        print("Volume reduction algorithm:", algorithm)

        mask_volume_reduction = algorithm(camera, image, **parameters)
        image[~mask_volume_reduction] = 0
        pulse_time[~mask_volume_reduction] = 0
        waveform[:, ~mask_volume_reduction, :] = 0


def zero_suppression_tailcut_dilation(geom, image, **kwargs):
    """
    `tailcut_clean` + 3 * `dilate` volume reducer method.

    Default parameters for the tailcut_clean:
        'picture_thresh': 8
        'boundary_thresh' 4
        'keep_isolated_pixels', True
        'min_number_picture_neighbors', 0

    Parameters
    ----------
    geom: `ctapipe.instrument.CameraGeometry`
        Camera geometry information
    image: array
        Pixel values. A `event.inst.subarray.tel[telescope_id].camera` object.
    kwargs: dict
        A dictionary containing the parameters of the selected volume reducer algorithm.

    Returns
    -------
    array
        A boolean mask (array) that contains the zero suppressed pixels.

    """
    kwargs.setdefault('picture_thresh', 8)
    kwargs.setdefault('boundary_thresh', 4)
    kwargs.setdefault('keep_isolated_pixels', True)
    kwargs.setdefault('min_number_picture_neighbors', 0)

    mask_0_suppression = tailcuts_clean(geom,
                                        image,
                                        picture_thresh=kwargs['picture_thresh'],
                                        boundary_thresh=kwargs['boundary_thresh'],
                                        keep_isolated_pixels=kwargs['keep_isolated_pixels'],
                                        min_number_picture_neighbors=kwargs['min_number_picture_neighbors']
                                        )
    mask_0_suppression = dilate_pixel_mask(geom, mask_0_suppression)

    return mask_0_suppression


def dilate_pixel_mask(geom, pixel_mask, number_of_dilation=3):
    """
    Dilates `number_of_dilation` times the boolean mask created by `zero_suppression_tailcut_dilation`.

    Parameters
    ----------
    geom: `ctapipe.instrument.CameraGeometry`
        Camera geometry information
    pixel_mask: array
        A boolean mask (array) containing the zero suppressed pixels.
    number_of_dilation: int
        The number of times dilation will be applied to `pixel_mask`. Set by default to 3.

    Returns
    -------
    array
        A boolean mask (array) containing the zero suppressed pixels.

    """
    for i in range(number_of_dilation):
        pixel_mask = dilate(geom, pixel_mask)

    return pixel_mask
