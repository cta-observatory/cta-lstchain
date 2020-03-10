"""
**TEMPORTAL** module with functions to perform a volume reduction to LST data:

NOTE : The volume reduction SHOULD BE performed at the dl0 data levels.

This is a temporal implementation of the 'tailcut and dilation' volume reduction method that computes
the mask from the dl1 (integrated image) level, and applies it to the image (dl1) as well as to the waveform (dl0) and
pulse_time arrays (dl1), filling the 'anti-mask' with zeros.

This means that the 'first' dl0 computed data is OVERWRITTEN by the vol. reduced dl0 data.

Usage:

"import volume_reducer"

In the configuration file:

(...)

"volume_reducer":{
    "algorithm": "zero_suppression_tailcut_dilation",
    "parameters": { # add here the desired tailcut parameters. See function's help for default parameters.
    }
 }

(...)
"""
from ctapipe.image.cleaning import tailcuts_clean, dilate

__all__ = ['check_volume_reduction_method',
           'apply_volume_reduction',
           'zero_suppression_tailcut_dilation'
           ]


def check_volume_reduction_method(config_file):
    """
    Checks in the configuration file if a volume reduction method has been set.

    Parameters
    ----------
    config_file: dict
        The configuration file used in the parent code.

    Returns
    -------
    flag: bool
        Flag indicating if the volume reducer method declared in the configuration file is(not) none
    algorithm: str
        Volume reduction algorithm name
    """
    if config_file['volume_reducer']['algorithm'] is None:
        flag = False
        algorithm = ''
    else:
        flag = True
        algorithm = config_file['volume_reducer']['algorithm']
        print(algorithm)

    return flag, algorithm


def apply_volume_reduction(event, tel_id, config):
    """
    Apply to an event the volume reduction method specified in the configuration file.

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
        Modifies the event container (at dl0) by applying the computed mask to the image, the waveform
        and the pulse_time objects, as:
            image[~mask] = 0, ...

    """
    camera = event.inst.subarray.tel[tel_id].camera

    image = event.dl1.tel[tel_id].image  # Volume reduction mask computed, to date, at dl1 level !

    pulse_time = event.dl1.tel[tel_id].pulse_time
    waveform = event.dl0.tel[tel_id].waveform

    algorithm = globals()[config['volume_reducer']['algorithm']]
    parameters = config['volume_reducer']['parameters']

    mask_volume_reduction = algorithm(camera, image, **parameters)

    image[~mask_volume_reduction] = 0
    pulse_time[~mask_volume_reduction] = 0
    waveform[~mask_volume_reduction, :] = 0


def zero_suppression_tailcut_dilation(geom, image, number_of_dilation=3, **kwargs):
    """
    `tailcut_clean` + 3 * `dilate` volume reducer method.

    Default parameters for the tailcut_clean:
        'picture_thresh': 8
        'boundary_thresh': 4
        'keep_isolated_pixels': True
        'min_number_picture_neighbors': 0

    Parameters
    ----------
    geom: `ctapipe.instrument.CameraGeometry`
        Camera geometry information
    image: object - ndarray
        Pixel values. A `event.inst.subarray.tel[telescope_id].camera` object.
    kwargs: dict
        A dictionary containing the parameters of the selected volume reducer algorithm.
    number_of_dilation: int
        The number of times dilation will be applied to `pixel_mask`. Set by default to 3.

    Returns
    -------
    mask_0_suppression: object - ndarray
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

    # Dilate pixel mask. 3 times by default
    for i in range(number_of_dilation):
        mask_0_suppression = dilate(geom, mask_0_suppression)

    return mask_0_suppression
