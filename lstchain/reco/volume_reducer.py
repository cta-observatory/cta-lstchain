#!/usr/bin/env python
"""
**TEMPORARY** module with functions to perform a volume reduction to LST data:

This is a temporary implementation of the 'tailcut and dilation' volume reduction method.
The volume reduction produces reduced waveforms and write them in the DL0 container.

To date, the ** DL0 container is overwritten ** if the volume reduction is applied.


Usage:
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

__all__ = ['get_volume_reduction_method',
           'check_and_apply_volume_reduction',
           'zero_suppression_tailcut_dilation'
           ]


def get_volume_reduction_method(config_file):
    """
    Checks in the configuration file if a volume reduction method has been set.

    Parameters
    ----------
    config_file: dict
        The configuration file used in the parent code.

    Returns
    -------
    algorithm: str
        Volume reduction algorithm name
    """
    if config_file['volume_reducer']['algorithm'] is None:
        algorithm = None
    else:
        algorithm = config_file['volume_reducer']['algorithm']

    return algorithm


def check_and_apply_volume_reduction(event, config):
    """
    Checks the volume reduction algorithm defined in the config file, and if not None, it applies
     to a **calibrated** event the volume reduction method.

    Parameters
    ----------
    event: 'ctapipe.io.containers.DataContainer'
    config: dict
        Read the parameters of the volume reduction method specified in the config file.

    Returns
    -------
    none
        Modifies the event container by applying the computed mask to the image, the waveform
        and the pulse_time objects, as:
            image[~mask] = 0, ...

    """
    volume_reduction_algorithm = get_volume_reduction_method(config)

    if volume_reduction_algorithm is None:
        pass

    else:
        print("Volume reduction algorithm being used: {}".format(volume_reduction_algorithm))

        volume_reduction_algorithm = globals()[volume_reduction_algorithm]
        parameters = config['volume_reducer']['parameters']

        for tel_id in event.r0.tels_with_data:

            camera = event.inst.subarray.tel[tel_id].camera

            image = event.dl1.tel[tel_id].image  # Volume reduction mask computed, to date, at dl1 level !
            pulse_time = event.dl1.tel[tel_id].pulse_time
            waveform = event.dl0.tel[tel_id].waveform

            pixels_to_keep = volume_reduction_algorithm(camera, image, **parameters)

            image[~pixels_to_keep] = 0
            pulse_time[~pixels_to_keep] = 0
            waveform[~pixels_to_keep, :] = 0


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
