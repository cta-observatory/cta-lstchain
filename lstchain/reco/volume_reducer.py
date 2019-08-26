"""
Module with functions to perform a volume reduction to the dl1 files:
NOTE : The volume reduction is indeed applied in the process
between R1 to dl1.

Usage:

"import volume_reducer"
"""
import numpy as np
from copy import deepcopy
from ctapipe.image.cleaning import tailcuts_clean, dilate

__all__ = ['apply_volume_reduction',
           'zero_suppression_tailcut_dilation',
           'dilate_pixel_mask',
           'none_volume_reduction'
           ]


def apply_volume_reduction(event, tel_id, config):
    """
    Function that looks into the loaded configuration and applies to the data (modifying
    the event container) the volume reduction method to be applied.
    A `none_volume_reduction` function is applied by default.

    :param event: [source event. ctapipe datacontainer]
            An event of a loaded source. A 'ctapipe.io.containers.DataContainer' object.
    :param tel_id: [int]
            A telescope id wih data: an `event.r0.tels_with_data` object.
    :param config: [dict]
            Loaded configuration: from where the algorithm (volume reduction method)
            and their corresponding parameters are loaded.

    :return: none.
            Modifies the event containers by applying the computed mask to the image,
            the waveform and the pulse_time objects, as:
            image[~mask] = 0, ...
    """
    camera = event.inst.subarray.tel[tel_id].camera
    image = event.dl1.tel[tel_id].image
    pulse_time = event.dl1.tel[tel_id].pulse_time
    waveform = event.r1.tel[tel_id].waveform

    if config['volume_reducer']['algorithm'] is None:
        algorithm = none_volume_reduction
        parameters = config['volume_reducer']['parameters']
    else:
        algorithm = globals()[config['volume_reducer']['algorithm']]
        parameters = config['volume_reducer']['parameters']

    print("Volume reduction algorithm:", algorithm)

    mask_volume_reduction = algorithm(image, camera, **parameters)
    image[~mask_volume_reduction] = 0
    waveform[:, ~mask_volume_reduction, :] = 0
    pulse_time[~mask_volume_reduction] = 0


def zero_suppression_tailcut_dilation(image, camera_geometry, **kwargs):
    """
    `tailcut_clean` + 3 `dilate` volume reducer method.

    Default parameters for the tailcut_clean:
        'picture_thresh': 8
        'boundary_thresh' 4
        'keep_isolated_pixels', True
        'min_number_picture_neighbors', 0

    :param image: [array]
            Pixel values. A `event.inst.subarray.tel[telescope_id].camera` object.
    :param camera_geometry: [Camera geometry information]
            `ctapipe.instrument.CameraGeometry` object.
    :param kwargs: [dict]
            A dictionary containing the parameters of the selected volume reducer algorithm.

    :return:[array]
            A boolean mask (array) that contains the zero suppressed pixels.
    """
    kwargs.setdefault('picture_thresh', 8)
    kwargs.setdefault('boundary_thresh', 4)
    kwargs.setdefault('keep_isolated_pixels', True)
    kwargs.setdefault('min_number_picture_neighbors', 0)

    picture_zs = kwargs['picture_thresh']
    boundary_zs = kwargs['boundary_thresh']
    keep_isolated_pix_zs = kwargs['keep_isolated_pixels']
    min_num_pict_neigh_zs = kwargs['min_number_picture_neighbors']

    mask_0_suppression = tailcuts_clean(camera_geometry,
                                        image,
                                        picture_thresh=picture_zs,
                                        boundary_thresh=boundary_zs,
                                        keep_isolated_pixels=keep_isolated_pix_zs,
                                        min_number_picture_neighbors=min_num_pict_neigh_zs
                                        )
    mask_pixel_0_suppressed = dilate_pixel_mask(mask_0_suppression, camera_geometry)

    return mask_pixel_0_suppressed


def dilate_pixel_mask(pixel_mask, camera_geometry):
    """
    Dilates three times the boolean mask created by `zero_suppression_tailcut_dilation`.

    :param pixel_mask: [array]
            A boolean mask (array) containing the zero suppressed pixels.
    :param camera_geometry: [Camera geometry information]
            `ctapipe.instrument.CameraGeometry` object.

    :return: [array]
        A boolean mask (array) containing the zero suppressed pixels.
    """
    for i in range(3):
        if i == 0:
            copy_mask = deepcopy(pixel_mask)
        else:
            copy_mask = deepcopy(dilated_mask)

        dilated_mask = dilate(camera_geometry, copy_mask)

    return dilated_mask


def none_volume_reduction(image, camera_geometry, **kwargs):
    """
    Methods applied by default.
    NO volume reduction of the data (image, waveform and pulse_time).

    Creates a boolean mask of the same size as the input `image`. All the values
    are set to be True, thus no pixel is rejected when the mask is applied.

    :param image: [array]
            Pixel values. A `event.inst.subarray.tel[telescope_id].camera` object.
    :param camera_geometry: [Camera geometry information]
            `ctapipe.instrument.CameraGeometry` object.
    :param kwargs: [dict]
            A dictionary containing the parameters of the selected volume reducer algorithm.

    :return: [array]
            A boolean mask (array) that contains only True values.
    """
    _, _ = camera_geometry, kwargs
    mask_no_vol_reduction = np.full(image.shape, True, dtype=bool)

    return mask_no_vol_reduction
