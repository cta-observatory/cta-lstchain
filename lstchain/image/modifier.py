__all__ = [
    'add_noise_in_pixels',
    'smear_light_in_pixels',
]

import numpy as np

def add_noise_in_pixels(image, extra_noise_in_dim_pixels,
                        extra_bias_in_dim_pixels, transition_charge,
                        extra_noise_in_bright_pixels):
    qcopy = image.copy()
    image[qcopy < transition_charge] += (np.random.poisson(
            extra_noise_in_dim_pixels, (qcopy < transition_charge).sum()) -
                                         extra_noise_in_dim_pixels +
                                         extra_bias_in_dim_pixels)
    image[qcopy > transition_charge] += (np.random.poisson(
            extra_noise_in_bright_pixels, (qcopy > transition_charge).sum())
                                         - extra_noise_in_bright_pixels)
    return image


def smear_light_in_pixels(image, camera_geom, smeared_light_fraction):
    # Move a fraction of the light in each pixel (fraction) into its neighbors,
    # to simulate a worse PSF:

    # number of neighbors of completely surrounded pixels:
    max_neighbors = np.max([len(pixnbs) for pixnbs in camera_geom.neighbors])
    q_smeared = (image * camera_geom.neighbor_matrix *
                 smeared_light_fraction / max_neighbors)

    q_remaining = image * (1 - smeared_light_fraction)
    image = q_remaining + np.sum(q_smeared, axis=1)
