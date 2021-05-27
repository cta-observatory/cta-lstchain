__all__ = [
    'add_noise_in_pixels',
    'random_psf_smearer',
]

import numpy as np
from numba import njit

# number of neighbors of completely surrounded pixels of hexagonal cameras:
N_PIXEL_NEIGHBORS = 6
SMEAR_PROBALITITES = np.full(N_PIXEL_NEIGHBORS, 1 / N_PIXEL_NEIGHBORS)


def add_noise_in_pixels(rng, image, extra_noise_in_dim_pixels,
                        extra_bias_in_dim_pixels, transition_charge,
                        extra_noise_in_bright_pixels):
    """

    Parameters
    ----------
    rng: numpy.random.default_rng  random number generator

    image: charges (p.e.) in the camera

    To be tuned by comparing the starting MC and data:

    extra_noise_in_dim_pixels: mean additional number of p.e. to be added (
    Poisson noise) to pixels with charge below transition_charge

    extra_bias_in_dim_pixels: mean bias (w.r.t. original charge) of the new
    charge in pixels. Should be 0 for non-peak-search pulse integrators

    transition_charge: border between "dim" and "bright" pixels

    extra_noise_in_bright_pixels: mean additional number of p.e. to be added (
    Poisson noise) to pixels with charge above transition_charge. This is
    unbiased, i.e. Poisson noise is introduced, and its average subtracted,
    so that the mean charge in bright pixels remains unaltered. This is
    because we assume that above transition_charge the integration window
    is determined by the Cherenkov light, and would not be modified by the
    additional NSB noise (presumably small compared to the C-light)

    Returns
    -------
    Modified (noisier) image

    """

    bright_pixels = image > transition_charge
    noise = np.where(bright_pixels, extra_noise_in_bright_pixels,
                     extra_noise_in_dim_pixels)
    bias = np.where(bright_pixels, -extra_noise_in_bright_pixels,
                    extra_bias_in_dim_pixels - extra_noise_in_dim_pixels)

    image = image + rng.poisson(noise) + bias

    return image

@njit()
def set_numba_seed(seed):
    np.random.seed(seed)

@njit(cache=True)
def random_psf_smearer(image, fraction, indices, indptr):
    """
    Parameters
    ----------
    image: charges (p.e.) in the camera

    indices: camera_geometry.neighbor_matrix_sparse.indices

    indptr: camera_geometry.neighbor_matrix_sparse.indptr

    fraction: fraction of the light in a pixel that will be
    distributed among its immediate surroundings, i.e. immediate
    neighboring pixels, according to Poisson statistics. Some light is
    lost for pixels which are at the camera edge and hence don't have all
    possible neighbors

    Returns
    -------
    Modified (smeared) image

    """

    new_image = image.copy()

    for pixel in range(len(image)):

        if image[pixel] <= 0:
            continue

        to_smear = np.random.poisson(image[pixel] * fraction)

        if to_smear == 0:
            continue

        # remove light from current pixel
        new_image[pixel] -= to_smear

        # add light to neighbor pixels
        neighbors = indices[indptr[pixel] : indptr[pixel + 1]]
        n_neighbors = len(neighbors)

        # all neighbors are equally likely to receive the charge
        # we always distribute the charge into 6 neighbors, so that charge
        # on the edges of the camera is lost
        neighbor_charges = np.random.multinomial(to_smear, SMEAR_PROBALITITES)

        for n in range(n_neighbors):
            neighbor = neighbors[n]
            new_image[neighbor] += neighbor_charges[n]

    return new_image
