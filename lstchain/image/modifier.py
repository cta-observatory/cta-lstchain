__all__ = [
    'add_noise_in_pixels',
    'smear_light_in_pixels',
]

import numpy as np

# number of neighbors of completely surrounded pixels of hexagonal cameras:
N_PIXEL_NEIGHBORS = 6

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


def smear_light_in_pixels(rng, image, camera_geom, smeared_light_fraction):
    """

    Parameters
    ----------
    rng: numpy.random.default_rng  random number generator

    image: charges (p.e.) in the camera

    camera_geom: camera geometry

    smeared_light_fraction: fraction of the light in a pixel that will be
    distributed among its immediate surroundings, i.e. immediate
    neighboring pixels, according to Poisson statistics. Some light is
    lost for pixels which are at the camera edge and hence don't have all
    possible neighbors

    Returns
    -------
    Modified (smeared) image

    """

    # How many p.e. to smear?  Poisson of the smeared light fraction (clipped to a minimum of 0):
    pe_to_smear = rng.poisson(np.clip(image * smeared_light_fraction, 0, np.inf))

    # How to distribute the smeared charge among neighboring pixels (multinomial):
    smeared_charges = np.zeros(shape=(len(image), N_PIXEL_NEIGHBORS))
    for q in np.unique(pe_to_smear):
        # generate the p.e. smearing patterns for all pixels with q  p.e.'s to be smeared:
        num_pixels = np.sum(pe_to_smear == q)
        # in this way we speed things up a lot through vectorization (vs. doing it pixel-wise):
        smeared_charges[pe_to_smear == q, :] = rng.multinomial(q, N_PIXEL_NEIGHBORS * [1 / N_PIXEL_NEIGHBORS],
                                                               size=num_pixels)

    q_smeared = np.zeros([len(image), len(image)])

    # The bulk of the execution time is spent here:
    for pixid, charges in enumerate(smeared_charges):
        q_smeared[pixid][camera_geom.neighbor_matrix[pixid]] = charges[:camera_geom.neighbor_matrix[pixid].sum()]
    # Any idea on how to speed this up? (note the number of neighbors is not the same for all pixels!)

    # Light remaining in pixel:
    q_remaining = image - pe_to_smear

    image = q_remaining + np.sum(q_smeared, axis=0)

    return image
