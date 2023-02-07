"""
Module containing workarounds / wrappers for ctapipe code with fixes for lstchain.

The idea is that everything here should be fixed upstream and than an import
like `from lstchain.ctapipe_compat import Foo` can just be replaced by doing
`from ctapipe.<module> import Foo` when upgrading to a ctapipe version containing
the fix.
"""
import numpy as np
import astropy.units as u


__all__ = [
    "ring_completeness"
]

# fix for a breaking change in astropy 5.2.1 that enforces correct units
# for np.histogram.
# Remove when upgrading to ctapipe 0.18.
# backport of https://github.com/cta-observatory/ctapipe/pull/2197
def ring_completeness(
    pixel_x, pixel_y, weights, radius, center_x, center_y, threshold=30, bins=30
):
    """
    Estimate how complete a ring is.
    Bin the light distribution along the the ring and apply a threshold to the
    bin content.

    Parameters
    ----------
    pixel_x: array-like
        x coordinates of the camera pixels
    pixel_y: array-like
        y coordinates of the camera pixels
    weights: array-like
        weights for the camera pixels, will usually be the pe charges
    radius: float
        radius of the ring
    center_x: float
        x coordinate of the ring center
    center_y: float
        y coordinate of the ring center
    threshold: float
        number of photons a bin must contain to be counted
    bins: int
        number of bins to use for the histogram

    Returns
    -------
    ring_completeness: float
        the ratio of bins above threshold
    """

    angle = np.arctan2(pixel_y - center_y, pixel_x - center_x)
    if hasattr(angle, "unit"):
        angle = angle.to_value(u.rad)

    hist, _ = np.histogram(angle, bins=bins, range=[-np.pi, np.pi], weights=weights)

    bins_above_threshold = hist > threshold

    return np.sum(bins_above_threshold) / bins
