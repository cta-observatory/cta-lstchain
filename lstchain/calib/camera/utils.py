import numpy as np


def check_outlier_mask(outliers, log, kind, n_pixels_threshold=50):
    """Check a mask of shape (n_events, n_pixels) for too many True values
    and create a warning message.
    """
    counts = np.count_nonzero(outliers, axis=-1)
    log.info("Total number of outliers in %s sample: %d", kind, np.count_nonzero(outliers))
    bad_index, bad_gain = np.nonzero(counts > n_pixels_threshold)
    if len(bad_index) >= 3:
        log.warning("Found large number of outliers in %s sample:", kind)
        log.warning(
            "Outliers: %d events with more than %d pixels",
            len(bad_index), n_pixels_threshold
        )
        log.warning("Bad event index: %s", bad_index)
        log.warning("Bad gain channel: %s", bad_gain)
        log.warning("Number of bad pixels in those entries: %s", counts[bad_index, bad_gain])
