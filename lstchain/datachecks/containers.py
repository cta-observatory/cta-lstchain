"""
Containers for data check
"""

__all__ = [
    'DL1DataCheckContainer',
    'count_trig_types',
    'DL1DataCheckHistogramBins',
]

import logging
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
import warnings
from ctapipe.core import Container, Field


class DL1DataCheckContainer(Container):
    """
    Container to store outcome of the DL1 data check
    """

    geomlogger = logging.getLogger('ctapipe.instrument.camera')
    geomlogger.setLevel(logging.ERROR)

    # scalar quantities:
    subrun_index = Field(-1, 'Subrun index')
    elapsed_time = Field(-1, 'Subrun time duration (from Dragon)')
    num_events = Field(-1, 'Total number of events')
    num_cleaned_events = Field(-1, 'Number of events surviving cleaning')
    trigger_type = Field(None, 'Number of events per trigger type')
    ucts_trigger_type = Field(None, 'Number of events per ucts trigger type')
    mean_alt_tel = Field(None, 'Mean telescope altitude')
    mean_az_tel = Field(None, 'Mean telescope azimuth')

    # sampled quantities, stored every few events:
    sampled_event_ids = Field(None, 'sampled event ids')
    ucts_time = Field(None, 'ucts time', unit=u.s)
    tib_time = Field(None, 'tib_time', unit=u.s)
    dragon_time = Field(None, 'dragon_time', unit=u.s)

    # histograms; they store arrays of counts. Binning is defined in class
    # DL1DataCheckHistogramBins (see below)
    hist_delta_t = Field(None, 'Histogram of time difference between '
                               'consecutive events')
    hist_npixels = Field(None, 'Histogram of number of pixels in image')
    hist_nislands = Field(None, 'Histogram of number of islands in image')
    hist_intensity = Field(None, 'Histogram of image intensity')
    hist_dist0 = Field(None, 'Histogram of cog-camera center distance')
    hist_dist0_intensity_gt_200 = \
        Field(None, 'Histogram of cog-camera center distance')
    hist_width = Field(None, 'Histogram image width vs. intensity')
    hist_length = Field(None, 'Histogram image length vs. intensity')
    hist_skewness = Field(None, 'Histogram of image skewness')
    # the histogram hist_pixelchargespectrum shows the pixel charge
    # distribution, filled from all pixels:
    hist_pixelchargespectrum = Field(None, 'Histogram of pixel charges')

    hist_psi = Field(None, 'Histogram of image axis orientation')
    hist_intercept = Field(None, 'Histogram of fitted pulse time for charge '
                                 'c.o.g.')
    hist_tgrad_vs_length = Field(None, 'Histogram of time gradient vs. length')
    hist_tgrad_vs_length_intensity_gt_200 = \
        Field(None, 'Histogram of time gradient vs. length, intensity>200pe')

    # pixel-wise quantities, one entry per pixel. Used also for 2d
    # histogramming of cog position.
    cog_within_pixel = Field(None, 'Number of image cogs within pixel')
    cog_within_pixel_intensity_gt_200 = \
        Field(None, 'Number of image within pixel, intensity>200pe')
    charge_mean = Field(-1, 'Mean of pixel charge')
    charge_stddev = Field(-1, 'Standard deviation of pixel charge')
    time_mean = Field(-1, 'Mean of pulse time')
    time_stddev = Field(-1, 'Standard deviaton of pulse time')
    time_mean_above_030_pe = Field(-1, 'Mean of pulse time, >30 p.e. pulses')
    time_stddev_above_030_pe = Field(-1, 'Standard deviaton of pulse time, '
                                         '>30 p.e. pulses')
    relative_time_mean = Field(-1, 'Mean of pulse time relative to average of '
                                   'rest of pixels')
    relative_time_stddev = Field(-1, 'Standard deviaton of pulse time '
                                     'relative to average of rest of pixels')

    # keep number of events above a few thresholds, like a low-res histogram
    # of pulse charges (2 points per decade in charge in p.e.)
    # This could be done in a cleaner way with a 2d hist charge vs. pixel (TBD)
    num_pulses_above_0010_pe = Field(None, 'Number of >10 p.e. pulses')
    num_pulses_above_0030_pe = Field(None, 'Number of >30 p.e. pulses')
    num_pulses_above_0100_pe = Field(None, 'Number of >100 p.e. pulses')
    num_pulses_above_0300_pe = Field(None, 'Number of >300 p.e. pulses')
    num_pulses_above_1000_pe = Field(None, 'Number of >1000 p.e. pulses')

    def fill_event_wise_info(self, subrun_index, table, mask, geom,
                             histogram_binnings):
        """
        Fills the container fields that depend on event-wise DL1 info

        Parameters
        ----------
        subrun_index
        table: DL1 parameters, event-wise pandas dataframe, "parameters" from
        DL1 files
        mask: defines which events in table should be considered
        geom: camera geometry (in standard frame, *not* engineering one)
        histogram_binnings: container of type DL1DataCheckHinstogramBins which
        defines the binning of the various histograms

        Returns
        -------
        None

        """

        self.subrun_index = subrun_index
        # the elapsed time is between first and last event of the events in
        # table (we do not apply the mask here since we want to have all
        # events!)
        self.elapsed_time = table['dragon_time'][len(table)-1] - \
                            table['dragon_time'][0]
        self.num_events = mask.sum()
        self.num_cleaned_events = np.isfinite(table['intensity'][mask]).sum()
        self.ucts_trigger_type = \
            count_trig_types(table['ucts_trigger_type'][mask])
        self.trigger_type = \
            count_trig_types(table['trigger_type'][mask])
        self.mean_alt_tel = np.mean(table['alt_tel'])
        self.mean_az_tel = np.mean(table['az_tel'])

        # number of time samples per subrun to be stored in the container:
        n_samples = 50
        n_jump = 1+int(self.num_events/n_samples)
        # keep some info every n-jump-th event:
        sampled_event_ids = np.array(table['event_id'][mask][0::n_jump])
        tib_time = u.Quantity(np.array(table['tib_time'][mask][0::n_jump]),
                              u.s, copy=False)
        ucts_time = u.Quantity(np.array(table['ucts_time'][mask][0::n_jump]),
                               u.s, copy=False)
        dragon_time = u.Quantity(np.array(table['dragon_time'][mask][
                                          0::n_jump]), u.s, copy=False)
        # in case the resulting number of entries is <n_samples, we have to pad
        # the arrays, because hdf vector columns must have the same number of
        # elements in each row. We repeat the last value in the array
        padding = (0, n_samples-len(sampled_event_ids))
        self.sampled_event_ids = np.pad(sampled_event_ids, padding, mode='edge')
        self.tib_time = np.pad(tib_time, padding, mode='edge')
        self.ucts_time = np.pad(ucts_time, padding, mode='edge')
        self.dragon_time = np.pad(dragon_time, padding, mode='edge')

        # for the delta_t histogram we do not apply the mask, we want to have
        # all events present in the original table:
        delta_t = np.array(table['dragon_time'][1:]) - \
                  np.array(table['dragon_time'][:-1])
        counts, _, _, = plt.hist(delta_t*1.e3,
                                 bins=histogram_binnings.hist_delta_t)
        self.hist_delta_t = counts

        n_pixels = table['n_pixels'][mask]
        counts, _, _, = plt.hist(n_pixels,
                                 bins=histogram_binnings.hist_npixels)
        self.hist_npixels = counts

        n_islands = table['n_islands'][mask]
        counts, _, _, = plt.hist(n_islands,
                                 bins=histogram_binnings.hist_nislands)
        self.hist_nislands = counts

        intensity = table.loc[mask, 'intensity'].to_numpy()
        counts, _, _ = plt.hist(intensity,
                                bins=histogram_binnings.hist_intensity)
        self.hist_intensity = counts

        dist0 = table['r'][mask]
        counts, _, _ = plt.hist(dist0, bins=histogram_binnings.hist_dist0)
        self.hist_dist0 = counts

        counts, _, _ = \
            plt.hist(dist0[intensity > 200],
                     bins=histogram_binnings.hist_dist0_intensity_gt_200)
        self.hist_dist0_intensity_gt_200 = counts

        counts, _, _, _ = plt.hist2d(intensity,
                                     table.loc[mask, 'width'].to_numpy(),
                                     bins=histogram_binnings.hist_width)
        self.hist_width = counts

        counts, _, _, _ = plt.hist2d(intensity,
                                     table.loc[mask, 'length'].to_numpy(),
                                     bins=histogram_binnings.hist_length)
        self.hist_length = counts

        counts, _, _, _ = plt.hist2d(intensity,
                                     table.loc[mask, 'skewness'].to_numpy(),
                                     bins=histogram_binnings.hist_skewness)
        self.hist_skewness = counts

        psi = table.loc[mask, 'psi'].to_numpy()
        counts, _, _ = \
            plt.hist(psi, bins=histogram_binnings.hist_psi)
        self.hist_psi = counts

        counts, _, _, _ = \
            plt.hist2d(intensity, table.loc[mask, 'intercept'].to_numpy(),
                       bins=histogram_binnings.hist_intercept)
        self.hist_intercept = counts

        length = table.loc[mask, 'length'].to_numpy()
        tgrad = np.abs(table.loc[mask, 'time_gradient'].to_numpy())
        counts, _, _, _ = \
            plt.hist2d(length, tgrad,
                       bins=histogram_binnings.hist_tgrad_vs_length)
        self.hist_tgrad_vs_length = counts

        # We noticed an occasional pyplot error that seems to be fixed by
        # making sure that the coordinates passed to hist2d are ndarrays
        # (instead of Pandas data series)

        counts, _, _, _ = \
            plt.hist2d(length[intensity > 200], tgrad[intensity > 200],
                       bins=histogram_binnings.
                       hist_tgrad_vs_length_intensity_gt_200)
        self.hist_tgrad_vs_length_intensity_gt_200 = counts

        x = table['x'][mask]
        y = table['y'][mask]
        # event-wise, id of camera pixel which contains the image's cog:
        cog_pixid = geom.position_to_pix_index(np.array(x)*u.m,
                                               np.array(y)*u.m)
        self.cog_within_pixel = np.zeros(geom.n_pixels)
        for pix in cog_pixid:
            self.cog_within_pixel[pix] += 1
        self.cog_within_pixel_intensity_gt_200 = np.zeros(geom.n_pixels)
        # now the same for relatively bright images (intensity > 200 p.e.)
        select = intensity > 200
        for pix in cog_pixid[select]:
            self.cog_within_pixel_intensity_gt_200[pix] += 1

    def fill_pixel_wise_info(self, table, mask, histogram_binnings,
                             event_type = ''):
        """
        Fills the quantities that are calculated pixel-wise

        Parameters
        ----------
        table: DL1 parameters, event-wise python table "image" from DL1 files
        mask: indicates rows that have to be used for filling this container
        histogram_binnings: container of type DL1DataCheckHistogramBins, with
        definition of the binnings of all the histograms
        fill_time_info: fill the information on the pixel pulse times; can be
        set to False e.g. for pedestal events

        Returns
        -------
        None

        """
        charge = table.col('image')[mask]

        # average charge in each pixel through the subrun:
        self.charge_mean = charge.mean(axis=0)
        self.charge_stddev = charge.std(axis=0)

        # count, for each pixel, the number of entries with charge>x pe:
        self.num_pulses_above_0010_pe = np.sum(charge > 10, axis=0)
        self.num_pulses_above_0030_pe = np.sum(charge > 30, axis=0)
        self.num_pulses_above_0100_pe = np.sum(charge > 100, axis=0)
        self.num_pulses_above_0300_pe = np.sum(charge > 300, axis=0)
        self.num_pulses_above_1000_pe = np.sum(charge > 1000, axis=0)

        counts, _, _ = \
            plt.hist(charge[charge > 0].flatten(),
                     bins=histogram_binnings.hist_pixelchargespectrum)
        self.hist_pixelchargespectrum = counts

        # for pedestal events nothing else to be done:
        if event_type == 'pedestals':
            return

        # as of ctapipe 0.7.0, pulse times can take absurd values for pixels
        # containing very little signal. For time plots we require at least 1
        # p.e. We will also exclude NaNs from the calculations

        time = table.col('peak_time')[mask]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            # Make nan all pulse times for charges less than 1 p.e.:
            selected_entries = np.where(charge > 1, time, np.nan)
            # count how many valid pixels per event:
            n_valid_pixels = np.array([np.sum([~np.isnan(row)])
                                       for row in selected_entries])
            # mean and std dev for each pixel through the whole subrun:
            self.time_mean = np.nanmean(selected_entries, axis=0)
            self.time_stddev = np.nanstd(selected_entries, axis=0)
            # Now the average time in the camera, for each event:
            tmean = np.nanmean(selected_entries, axis=1)

            # tile it to the same shape as time, to allow subtracting it from each
            # pixel's pulse time:
            camera_time_mean = np.tile(tmean, (time.shape[1], 1)).transpose()
            # from camera mean time, to the same but excluding one pixel at a
            # time:
            camera_valid_pixels = np.tile(n_valid_pixels,
                                          (time.shape[1], 1)).transpose()
            rest_of_camera_valid_pixels = camera_valid_pixels - \
                                          np.ones(camera_valid_pixels.shape)
            mean_t_of_rest_of_camera = ((camera_time_mean *
                                         camera_valid_pixels - time) /
                                        rest_of_camera_valid_pixels)

            relative_time_t = time - mean_t_of_rest_of_camera
            selected_entries = np.where(charge > 1, relative_time_t, np.nan)
            self.relative_time_mean = np.nanmean(selected_entries, axis=0)
            self.relative_time_stddev = np.nanstd(selected_entries, axis=0)

            if event_type == 'flatfield':
                return

            selected_entries = np.where(charge>30, time, np.nan)
            self.time_mean_above_030_pe = np.nanmean(selected_entries, axis=0)
            self.time_stddev_above_030_pe = np.nanstd(selected_entries, axis=0)


def count_trig_types(array):
    """
    Counts the trigger of each type inside array

    Parameters
    ----------
    array: ndarray of event-wise trigger types

    Returns
    -------
    an ndarray of shape (10, 2) [i, j] means we found j events of type i

    """
    ucts_trig_types, counts = np.unique(array, return_counts=True)
    # write the different trigger types, then the number of events of
    # each type. Pad to 10 entries (more than enough for trigger types):
    ucts_trig_types = np.append(ucts_trig_types, (10-len(ucts_trig_types))*[0])
    counts = np.append(counts, (10 - len(counts)) * [0])
    return np.array([[t, n] for t, n in zip(ucts_trig_types, counts)])

class DL1DataCheckHistogramBins(Container):

    # delta_t between consecutive events (ms)
    hist_delta_t = Field(np.linspace(-1.e-2, 2., 200),
                         'hist_delta_t binning')
    # pixel charge and image intensity (units: p.e):
    hist_pixelchargespectrum = Field(np.logspace(-1., 4.7, 121),
                                     'hist_pixelchargespectrum binning')
    hist_intensity = Field(np.logspace(1., 6., 101), 'hist_intensity binning')

    # dist0, width and length (units: degrees):
    hist_dist0 = Field(np.linspace(0., 2.5, 50), 'hist_dist0 binning')
    hist_dist0_intensity_gt_200 = Field(np.linspace(0., 2.5, 50),
                                        'hist_dist0_intensity_gt_200 binning')
    hist_psi = Field(np.linspace(-100., 100., 101), 'hist_psi binning')
    hist_psi_intensity_gt_200 = Field(np.linspace(-100., 100., 101),
                                      'hist_psi_intensity_gt_200 binning')

    hist_nislands = Field(np.linspace(-0.5, 29.5, 31), 'hist_nislands binning')
    hist_npixels = Field(np.linspace(0.5, 2000.5, 400), 'hist_npixels binning')

    # 2d histograms
    # width and length vs. image intensity:
    hist_width = Field(np.array([np.logspace(0.7, 5.7, 101),
                                 np.linspace(0., 0.8, 101)]),
                       'hist_width binning')
    hist_length = Field(np.array([np.logspace(0.7, 5.7, 101),
                                  np.linspace(0., 1., 101)]),
                        'hist_length binning')
    hist_skewness = Field(np.array([np.logspace(0.7, 5.7, 101),
                                    np.linspace(-4., 4., 101)]),
                          'hist_skewness binning')
    # time gradient vs. length:
    hist_tgrad_vs_length = Field(np.array([np.linspace(0., 1.0, 101),
                                           np.linspace(0., 200., 101)]),
                                 'hist_tgrad_vs_length binning')
    hist_tgrad_vs_length_intensity_gt_200 =\
        Field(np.array([np.linspace(0., 1.0, 101), np.linspace(0., 50., 101)]),
              'hist_tgrad_vs_length_intensity_gt_200 binning')
    # time intercept (image time @ charge c.o.g.) vs. image intensity:
    hist_intercept = Field(np.array([np.logspace(0.7, 5.7, 101),
                                     np.linspace(-30., 40., 101)]),
                           'hist_intercept binning')
