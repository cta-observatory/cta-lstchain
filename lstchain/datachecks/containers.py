"""
Containers for data check
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, AltAz

import warnings
from ctapipe.core import Container, Field
from ctapipe.utils import get_bright_stars
from ctapipe.coordinates import CameraFrame, TelescopeFrame
from ctapipe_io_lst.constants import LST1_LOCATION

__all__ = [
    'DL1DataCheckContainer',
    'DL1DataCheckHistogramBins',
    'count_trig_types',
]


class DL1DataCheckContainer(Container):
    """
    Container to store the subrun-wise outcome of the DL1 data check
    """

    # scalar quantities:
    subrun_index = Field(-1, 'Subrun index')
    elapsed_time = Field(-1*u.s, 'Subrun time duration (from Dragon)', unit=u.s)
    num_events = Field(-1, 'Total number of events')
    num_cleaned_events = Field(-1, 'Number of events surviving cleaning')
    trigger_type = Field(-1, 'Number of events per trigger type')
    ucts_trigger_type = Field(-1, 'Number of events per ucts trigger type')
    num_ucts_jumps = Field(-1, 'Number of observed (and corrected) UCTS jumps')
    mean_alt_tel = Field(np.nan*u.rad, 'Mean telescope altitude', unit=u.rad)
    mean_az_tel = Field(np.nan*u.rad, 'Mean telescope azimuth', unit=u.rad)
    tel_ra = Field(np.nan*u.deg, 'Telescope pointing RA', unit=u.deg)
    tel_dec = Field(np.nan*u.deg, 'Telescope pointing declination', unit=u.deg)

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

    num_nearby_stars = Field(-1, 'Number of nearby bright stars')
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
        table: DL1 parameters, event-wise astropy table, "parameters" from
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
        self.elapsed_time = (table['dragon_time'][len(table) - 1] -
                             table['dragon_time'][0]) * u.s
        self.num_events = mask.sum()
        self.num_cleaned_events = np.isfinite(table['intensity'][mask]).sum()
        self.ucts_trigger_type = \
            count_trig_types(table['ucts_trigger_type'][mask])
        self.trigger_type = \
            count_trig_types(table['trigger_type'][mask])
        if 'ucts_jump' in table.columns:
            # After one (or n) genuine UCTS jumps in a run, the first event (or n events)
            # of every subsequent subrun file (if analyzed on its own) will have ucts_jump=True,
            # but these are not new jumps, just the ones from previous subruns, so they should
            # not be counted.
            uj = table['ucts_jump'].data.copy()
            # find the first False value, and set to False also all the earlier ones:
            if np.sum(uj == False) > 0:
                first_non_jump = np.where(uj==False)[0][0]
                uj[:first_non_jump] = False

            # count only the jumps occurring in events of the type we are
            # processing:
            self.num_ucts_jumps = np.sum(uj[mask])

        # since azimuth can go through 0, just take the pointing of the
        # event in the middle of the table (the actual mean value would be
        # problematic for culmination towards north, az= ~0  ~2pi):
        self.mean_az_tel = table['az_tel'].quantity[int(len(table)/2)]
        self.mean_alt_tel = table['alt_tel'].quantity[int(len(table)/2)]
        time_utc = Time(table['dragon_time'][int(len(table)/2)],
                        format="unix", scale="utc")
        # Calculate telescope pointing in sky coordinates
        telescope_pointing = SkyCoord(alt=self.mean_alt_tel,
                                      az=self.mean_az_tel,
                                      frame=AltAz(obstime=time_utc,
                                                  location=LST1_LOCATION))
        self.tel_ra = telescope_pointing.icrs.ra.to(u.deg)
        self.tel_dec = telescope_pointing.icrs.dec.to(u.deg)

        # number of time samples per subrun to be stored in the container:
        n_samples = 50
        n_jump = 1 + int(self.num_events / n_samples)
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
        padding = (0, n_samples - len(sampled_event_ids))
        self.sampled_event_ids = np.pad(sampled_event_ids, padding, mode='edge')
        self.tib_time = np.pad(tib_time, padding, mode='edge')
        self.ucts_time = np.pad(ucts_time, padding, mode='edge')
        self.dragon_time = np.pad(dragon_time, padding, mode='edge')

        # for the delta_t histogram we do not apply the mask, we want to have
        # all events present in the original table:
        delta_t = np.array(table['dragon_time'][1:]) - \
                  np.array(table['dragon_time'][:-1])
        counts, _, _, = plt.hist(delta_t * 1.e3,
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

        intensity = table[mask]['intensity'].data
        counts, _, _ = plt.hist(intensity,
                                bins=histogram_binnings.hist_intensity)
        self.hist_intensity = counts

        dist0 = table[mask]['r']
        counts, _, _ = plt.hist(dist0, bins=histogram_binnings.hist_dist0)
        self.hist_dist0 = counts

        counts, _, _ = \
            plt.hist(dist0[intensity > 200],
                     bins=histogram_binnings.hist_dist0_intensity_gt_200)
        self.hist_dist0_intensity_gt_200 = counts

        counts, _, _, _ = plt.hist2d(intensity,
                                     table[mask]['width'].data,
                                     bins=histogram_binnings.hist_width)
        self.hist_width = counts

        counts, _, _, _ = plt.hist2d(intensity,
                                     table[mask]['length'].data,
                                     bins=histogram_binnings.hist_length)
        self.hist_length = counts

        counts, _, _, _ = plt.hist2d(intensity,
                                     table[mask]['skewness'].data,
                                     bins=histogram_binnings.hist_skewness)
        self.hist_skewness = counts

        psi = table[mask]['psi'].data
        counts, _, _ = \
            plt.hist(psi, bins=histogram_binnings.hist_psi)
        self.hist_psi = counts

        counts, _, _, _ = \
            plt.hist2d(intensity, table[mask]['intercept'].data,
                       bins=histogram_binnings.hist_intercept)
        self.hist_intercept = counts

        length = table[mask]['length'].data
        tgrad = np.abs(table[mask]['time_gradient'].data)
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

        # event-wise, id of camera pixel which contains the image's cog:
        # we skip nan coordinates to avoid a lot of ctapipe warnings
        cog_pixid = np.zeros(mask.sum(), dtype='int')
        for k, x, y in zip(range(mask.sum()),
                           table['x'].quantity[mask],
                           table['y'].quantity[mask]):
            if np.isfinite(x) & np.isfinite(y):
                cog_pixid[k] = geom.position_to_pix_index(x, y)
            else:
                cog_pixid[k] = -1

        self.cog_within_pixel = np.zeros(geom.n_pixels)
        # explicitly skip -1 values, lest they end in the highest pixel id...
        # position_to_pix_index returns -1 for nan inputs or x,y outside camera!
        for pix in cog_pixid[cog_pixid != -1]:
            self.cog_within_pixel[pix] += 1

        self.cog_within_pixel_intensity_gt_200 = np.zeros(geom.n_pixels)
        # now the same for relatively bright images (intensity > 200 p.e.)
        select = intensity > 200
        for pix in cog_pixid[select]:
            if pix == -1:  # out of camera or non-reconstructed event
                continue
            self.cog_within_pixel_intensity_gt_200[pix] += 1

    def fill_pixel_wise_info(self, table, mask, histogram_binnings,
                             focal_length, geom, event_type = ''):
        """
        Fills the quantities that are calculated pixel-wise

        Parameters
        ----------
        table: DL1 parameters, event-wise astropy table "image" from DL1 files
        mask: indicates rows that have to be used for filling this container
        histogram_binnings: container of type DL1DataCheckHistogramBins, with
                            definition of the binnings of all the histograms
        focal_length: quantity; telescope focal length
        geom: camera geometry, ctapipe.instrument.camera.geometry.CameraGeometry
        event_type: 'pedestals' 'flatfield' or 'cosmics'

        Returns
        -------
        None

        """
        charge = table['image'][mask]

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

        # Find bright stars (mag<=8 within 3 deg of telescope pointing) and
        # count how many of them are close to each pixel:

        # Just use the time in the middle of the subrun, from the sampled times:
        sampled_times = self.dragon_time
        obstime = Time(sampled_times[int(len(sampled_times)/2)],
                       scale='utc', format='unix')
        horizon_frame = AltAz(location=LST1_LOCATION, obstime=obstime)
        pointing = SkyCoord(az=self.mean_az_tel,
                            alt=self.mean_alt_tel,
                            frame=horizon_frame)
        bright_stars = get_bright_stars(pointing=pointing, radius=3*u.deg,
                                        magnitude_cut=8)
        # Account for average relative spot shift (outwards) due to coma
        # aberration:
        relative_shift = 1.0466 # For LST's paraboloid
        camera_frame = CameraFrame(telescope_pointing=pointing,
                                   focal_length=focal_length*relative_shift,
                                   obstime=obstime,
                                   location=LST1_LOCATION)
        telescope_frame = TelescopeFrame(obstime=obstime, location=LST1_LOCATION)

        # radius around star within which we consider the pixel may be affected
        # (hence we will later not raise a flag if e.g. its pedestal std dev is
        # high):
        r_around_star = 0.25 * u.deg
        stars = bright_stars['ra_dec']
        pixels = SkyCoord(x=geom.pix_x, y=geom.pix_y,
                          frame=camera_frame).transform_to(telescope_frame)
        angular_distance = pixels[:, np.newaxis].separation(stars)

        # This counts how many stars are close to each pixel; stars can be
        # counted more than once (for different pixels!) so don't add them up.
        self.num_nearby_stars = np.count_nonzero(angular_distance < r_around_star,
                                                 axis=1)

        # for pedestal events nothing else to be done:
        if event_type == 'pedestals':
            return

        # For time plots we require at least 1 p.e. We will also exclude nans
        # from the calculations

        time = table['peak_time'][mask]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            # Make nan all pulse times for charges less than 1 p.e.:
            time = np.where(charge > 1, time, np.nan)
            # count how many valid pixels per event:
            n_valid_pixels = np.count_nonzero(np.isfinite(time), axis=1)

            # mean and std dev for each pixel through the whole subrun:
            self.time_mean = np.nanmean(time, axis=0)
            self.time_stddev = np.nanstd(time, axis=0)
            # Now the average time in the camera, for each event:
            tmean = np.nanmean(time, axis=1)

            # We do the calculation of the relative times event by event,
            # instead of using events*pixels matrices, because keeping all
            # necessary matrices in memory to do it in one go results in too
            # large memory use (>5GB)
            for ievt, event_pixtimes in enumerate(time):
                # for each pixel we want the mean time of all the other pixels:
                mean_t_other = np.ones_like(event_pixtimes) * tmean[ievt]
                mean_t_other *= n_valid_pixels[ievt]
                mean_t_other -= event_pixtimes
                mean_t_other /= (n_valid_pixels[ievt] - 1)
                time[ievt] -= mean_t_other

            # Now time contains the times of each pixel relative to the average
            # of the rest of the pixels in the same event

            self.relative_time_mean = np.nanmean(time, axis=0)
            self.relative_time_stddev = np.nanstd(time, axis=0)

            if event_type == 'flatfield':
                return

            selected_entries = np.where(charge > 30, time, np.nan)
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
    types, counts = np.unique(array, return_counts=True)
    # write the different trigger types, then the number of events of
    # each type. Pad to 10 entries (more than enough for trigger types):
    types = np.append(types, (10 - len(types)) * [0])
    counts = np.append(counts, (10 - len(counts)) * [0])
    return np.array([[t, n] for t, n in zip(types, counts)])



with np.printoptions(threshold=3, precision=3, edgeitems=1):
    class DL1DataCheckHistogramBins(Container):
        """
        Histogram bins for the DL1 Datacheck
        """

        # delta_t between consecutive events (ms)
        hist_delta_t = Field(
            default_factory=lambda: np.linspace(-1.e-2, 2., 200),
            description='hist_delta_t binning',
        )
        # pixel charge and image intensity (units: p.e):
        hist_pixelchargespectrum = Field(
            default_factory=lambda: np.logspace(-1., 4.7, 121),
            description='hist_pixelchargespectrum binning',
        )
        hist_intensity = Field(
            default_factory=lambda: np.logspace(1., 6., 101),
            description='hist_intensity binning',
        )

        # dist0, width and length (units: degrees):
        hist_dist0 = Field(
            default_factory=lambda: np.linspace(0., 2.5, 50),
            description='hist_dist0 binning',
        )
        hist_dist0_intensity_gt_200 = Field(
            default_factory=lambda: np.linspace(0., 2.5, 50),
            description='hist_dist0_intensity_gt_200 binning',
        )
        hist_psi = Field(
            default_factory=lambda: np.linspace(-100., 100., 101),
            description='hist_psi binning',
        )
        hist_psi_intensity_gt_200 = Field(
            default_factory=lambda: np.linspace(-100., 100., 101),
            description='hist_psi_intensity_gt_200 binning'
        )

        hist_nislands = Field(
            default_factory=lambda: np.linspace(-0.5, 29.5, 31),
            description='hist_nislands binning'
        )
        hist_npixels = Field(
            default_factory=lambda: np.linspace(0.5, 2000.5, 400),
            description='hist_npixels binning',
        )

        # 2d histograms
        # width and length vs. image intensity:
        hist_width = Field(
            default_factory=lambda: np.array([np.logspace(0.7, 5.7, 101), np.linspace(0., 0.8, 101)]),
            description='hist_width binning'
        )
        hist_length = Field(
            default_factory=lambda: np.array([np.logspace(0.7, 5.7, 101), np.linspace(0., 1., 101)]),
            description='hist_length binning'
        )
        hist_skewness = Field(
            default_factory=lambda: np.array([np.logspace(0.7, 5.7, 101), np.linspace(-4., 4., 101)]),
            description='hist_skewness binning',
        )
        # time gradient vs. length:
        hist_tgrad_vs_length = Field(
            default_factory=lambda: np.array([np.linspace(0., 1.0, 101), np.linspace(0., 200., 101)]),
            description='hist_tgrad_vs_length binning',
        )
        hist_tgrad_vs_length_intensity_gt_200 = Field(
            default_factory=lambda: np.array([np.linspace(0., 1.0, 101), np.linspace(0., 50., 101)]),
            description='hist_tgrad_vs_length_intensity_gt_200 binning',
        )
        # time intercept (image time @ charge c.o.g.) vs. image intensity:
        hist_intercept = Field(
            default_factory=lambda: np.array([np.logspace(0.7, 5.7, 101), np.linspace(-30., 40., 101)]),
            description='hist_intercept binning'
        )
