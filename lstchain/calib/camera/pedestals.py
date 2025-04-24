"""
Factory for the estimation of the flat field coefficients
"""


import numpy as np
from astropy import units as u
from astropy.stats import sigma_clipped_stats

from ctapipe.core.traits import List, Path, Int
from ctapipe.image.extractor import ImageExtractor

from ctapipe.calib.camera.pedestals import PedestalCalculator
from lstchain.calib.camera.time_sampling_correction import TimeSamplingCorrection
from lstchain.calib.camera.utils import check_outlier_mask


__all__ = [
    'PedestalIntegrator'
]


class PedestalIntegrator(PedestalCalculator):
    """Calculates pedestal parameters integrating the charge of pedestal events:
       the pedestal value corresponds to the charge estimated with the selected
       charge extractor
       The pixels are set as outliers on the base of a cut on the pixel charge median
       over the pedestal sample and the pixel charge standard deviation over
       the pedestal sample with respect to the camera median values


     Parameters:
     ----------
     charge_median_cut_outliers : List[2]
         Interval (number of std) of accepted charge values around camera median value
     charge_std_cut_outliers : List[2]
         Interval (number of std) of accepted charge standard deviation around camera median value

     """
    charge_median_cut_outliers = List(
        [-3, 3],
        help='Interval (number of std) of accepted charge values around camera median value'
    ).tag(config=True)

    charge_std_cut_outliers = List(
        [-3, 3],
        help='Interval (number of std) of accepted charge standard deviation around camera median value'
    ).tag(config=True)

    time_sampling_correction_path = Path(
        default_value=None,
        allow_none=True,
        directory_ok=False,
        help='Path to time sampling correction file',
    ).tag(config=True)

    sigma_clipping_max_sigma = Int(
        default_value=4,
        help="max_sigma value for the sigma clipping outlier removal",
    ).tag(config=True)

    sigma_clipping_iterations = Int(
        default_value=5,
        help="Number of iterations for the sigma clipping outlier removal",
    ).tag(config=True)


    def __init__(self, subarray, **kwargs):
        """Calculates pedestal parameters integrating the charge of pedestal events:
           the pedestal value corresponds to the charge estimated with the selected
           charge extractor
           The pixels are set as outliers on the base of a cut on the pixel charge median
           over the pedestal sample and the pixel charge standard deviation over
           the pedestal sample with respect to the camera median values


         Parameters:
         ----------
         charge_median_cut_outliers : List[2]
             Interval (number of std) of accepted charge values around camera median value
         charge_std_cut_outliers : List[2]
             Interval (number of std) of accepted charge standard deviation around camera median value
        """

        super().__init__(subarray, **kwargs)

        self.log.info("Used events statistics : %d", self.sample_size)

        # members to keep state in calculate_relative_gain()
        self.num_events_seen = 0
        self.time_start = None  # trigger time of first event in sample
        self.trigger_time = None  # trigger time of present event

        self.charge_medians = None  # med. charge in camera per event in sample
        self.charges = None  # charge per event in sample
        self.sample_masked_pixels = None  # pixels tp be masked per event in sample

        # declare the charge sampling corrector
        if self.time_sampling_correction_path is not None:
            self.time_sampling_corrector = TimeSamplingCorrection(
                time_sampling_correction_path=self.time_sampling_correction_path)
        else:
            self.time_sampling_corrector = None

        # fix for broken extractor setup in ctapipe baseclass
        self.extractor = ImageExtractor.from_name(
            self.charge_product, parent=self, subarray=subarray
        )
       

    def _extract_charge(self, event):
        """
        Extract the charge and the time from a pedestal event

        Parameters
        ----------

        event : general event container

        """
        
        # copy the waveform be cause we do not want to change it for the moment
        waveforms = np.copy(event.r1.tel[self.tel_id].waveform)

        # pedestal event do not have gain selection
        no_gain_selection = np.zeros((waveforms.shape[0], waveforms.shape[1]), dtype=np.int64)
        no_gain_selection[1] = 1
        n_pixels = 1855

        # correct the r1 waveform for the sampling time corrections
        if self.time_sampling_corrector:
            waveforms *= (self.time_sampling_corrector.get_corrections(event, self.tel_id)
            [no_gain_selection, np.arange(n_pixels)])

        # Extract charge and time
        charge = 0
        peak_pos = 0
        if self.extractor:
            broken_pixels = event.mon.tel[self.tel_id].pixel_status.hardware_failing_pixels
         
            dl1 = self.extractor(waveforms, self.tel_id, 
                                 selected_gain_channel=None, 
                                 broken_pixels=broken_pixels)
            charge = dl1.image
            peak_pos = dl1.peak_time

        return charge, peak_pos

    def calculate_pedestals(self, event):
        """
        calculate the pedestal statistical values from
        the charge extracted from pedestal events
        and fill the mon.tel[tel_id].pedestal container

        Parameters
        ----------
        event : general event container

        Returns: True if the mon.tel[tel_id].pedestal is updated, False otherwise

        """
        
        # initialize the np array at each cycle
        waveform = event.r1.tel[self.tel_id].waveform
    
        # re-initialize counter
        if self.num_events_seen == self.sample_size:
            self.num_events_seen = 0

        pixel_mask = event.mon.tel[self.tel_id].pixel_status.hardware_failing_pixels

        self.trigger_time = event.trigger.time
        
        if self.num_events_seen == 0:
            self.time_start = self.trigger_time
            self.setup_sample_buffers(waveform, self.sample_size)

        # extract the charge of the event and
        # the peak position (assumed as time for the moment)
        charge = self._extract_charge(event)[0]

        self.collect_sample(charge, pixel_mask)

        sample_age = (self.trigger_time - self.time_start).to_value(u.s)
        # check if to create a calibration event
        if (self.num_events_seen > 0 and
                (sample_age > self.sample_duration or
                self.num_events_seen == self.sample_size)
        ):
            # update the monitoring container
            self.store_results(event)
            return True

        else:

            return False

    def store_results(self, event):
        """
         Store statistical results in monitoring container

         Parameters
         ----------
         event : general event container
        """

        # something wrong if you are here and no statistic is there
        if self.num_events_seen == 0:
            raise ValueError("No pedestal events in statistics, zero results")

        container = event.mon.tel[self.tel_id].pedestal

        # mask the part of the array not filled
        self.sample_masked_pixels[self.num_events_seen:] = 1

        pedestal_results = self.calculate_pedestal_results(
            self.charges,
            self.sample_masked_pixels,
        )
        time_results = calculate_time_results(
            self.time_start,
            self.trigger_time,
        )

        result = {
            'n_events': self.num_events_seen,
            **pedestal_results,
            **time_results,
        }
        for key, value in result.items():
            setattr(container, key, value)

        # update pedestal mask
        event.mon.tel[self.tel_id].pixel_status.pedestal_failing_pixels = \
            np.logical_or(container.charge_median_outliers, container.charge_std_outliers)

    def setup_sample_buffers(self, waveform, sample_size):
        """Initialize sample buffers"""
       
        n_channels = waveform.shape[0]
        n_pix = waveform.shape[1]
        shape = (sample_size, n_channels, n_pix)

        self.charge_medians = np.zeros((sample_size, n_channels))
        self.charges = np.zeros(shape)
        self.sample_masked_pixels = np.zeros(shape)

    def collect_sample(self, charge, pixel_mask):
        """Collect the sample data"""

        good_charge = np.ma.array(charge, mask=pixel_mask)
        charge_median = np.ma.median(good_charge, axis=1)

        self.charges[self.num_events_seen] = charge
        self.sample_masked_pixels[self.num_events_seen] = pixel_mask
        self.charge_medians[self.num_events_seen] = charge_median
        self.num_events_seen += 1

    def calculate_pedestal_results(self, trace_integral, masked_pixels_of_sample):
        """Calculate and return the sample statistics"""
        masked_trace_integral = np.ma.array(
            trace_integral,
            mask=masked_pixels_of_sample
        )
        
        # mean and std over the sample per pixel
        max_sigma = self.sigma_clipping_max_sigma
        pixel_mean, pixel_median, pixel_std = sigma_clipped_stats(
            masked_trace_integral,
            sigma=max_sigma,
            maxiters=self.sigma_clipping_iterations,
            cenfunc="mean",
            axis=0,
        )

        # mask pixels without defined statistical values (mainly due to hardware problems)
        pixel_mean = np.ma.array(pixel_mean, mask=np.isnan(pixel_mean))
        pixel_median = np.ma.array(pixel_median, mask=np.isnan(pixel_median))
        pixel_std = np.ma.array(pixel_std, mask=np.isnan(pixel_std))

        unused_values = np.abs(masked_trace_integral - pixel_mean) > (max_sigma * pixel_std)

        # only warn for values discard in the sigma clipping, not those from before
        outliers = unused_values & (~masked_trace_integral.mask)
        check_outlier_mask(outliers, self.log, "pedestal")

        # add outliers identified by sigma clipping for following operations
        masked_trace_integral.mask |= unused_values

        # median over the camera
        median_of_pixel_median = np.ma.median(pixel_median, axis=1)

        # std of median over the camera
        std_of_pixel_median = np.ma.std(pixel_median, axis=1)

        # median of the std over the camera
        median_of_pixel_std = np.ma.median(pixel_std, axis=1)

        # std of the std over camera
        std_of_pixel_std = np.ma.std(pixel_std, axis=1)

        # outliers from standard deviation
        deviation = pixel_std - median_of_pixel_std[:, np.newaxis]
        charge_std_outliers = np.logical_or(
            deviation < self.charge_std_cut_outliers[0] * std_of_pixel_std[:,np.newaxis],
            deviation > self.charge_std_cut_outliers[1] * std_of_pixel_std[:,np.newaxis],
        )

        # outliers from median
        deviation = pixel_median - median_of_pixel_median[:, np.newaxis]
        charge_median_outliers = np.logical_or(
            deviation < self.charge_median_cut_outliers[0] * std_of_pixel_median[:,np.newaxis],
            deviation > self.charge_median_cut_outliers[1] * std_of_pixel_median[:,np.newaxis],
        )

        return {
            'charge_median': pixel_median.filled(np.nan),
            'charge_mean': pixel_mean.filled(np.nan),
            'charge_std': pixel_std.filled(np.nan),
            'charge_std_outliers': charge_std_outliers.filled(True),
            'charge_median_outliers': charge_median_outliers.filled(True)
        }


def calculate_time_results(
    time_start,
    trigger_time,
):
    """Calculate and return the sample time"""
    return {
        'sample_time': (time_start + (trigger_time - time_start) / 2).unix*u.s,
        'sample_time_min': time_start.unix*u.s,
        'sample_time_max': trigger_time.unix*u.s,
    }


