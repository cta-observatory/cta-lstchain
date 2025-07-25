"""
Factory for the estimation of the flat field coefficients
"""
import numpy as np
from astropy import units as u
from astropy.stats import sigma_clipped_stats

from ctapipe.core.traits import  List, Path, Int
from ctapipe.image.extractor import ImageExtractor
from ctapipe.calib.camera.flatfield import FlatFieldCalculator

from lstchain.calib.camera.time_sampling_correction import TimeSamplingCorrection
from lstchain.calib.camera.utils import check_outlier_mask

__all__ = [
    'FlasherFlatFieldCalculator'
]


class FlasherFlatFieldCalculator(FlatFieldCalculator):
    """Calculates flat-field parameters from flasher data
       based on the best algorithm described by S. Fegan in MST-CAM-TN-0060 (eq. 19)
       Pixels are defined as outliers on the base of a cut on the pixel charge median
       over the full sample distribution and the pixel signal time inside the
       waveform time


     Parameters:
     ----------
     charge_cut_outliers : List[2]
         Interval of accepted charge values (fraction with respect to camera median value)
     time_cut_outliers : List[2]
         Interval (in waveform samples) of accepted time values

    """

    charge_median_cut_outliers = List(
        [-0.3, 0.3],
        help='Interval of accepted charge values (fraction with respect to camera median value)'
    ).tag(config=True)
    charge_std_cut_outliers = List(
        [-3, 3],
        help='Interval (number of std) of accepted charge standard deviation around camera median value'
    ).tag(config=True)
    time_cut_outliers = List(
        [0, 60], help="Interval (in waveform samples) of accepted time values"
    ).tag(config=True)

    time_sampling_correction_path = Path(
        default_value=None,
        allow_none=True,
        exists=True, directory_ok=False,
        help='Path to time sampling correction file'
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

        """Calculates flat-field parameters from flasher data
           based on the best algorithm described by S. Fegan in MST-CAM-TN-0060 (eq. 19)
           Pixels are defined as outliers on the base of a cut on the pixel charge median
           over the full sample distribution and the pixel signal time inside the
           waveform time


         Parameters:
         ----------
         charge_cut_outliers : List[2]
             Interval of accepted charge values (fraction with respect to camera median value)
         time_cut_outliers : List[2]
             Interval (in waveform samples) of accepted time values

        """
        super().__init__(subarray, **kwargs)

        self.log.info("Used events statistics : %d", self.sample_size)

        # members to keep state in calculate_relative_gain()
        self.num_events_seen = 0
        self.time_start = None  # trigger time of first event in sample
        self.trigger_time = None  # trigger time of present event

        self.charge_medians = None  # med. charge in camera per event in sample
        self.charges = None  # charge per event in sample
        self.arrival_times = None  # arrival time per event in sample
        self.sample_masked_pixels = None  # masked pixels per event in sample

        # declare the charge sampling corrector
        if self.time_sampling_correction_path is not None:
            self.time_sampling_corrector = TimeSamplingCorrection(
                    time_sampling_correction_path=self.time_sampling_correction_path
            )
        else:
            self.time_sampling_corrector = None

        # fix for broken extractor setup in ctapipe baseclass
        self.extractor = ImageExtractor.from_name(
            self.charge_product, parent=self, subarray=subarray
        )

    def _extract_charge(self, event):
        """
        Extract the charge and the time from a calibration event

        Parameters
        ----------
        event : general event container

        """
        # copy the waveform be cause we do not want to change it for the moment
        waveforms = np.copy(event.r1.tel[self.tel_id].waveform)

        # In case of no gain selection the selected gain channels are  [0,0,..][1,1,..]
        no_gain_selection = np.zeros((waveforms.shape[0], waveforms.shape[1]), dtype=np.int64)
        no_gain_selection[1] = 1
        n_pixels = 1855

        # correct the r1 waveform for the sampling time corrections
        if self.time_sampling_corrector:
            waveforms*= (self.time_sampling_corrector.get_corrections(event,self.tel_id)
                         [no_gain_selection, np.arange(n_pixels)])

        # Extract charge and time
        charge = 0
        peak_pos = 0
        if self.extractor:
            broken_pixels = event.mon.tel[self.tel_id].pixel_status.hardware_failing_pixels
            dl1 = self.extractor(waveforms, self.tel_id, selected_gain_channel=None, broken_pixels=broken_pixels)
            charge = dl1.image
            peak_pos = dl1.peak_time

        # shift the time if time shift is already defined
        # (e.g. drs4 waveform time shifts for LST)
        time_shift = event.calibration.tel[self.tel_id].dl1.time_shift
        if time_shift is not None:
            peak_pos -= time_shift

        return charge, peak_pos

    def calculate_relative_gain(self, event):
        """
         calculate the flatfield statistical values
         and fill mon.tel[tel_id].flatfield container

         Parameters
         ----------
         event : general event container

         Returns: True if the mon.tel[tel_id].flatfield is updated, False otherwise

         """

        # initialize the np array at each cycle
        waveform = event.r1.tel[self.tel_id].waveform

        # re-initialize counter
        if self.num_events_seen == self.sample_size:
            self.num_events_seen = 0

        pixel_mask = event.mon.tel[self.tel_id].pixel_status.hardware_failing_pixels

        # time
        self.trigger_time = event.trigger.tel[self.tel_id].time
        
        if self.num_events_seen == 0:
            self.time_start = self.trigger_time
            self.setup_sample_buffers(waveform, self.sample_size)

        # extract the charge of the event and
        # the peak position (assumed as time for the moment)
        charge, arrival_time = self._extract_charge(event)

        self.collect_sample(charge, pixel_mask, arrival_time)

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
        if self.num_events_seen == 0:
            raise ValueError("No flat-field events in statistics, zero results")
       
        container = event.mon.tel[self.tel_id].flatfield

        # mask the part of the array not filled
        self.sample_masked_pixels[self.num_events_seen:] = 1

        relative_gain_results = self.calculate_relative_gain_results(
            self.charge_medians,
            self.charges,
            self.sample_masked_pixels
        )
        time_results = self.calculate_time_results(
            self.arrival_times,
            self.sample_masked_pixels,
            self.time_start,
            self.trigger_time
        )

        result = {
            'n_events': self.num_events_seen,
            **relative_gain_results,
            **time_results,
        }
        for key, value in result.items():
            setattr(container, key, value)

        # update the flatfield mask
        ff_charge_failing_pixels = np.logical_or(container.charge_median_outliers,
                                                 container.charge_std_outliers)
        event.mon.tel[self.tel_id].pixel_status.flatfield_failing_pixels = \
            np.logical_or(ff_charge_failing_pixels, container.time_median_outliers)

    def setup_sample_buffers(self, waveform, sample_size):
        """Initialize sample buffers"""

        n_channels = waveform.shape[0]
        n_pix = waveform.shape[1]
        shape = (sample_size, n_channels, n_pix)

        self.charge_medians = np.zeros((sample_size, n_channels))
        self.charges = np.zeros(shape)
        self.arrival_times = np.zeros(shape)
        self.sample_masked_pixels = np.zeros(shape)

    def collect_sample(self, charge, pixel_mask, arrival_time):
        """Collect the sample data"""

        # extract the charge of the event and
        # the peak position (assumed as time for the moment)

        good_charge = np.ma.array(charge, mask=pixel_mask)
        charge_median = np.ma.median(good_charge, axis=1)

        self.charges[self.num_events_seen] = charge
        self.arrival_times[self.num_events_seen] = arrival_time
        self.sample_masked_pixels[self.num_events_seen] = pixel_mask
        self.charge_medians[self.num_events_seen] = charge_median
        self.num_events_seen += 1

    def calculate_time_results(
        self,
        trace_time,
        masked_pixels_of_sample,
        time_start,
        trigger_time,
    ):
        """Calculate and return the time results """
        masked_trace_time = np.ma.array(
            trace_time,
            mask=masked_pixels_of_sample
        )

        # median over the sample per pixel
        pixel_median = np.ma.median(masked_trace_time, axis=0)

        # mean over the sample per pixel
        pixel_mean = np.ma.mean(masked_trace_time, axis=0)

        # std over the sample per pixel
        pixel_std = np.ma.std(masked_trace_time, axis=0)

        # median of the median over the camera
        median_of_pixel_median = np.ma.median(pixel_median, axis=1)

        # time outliers from median
        relative_median = pixel_median - median_of_pixel_median[:, np.newaxis]
        time_median_outliers = np.logical_or(pixel_median < self.time_cut_outliers[0],
                                             pixel_median > self.time_cut_outliers[1])

        return {
            'sample_time': (time_start + (trigger_time - time_start) / 2).unix * u.s,
            'sample_time_min': time_start.unix*u.s,
            'sample_time_max': trigger_time.unix*u.s,
            'time_mean': pixel_mean.filled(np.nan)*u.ns,
            'time_median': pixel_median.filled(np.nan)*u.ns,
            'time_std': pixel_std.filled(np.nan)*u.ns,
            'relative_time_median': relative_median.filled(np.nan)*u.ns,
            'time_median_outliers': time_median_outliers.filled(True),

        }

    def calculate_relative_gain_results(
        self,
        event_median,
        trace_integral,
        masked_pixels_of_sample,
    ):
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
        check_outlier_mask(outliers, self.log, "flatfield")

        # add outliers identified by sigma clipping for following operations
        masked_trace_integral.mask |= unused_values

        # median of the median over the camera
        median_of_pixel_median = np.ma.median(pixel_median, axis=1)

        # median of the std over the camera
        median_of_pixel_std = np.ma.median(pixel_std, axis=1)

        # std of the std over camera
        std_of_pixel_std = np.ma.std(pixel_std, axis=1)

        # relative gain
        relative_gain_event = masked_trace_integral / event_median[:, :, np.newaxis]

        # outliers from median
        charge_deviation = pixel_median - median_of_pixel_median[:, np.newaxis]

        charge_median_outliers = (
            np.logical_or(charge_deviation < self.charge_median_cut_outliers[0] * median_of_pixel_median[:,np.newaxis],
                          charge_deviation > self.charge_median_cut_outliers[1] * median_of_pixel_median[:,np.newaxis]))

        # outliers from standard deviation
        deviation = pixel_std - median_of_pixel_std[:, np.newaxis]
        charge_std_outliers = (
            np.logical_or(deviation < self.charge_std_cut_outliers[0] * std_of_pixel_std[:, np.newaxis],
                          deviation > self.charge_std_cut_outliers[1] * std_of_pixel_std[:, np.newaxis]))
        
        return {
            'relative_gain_median': np.ma.median(relative_gain_event, axis=0).filled(np.nan),
            'relative_gain_mean': np.ma.mean(relative_gain_event, axis=0).filled(np.nan),
            'relative_gain_std': np.ma.std(relative_gain_event, axis=0).filled(np.nan),
            'charge_median': pixel_median.filled(np.nan),
            'charge_mean': pixel_mean.filled(np.nan),
            'charge_std': pixel_std.filled(np.nan),
            'charge_std_outliers': charge_std_outliers.filled(True),
            'charge_median_outliers': charge_median_outliers.filled(True)
        }
