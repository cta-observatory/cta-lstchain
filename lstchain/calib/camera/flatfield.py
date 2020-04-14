"""
Factory for the estimation of the flat field coefficients
"""


import numpy as np
from astropy import units as u
from ctapipe.calib.camera.flatfield import FlatFieldCalculator
from ctapipe.core.traits import  List


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
    time_cut_outliers = List(
        [0, 60],
        help='Interval (in waveform samples) of accepted time values'
    ).tag(config=True)
    charge_std_cut_outliers = List(
        [-3, 3],
        help='Interval (number of std) of accepted charge standard deviation around camera median value'
    ).tag(config=True)

    def __init__(self, **kwargs):
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
        super().__init__(**kwargs)

        self.log.info("Used events statistics : %d", self.sample_size)

        # members to keep state in calculate_relative_gain()
        self.num_events_seen = 0
        self.time_start = None  # trigger time of first event in sample
        self.trigger_time = None  # trigger time of present event

        self.charge_medians = None  # med. charge in camera per event in sample
        self.charges = None  # charge per event in sample
        self.arrival_times = None  # arrival time per event in sample
        self.sample_masked_pixels = None  # masked pixels per event in sample

    def _extract_charge(self, event):
        """
        Extract the charge and the time from a calibration event

        Parameters
        ----------
        event : general event container

        """

        waveforms = event.r1.tel[self.tel_id].waveform

        # Extract charge and time
        charge = 0
        peak_pos = 0
        if self.extractor:
            if self.extractor.requires_neighbors():
                camera = event.inst.subarray.tel[self.tel_id].camera
                self.extractor.neighbours = camera.neighbor_matrix_where

            charge, peak_pos = self.extractor(waveforms)

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

        pixel_mask = np.logical_or(
            event.mon.tel[self.tel_id].pixel_status.hardware_failing_pixels,
            event.mon.tel[self.tel_id].pixel_status.flatfield_failing_pixels)

        # real data
        if event.meta['origin'] != 'hessio':
            self.trigger_time = event.r1.tel[self.tel_id].trigger_time

        else:  # patches for MC data
            if event.trig.tels_with_trigger:
                self.trigger_time = event.trig.gps_time.unix
            else:
                self.trigger_time = 0


        if self.num_events_seen == 0:
            self.time_start = self.trigger_time
            self.setup_sample_buffers(waveform, self.sample_size)

        # extract the charge of the event and
        # the peak position (assumed as time for the moment)
        charge, arrival_time = self._extract_charge(event)

        self.collect_sample(charge, pixel_mask, arrival_time)

        sample_age = self.trigger_time - self.time_start

        # check if to create a calibration event
        if (
            sample_age > self.sample_duration
            or self.num_events_seen == self.sample_size
        ):
            # update the monitoring container
            self.store_results(event)
            return True

        else:

            return False

    def store_results(self, event):
        """
         Store stastical results in monitoring container

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
            'sample_time': (trigger_time - time_start) / 2 * u.s,
            'sample_time_range': [time_start, trigger_time] * u.s,
            'time_mean': np.ma.getdata(pixel_mean),
            'time_median': np.ma.getdata(pixel_median),
            'time_std': np.ma.getdata(pixel_std),
            'relative_time_median': np.ma.getdata(relative_median),
            'time_median_outliers': np.ma.getdata(time_median_outliers),

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

        # median over the sample per pixel
        pixel_median = np.ma.median(masked_trace_integral, axis=0)

        # mean over the sample per pixel
        pixel_mean = np.ma.mean(masked_trace_integral, axis=0)

        # std over the sample per pixel
        pixel_std = np.ma.std(masked_trace_integral, axis=0)

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
            'relative_gain_median': np.ma.getdata(np.ma.median(relative_gain_event, axis=0)),
            'relative_gain_mean': np.ma.getdata(np.ma.mean(relative_gain_event, axis=0)),
            'relative_gain_std': np.ma.getdata(np.ma.std(relative_gain_event, axis=0)),
            'charge_median': np.ma.getdata(pixel_median),
            'charge_mean': np.ma.getdata(pixel_mean),
            'charge_std': np.ma.getdata(pixel_std),
            'charge_std_outliers': np.ma.getdata(charge_std_outliers),
            'charge_median_outliers': np.ma.getdata(charge_median_outliers),
        }

