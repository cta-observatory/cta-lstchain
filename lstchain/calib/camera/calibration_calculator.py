"""
Component for the estimation of the calibration coefficients  events
"""

import numpy as np
from ctapipe.core import Component
from ctapipe.core import traits
from ctapipe.core.traits import  Float, List
from lstchain.calib.camera.flatfield import FlatFieldCalculator
from lstchain.calib.camera.pedestals import PedestalCalculator
from ctapipe.containers import EventType

__all__ = [
    'CalibrationCalculator',
    'LSTCalibrationCalculator'
]


class CalibrationCalculator(Component):
    """
    Parent class for the camera calibration calculators.
    Fills the MonitoringCameraContainer on the base of calibration events

    Parameters
    ----------

    flatfield_calculator: lstchain.calib.camera.flatfield
         The flatfield to use. If None, then FlatFieldCalculator
            will be used by default.

    pedestal_calculator: lstchain.calib.camera.pedestal
         The pedestal to use. If None, then
           PedestalCalculator will be used by default.

    kwargs

    """
    squared_excess_noise_factor = Float(
        1.222,
        help='Excess noise factor squared: 1+ Var(gain)/Mean(Gain)**2'
    ).tag(config=True)

    pedestal_product = traits.create_class_enum_trait(
        PedestalCalculator,
        default_value='PedestalIntegrator'
    )

    flatfield_product = traits.create_class_enum_trait(
        FlatFieldCalculator,
        default_value='FlasherFlatFieldCalculator'
    )

    classes = List([
                    FlatFieldCalculator,
                    PedestalCalculator
                    ]
                   + traits.classes_with_traits(FlatFieldCalculator)
                   + traits.classes_with_traits(PedestalCalculator)

                   )

    def __init__(
        self,
        subarray,
        parent=None,
        config=None,
        **kwargs
    ):

        """
        Parent class for the camera calibration calculators.
        Fills the MonitoringCameraContainer on the base of calibration events

        Parameters
        ----------

        flatfield_calculator: lstchain.calib.camera.flatfield
             The flatfield to use. If None, then FlatFieldCalculator
                will be used by default.

        pedestal_calculator: lstchain.calib.camera.pedestal
             The pedestal to use. If None, then
               PedestalCalculator will be used by default.

        """

        super().__init__(parent=parent, config=config,**kwargs)

        self.flatfield = FlatFieldCalculator.from_name(
            self.flatfield_product,
            parent=self,
            subarray = subarray
        )
        self.pedestal = PedestalCalculator.from_name(
            self.pedestal_product,
            parent=self,
            subarray = subarray
        )

        msg = "tel_id not the same for all calibration components"
        if self.pedestal.tel_id != self.flatfield.tel_id:
            raise ValueError(msg)

        self.tel_id = self.flatfield.tel_id

        self.log.debug(f"{self.pedestal}")
        self.log.debug(f"{self.flatfield}")


class LSTCalibrationCalculator(CalibrationCalculator):
    """
    Calibration calculator for LST camera
    Fills the MonitoringCameraContainer on the base of calibration events
    """


    def calculate_calibration_coefficients(self, event):
        """
        Calculate calibration coefficients from flatfield and pedestal statistics
        associated to the present event

        Parameters
        ----------
        event: ArrayArrayEventContainer

        """

        ped_data = event.mon.tel[self.tel_id].pedestal
        ff_data = event.mon.tel[self.tel_id].flatfield
        status_data = event.mon.tel[self.tel_id].pixel_status
        calib_data = event.mon.tel[self.tel_id].calibration

        # mask from pedestal and flat-field data
        monitoring_unusable_pixels = np.logical_or(status_data.pedestal_failing_pixels,
                                                   status_data.flatfield_failing_pixels)

        # calibration unusable pixels are an OR of all masks
        calib_data.unusable_pixels = np.logical_or(monitoring_unusable_pixels,
                                                   status_data.hardware_failing_pixels)

        # Extract calibration coefficients with F-factor method
        # Assume fixed excess noise factor must be known from elsewhere

        # calculate photon-electrons
        numerator = self.squared_excess_noise_factor  * (ff_data.charge_median - ped_data.charge_median) ** 2
        denominator = ff_data.charge_std ** 2 - ped_data.charge_std ** 2
        n_pe = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)

        # fill WaveformCalibrationContainer
        calib_data.time = ff_data.sample_time
        calib_data.time_min = ff_data.sample_time_min
        calib_data.time_max = ff_data.sample_time_max
        calib_data.n_pe = n_pe

        # find signal median of good pixels
        masked_npe = np.ma.array(n_pe, mask=calib_data.unusable_pixels)
        npe_signal_median = np.ma.median(masked_npe, axis=1)

        # Flat field factor
        numerator = npe_signal_median[:, np.newaxis]
        denominator = n_pe
        ff = np.divide(numerator, denominator, out=np.zeros_like(denominator), where=denominator != 0)

        # calibration coefficients
        numerator = n_pe * ff

        # correct the signal for the integration window
        denominator = (ff_data.charge_median - ped_data.charge_median) 
        calib_data.dc_to_pe = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)

        # flat-field time corrections
        calib_data.time_correction = -ff_data.relative_time_median

        calib_data.pedestal_per_sample = ped_data.charge_median / self.pedestal.extractor.window_width.tel[self.tel_id]

        # put to zero unusable pixels
        calib_data.dc_to_pe[calib_data.unusable_pixels] = 0
        calib_data.pedestal_per_sample[calib_data.unusable_pixels] = 0

        # eliminate inf values id any (still necessary?)
        calib_data.dc_to_pe[np.isinf(calib_data.dc_to_pe)] = 0

    def process_interleaved(self, event):
        """
        Process interleaved calibration events (pedestals and FF)
        Parameters
        ----------
        """
        new_ped = False
        new_ff = False

        # if pedestal event:
        if event.trigger.event_type == EventType.SKY_PEDESTAL:

            new_ped = self.pedestal.calculate_pedestals(event)

        # if flat-field event:
        elif event.trigger.event_type == EventType.FLATFIELD:

            new_ff = self.flatfield.calculate_relative_gain(event)

            # if new ff, calculate new calibration coefficients
            if new_ff:
                self.calculate_calibration_coefficients(event)

        return new_ped, new_ff

    def output_interleaved_results(self, event):
        """
        Output interleaved results on request

        """
        new_ped = False
        new_ff = False

        # store results
        if self.pedestal.num_events_seen > 0:
            self.pedestal.store_results(event)
            new_ped = True

            if self.flatfield.num_events_seen > 0:
                self.flatfield.store_results(event)

                # calculates calibration values
                self.calculate_calibration_coefficients(event)
                new_ff = True

        return new_ped, new_ff
