"""
Extract flat field coefficients from flasher data files.
"""
import numpy as np
from ctapipe.containers import PixelStatusContainer
from ctapipe.core import Provenance, Tool, traits
from ctapipe.io import EventSource, HDF5TableWriter

from lstchain.calib.camera.calibration_calculator import CalibrationCalculator
from ctapipe.containers import EventType
from tqdm.autonotebook import tqdm
from lstchain.io import add_config_metadata, add_global_metadata, global_metadata, write_metadata


class CalibrationHDF5Writer(Tool):
    """
    Tool that generates a HDF5 file with camera calibration coefficients.
    Input file must contain interleaved pedestal and flat-field events.

    For getting help run:
    lstchain_create_calibration_file --help
    """

    name = "CalibrationHDF5Writer"
    description = "Generate a HDF5 file with camera calibration coefficients"

    one_event = traits.Bool(False, help="Stop after first calibration event").tag(config=True)

    output = traits.Path(
        help="Name of the output file",
        directory_ok=False,
        default_value="calibration.hdf5",
    ).tag(config=True)
    drs4_pedestal_path = traits.Path(
        help="Path of the DRS4 pedestal file",
        directory_ok=False,
        exists=True,
    ).tag(config=True)
    progress_bar = traits.Bool(
        help="Show progress bar during processing",
        default_value=True,
    ).tag(config=True)

    calibration_product = traits.create_class_enum_trait(
        CalibrationCalculator, default_value="LSTCalibrationCalculator"
    )

    aliases = {
        ("i", "input_file"): "EventSource.input_url",
        ("o", "output_file"): "CalibrationHDF5Writer.output",
        "max_events": "EventSource.max_events",
        "calibration_product": "CalibrationHDF5Writer.calibration_product",
        "drs4_pedestal_path": "LSTR0Corrections.drs4_pedestal_path",
    }

    flags = {
        "one-event": (
            {"CalibrationHDF5Writer": {"one_event": True}},
            "Stop after first calibration event",
        )
    }

    classes = [] + traits.classes_with_traits(EventSource) + traits.classes_with_traits(CalibrationCalculator)

    def setup(self):

        self.tot_events = 0
        self.simulation = False

        self.eventsource = EventSource.from_config(parent=self)
        self.tel_id = self.eventsource.lst_service.telescope_id
        if self.eventsource.r0_r1_calibrator.drs4_pedestal_path.tel[self.tel_id] is None:
            raise IOError("Missing (mandatory) drs4 pedestal file in traitlets")

        # if data remember how many events in the files
        if "LSTEventSource" in str(type(self.eventsource)):
            self.tot_events = len(self.eventsource.multi_file)
            self.log.debug(f"Input file has file {self.tot_events} events")
        else:
            self.tot_events = self.eventsource.max_events
            self.simulation = True

        self.processor = CalibrationCalculator.from_name(
            self.calibration_product, parent=self, subarray=self.eventsource.subarray
        )

        self.writer = HDF5TableWriter(
            filename=self.output,
            group_name=f"tel_{self.tel_id}",
            overwrite=True,
        )

    def start(self):
        """Calibration coefficient calculator"""

        new_ped = False
        new_ff = False
        end_of_file = False

        # skip the first events which are badly drs4 corrected
        events_to_skip = 1000

        self.log.debug(f"Start loop")
        for count, event in tqdm(
            enumerate(self.eventsource),
            desc=self.eventsource.__class__.__name__,
            total=self.tot_events,
            unit="ev",
            disable=not self.progress_bar,
        ):

            # if last event write results
            max_events_reached = (
                self.eventsource.max_events is not None and count == self.eventsource.max_events - 1
            )
            if count == self.tot_events - 1 or max_events_reached:
                self.log.debug(f"Last event, count = {count}")
                end_of_file = True

            # save the config - to be retrieved as data.meta['config']
            if count == 0:
                if self.simulation:
                    self.initialize_pixel_status(event.mon.tel[self.tel_id], event.r1.tel[self.tel_id].waveform.shape)

                ped_data = event.mon.tel[self.tel_id].pedestal
                ped_data.meta["config"] = self.config

                ff_data = event.mon.tel[self.tel_id].flatfield
                ff_data.meta["config"] = self.config

                status_data = event.mon.tel[self.tel_id].pixel_status
                status_data.meta["config"] = self.config

                calib_data = event.mon.tel[self.tel_id].calibration
                calib_data.meta["config"] = self.config

            # skip first events which are badly drs4 corrected
            if not self.simulation and count < events_to_skip:
                continue
            # if pedestal event
            if event.trigger.event_type == EventType.SKY_PEDESTAL or (

                self.simulation
                and np.median(np.sum(event.r1.tel[self.tel_id].waveform[0], axis=1))
                < self.processor.minimum_hg_charge_median
            ):

                new_ped = self.processor.pedestal.calculate_pedestals(event)

            # if flat-field event: no calibration TIB for the moment,
            # use a cut on the charge for ff events and on std for rejecting Magic Lidar events
            elif event.trigger.event_type == EventType.FLATFIELD or (
                self.simulation
                and np.median(np.sum(event.r1.tel[self.tel_id].waveform[0], axis=1))
                > self.processor.minimum_hg_charge_median
                and np.std(np.sum(event.r1.tel[self.tel_id].waveform[1], axis=1))
                < self.processor.maximum_lg_charge_std
            ):

                new_ff = self.processor.flatfield.calculate_relative_gain(event)

            # write pedestal results when enough statistics or end of file
            if new_ped or end_of_file:

                # update the monitoring container with the last statistics
                if end_of_file:
                    self.processor.pedestal.store_results(event)

                # write the event
                self.log.debug(
                    f"Write pedestal data at event n. {count+1}, id {event.index.event_id} "
                    f"stat = {ped_data.n_events} events"
                )

                # write on file
                self.writer.write("pedestal", ped_data)
                new_ped = False

            # write flatfield results when enough statistics (also for pedestals) or end of file
            if (new_ff and ped_data.n_events > 0) or end_of_file:

                # update the monitoring container with the last statistics
                if end_of_file:
                    self.processor.flatfield.store_results(event)

                self.log.debug(
                    f"Write flatfield data at event n. {count+1}, id {event.index.event_id} "
                    f"stat = {ff_data.n_events} events"
                )

                # write on file
                self.writer.write("flatfield", ff_data)
                new_ff = False

                # calculate calibration coefficients
                self.processor.calculate_calibration_coefficients(event)

                # write calib and pixel status
                self.log.debug(f"Write pixel_status data")
                self.writer.write("pixel_status", status_data)

                self.log.debug(f"Write calibration data")
                self.writer.write("calibration", calib_data)
                if self.one_event:
                    break

                # self.writer.write('mon', event.mon.tel[self.tel_id])

    def finish(self):
        Provenance().add_output_file(
            self.output,
            role='mon.tel.calibration'
        )
        self.writer.close()

    @staticmethod
    def initialize_pixel_status(mon_camera_container, shape):
        """
        Initialize the pixel status container in the case of
        simulation events (this should be done in the event source, but
        added here for the moment)
        """
        # initialize the container
        status_container = PixelStatusContainer()
        status_container.hardware_failing_pixels = np.zeros((shape[0], shape[1]), dtype=bool)
        status_container.pedestal_failing_pixels = np.zeros((shape[0], shape[1]), dtype=bool)
        status_container.flatfield_failing_pixels = np.zeros((shape[0], shape[1]), dtype=bool)

        mon_camera_container.pixel_status = status_container


def main():
    exe = CalibrationHDF5Writer()
    exe.run()


if __name__ == "__main__":
    main()
