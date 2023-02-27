"""
Extract flat field coefficients from flasher data files.
"""
import numpy as np
from pathlib import Path 
from traitlets import Unicode,  Bool
from tqdm.auto import tqdm
from ctapipe.instrument.subarray import SubarrayDescription
from ctapipe.core import Provenance, traits
from ctapipe.io import HDF5TableWriter
from ctapipe.core import Tool
from ctapipe.io import EventSource
from ctapipe.containers import PixelStatusContainer
from lstchain.calib.camera.calibration_calculator import CalibrationCalculator
from lstchain.io import add_config_metadata, add_global_metadata, global_metadata, write_metadata
from ctapipe.containers import EventType
from lstchain.io.calibration import read_calibration_file


__all__ = [
    'CalibrationHDF5Writer'
]

class CatBCalibrationHDF5Writer(Tool):
    """
     Tool that generates a HDF5 file with CatB camera calibration coefficients.
     This is just an example on how the monitoring containers can be filled using
     the calibration Components in calib/camera.
     This example is based on an input file with pedestal and flat-field events

     For getting help run:
     python calc_camera_calibration.py --help

     """

    name = "CatBCalibrationHDF5Writer"
    description = "Generate a HDF5 file with camera calibration coefficients"

    one_event = Bool(
        False,
        help='Stop after first calibration event'
    ).tag(config=True)

    output_file = Unicode(
        'calibration.hdf5',
        help='Name of the output file'
    ).tag(config=True)

    input_path = Unicode(
        '.',
        help='Path of input file'
    ).tag(config=True)

    input_file_pattern = Unicode(
        'interleaved_LST-1.Run*.*.h5',
        help='Pattern for searching the input files with interleaved events to be processed'
    ).tag(config=True)

    cat_A_calibration_file = Unicode(
        'catA_calibration.hdf5',
        help='Name of category A calibration file'
    ).tag(config=True)

    calibration_product = traits.create_class_enum_trait(
       CalibrationCalculator,
        default_value='LSTCalibrationCalculator'
    )

    aliases = {
        ("i", "input_file"): 'EventSource.input_url',
        ("m", "max_events"): 'EventSource.max_events',
        ("o", "output_file"): 'CatBCalibrationHDF5Writer.output_file',
        ("k", "cat_A_calibration_file"): 'CatBCalibrationHDF5Writer.cat_A_calibration_file',  
        ("s", "systematics_file"): "LSTCalibrationCalculator.systematic_correction_path",
        ("input_file_pattern"): 'CatBCalibrationHDF5Writer.input_file_pattern',
        ("input_path"): 'CatBCalibrationHDF5Writer.input_path',      

    }

    flags = {
        **traits.flag(
            "flatfield-heuristic",
            "LSTEventSource.use_flatfield_heuristic",
            "Use flatfield heuristic",
            "Do not use flatfield heuristic",
        )
    }

    classes = (
        [EventSource, CalibrationCalculator]
        + traits.classes_with_traits(CalibrationCalculator)
        + traits.classes_with_traits(EventSource)
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        """
        Tool that generates a HDF5 file with camera calibration coefficients.
        Input file must contain interleaved pedestal and flat-field events

        For getting help run:
            lstchain_create_catA_calibration_file --help
        """
        self.eventsource = None
        self.processor = None
        self.writer = None
        
    def setup(self):

        self.log.debug("Opening file")
        #self.eventsource = EventSource.from_config(parent=self)

        self.input_paths = sorted(Path(f"{self.input_path}").rglob(f"{self.input_file_pattern}"))
        
        self.subarray = SubarrayDescription.from_hdf(self.input_paths[0])

        self.processor = CalibrationCalculator.from_name(
            self.calibration_product,
            parent=self,
            subarray = self.subarray
        )

        tel_id = self.processor.tel_id

        group_name = 'tel_' + str(tel_id)

        self.log.debug(f"Open output file {self.output_file}")

        self.writer = HDF5TableWriter(
            filename=self.output_file, group_name=group_name, overwrite=True
        )

        # initialize the monitoring data
        self.monitoring_data = read_calibration_file(self.cat_A_calibration_file)

        # extract flat-fielding factor 
        mask= self.monitoring_data.calibration.unusable_pixels
        masked_npe = np.ma.array(self.monitoring_data.calibration.n_pe,mask=mask)
        npe_signal_median = np.ma.median(masked_npe, axis=1)
        self.inverse_FF_factor = self.monitoring_data.calibration.n_pe/npe_signal_median[:, np.newaxis]
        
    def start(self):
        '''Calibration coefficient calculator'''

        metadata = global_metadata()
        write_metadata(metadata, self.output_file)

        tel_id = self.processor.tel_id
        new_ped = False
        new_ff = False
    
        scale = np.array([1.088,1.004])
        self.log.debug("Start loop")
        for path in self.input_paths:                    
            self.log.debug(f"read {path}")
            with EventSource(path,parent=self) as eventsource:
               
                for count, event in enumerate(tqdm(eventsource)):
                    
                    # initialize the event monitoring data
                    event.mon.tel[tel_id] = self.monitoring_data

                    # unscale the R1 waveform for the flat-fielding factor 
                    #event.r1.tel[tel_id].waveform = event.r1.tel[tel_id].waveform * self.inverse_FF_factor[:,:,np.newaxis]

                    # unscale the R1 waveform window integration scaling factor
                    #event.r1.tel[tel_id].waveform = event.r1.tel[tel_id].waveform / scale[:,np.newaxis,np.newaxis]

                    # save the config, to be retrieved as data.meta['config']
                    if count == 0:
                        
                        ped_data = event.mon.tel[tel_id].pedestal
                        add_config_metadata(ped_data, self.config)
                        add_global_metadata(ped_data, metadata)

                        ff_data = event.mon.tel[tel_id].flatfield
                        add_config_metadata(ff_data, self.config)
                        add_global_metadata(ff_data, metadata)

                        status_data = event.mon.tel[tel_id].pixel_status
                        add_config_metadata(status_data, self.config)
                        add_global_metadata(status_data, metadata)

                        calib_data = event.mon.tel[tel_id].calibration
                        add_config_metadata(calib_data, self.config)
                        add_global_metadata(calib_data, metadata)
        
                    # if pedestal event
                    if self._is_pedestal(event, tel_id):
                    
                        if self.processor.pedestal.calculate_pedestals(event):
                            new_ped = True
                            count_ped = count+1

                    # if flat-field event
                    elif self._is_flatfield(event, tel_id):

                        if self.processor.flatfield.calculate_relative_gain(event):
                            new_ff = True
                            count_ff = count+1
                        
                    # write flatfield results when enough statistics (also for pedestals) 
                    if (new_ff and new_ped):
                        self.log.debug(f"Write calibration at event n. {count+1}, event id {event.index.event_id} ")
                                        
                        self.log.debug(f"Ready flatfield data at event n. {count_ff} "
                                        f"stat = {ff_data.n_events} events")

                        # write on file
                        self.writer.write('flatfield', ff_data)

                        self.log.debug(f"Ready pedestal data at event n. {count_ped} "
                                        f"stat = {ped_data.n_events} events")

                        # write only pedestal data used for calibration                                 
                        self.writer.write('pedestal', ped_data)                  

                        new_ff = False
                        new_ped = False
                        
                        # Then, calculate calibration coefficients
                        self.processor.calculate_calibration_coefficients(event)

                        # write calib and pixel status
                        self.log.debug("Write pixel_status data")
                        self.writer.write('pixel_status', status_data)

                        self.log.debug("Write calibration data")
                        self.writer.write('calibration', calib_data)
                        if self.one_event:
                            break
            
                    # store the updated version of the data
                    self.monitoring_data = event.mon.tel[tel_id]     

    def finish(self):
        Provenance().add_output_file(
            self.output_file,
            role='mon.tel.calibration'
        )
        self.writer.close()


    @staticmethod
    def _median_waveform_sum(event, tel_id):
        return np.median(np.sum(event.r1.tel[tel_id].waveform[0], axis=1))

    def _is_pedestal(self, event, tel_id=1):
        return (
            event.trigger.event_type == EventType.SKY_PEDESTAL
        )

    def _is_flatfield(self, event, tel_id):
        return (
            event.trigger.event_type == EventType.FLATFIELD
        )

def initialize_pixel_status(mon_camera_container,shape):
    """
    Initialize the pixel status container in the case of
    simulation events (this should be done in the event source, but
    added here for the moment)
    """

    # initialize the container
    status_container = PixelStatusContainer()
    status_container.hardware_failing_pixels = np.zeros((shape[0],shape[1]), dtype=bool)
    status_container.pedestal_failing_pixels = np.zeros((shape[0],shape[1]), dtype=bool)
    status_container.flatfield_failing_pixels = np.zeros((shape[0],shape[1]), dtype=bool)

    mon_camera_container.pixel_status = status_container


def main():
    exe = CatBCalibrationHDF5Writer()

    exe.run()


if __name__ == '__main__':
    main()
