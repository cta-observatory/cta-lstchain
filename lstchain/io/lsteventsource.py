# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
EventSource for LSTCam protobuf-fits.fz-files.

Needs protozfits v1.4.2 from github.com/cta-sst-1m/protozfitsreader
"""
import glob
import numpy as np
from astropy import units as u
import struct

from ctapipe.core import Provenance
from ctapipe.instrument import TelescopeDescription, SubarrayDescription, \
    CameraGeometry, OpticsDescription
from ctapipe.io.eventsource import EventSource

from lstchain.io.containers import LSTDataContainer, PixelStatusContainer


__all__ = ['LSTEventSource']


class LSTEventSource(EventSource):
    """EventSource for LST r0 data."""

    def __init__(self, config=None, tool=None, **kwargs):
        """
        Constructor
        Parameters
        ----------
        config: traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            Set to None if no configuration to pass.
        tool: ctapipe.core.Tool
            Tool executable that is calling this component.
            Passes the correct logger to the component.
            Set to None if no Tool to pass.
        kwargs: dict
            Additional parameters to be passed.
            NOTE: The file mask of the data to read can be passed with
            the 'input_url' parameter.
        """

        # EventSource can not handle file wild cards as input_url
        # To overcome this we substitute the input_url with first file matching
        # the specified file mask (copied from  MAGICEventSourceROOT).

        if 'input_url' in kwargs.keys():
            self.file_list = glob.glob(kwargs['input_url'])
            self.file_list.sort()
            kwargs['input_url'] = self.file_list[0]
            super().__init__(config=config, tool=tool, **kwargs)
        else:
            super().__init__(config=config, tool=tool, **kwargs)
            self.file_list = [self.input_url]

        self.multi_file = MultiFiles(self.file_list)

        self.camera_config = self.multi_file.camera_config
        self.log.info(
            "Read {} input files".format(
                self.multi_file.num_inputs()
            )
        )

    def rewind(self):
        self.multi_file.rewind()

    def _generator(self):

        # container for LST data
        self.data = LSTDataContainer()
        self.data.meta['input_url'] = self.input_url
        self.data.meta['max_events'] = self.max_events

        # fill LST data from the CameraConfig table
        self.fill_lst_service_container_from_zfile()

        # Instrument information
        for tel_id in self.data.lst.tels_with_data:

            assert (tel_id == 0 or tel_id == 1) # only LST1 (for the moment id = 0)

            # optics info from standard optics.fits.gz file
            optics = OpticsDescription.from_name("LST")
            optics.tel_subtype = ''  # to correct bug in reading

            # camera info from LSTCam-[geometry_version].camgeom.fits.gz file
            geometry_version = 2
            camera = CameraGeometry.from_name("LSTCam", geometry_version)

            tel_descr = TelescopeDescription(optics, camera)

            self.n_camera_pixels = tel_descr.camera.n_pixels
            tels = {tel_id: tel_descr}

            # LSTs telescope position taken from MC from the moment
            tel_pos = {tel_id: [50., 50., 16] * u.m}

        subarray = SubarrayDescription("LST1 subarray")
        subarray.tels = tels
        subarray.positions = tel_pos

        self.data.inst.subarray = subarray

        # loop on events
        for count, event in enumerate(self.multi_file):

            self.data.count = count

            # fill specific LST event data
            self.fill_lst_event_container_from_zfile(event)

            # fill general monitoring data
            self.fill_mon_container_from_zfile(event)

            # fill general R0 data
            self.fill_r0_container_from_zfile(event)

            yield self.data

    @staticmethod
    def is_compatible(file_path):
        from astropy.io import fits
        try:
            # The file contains two tables:
            #  1: CameraConfig
            #  2: Events
            h = fits.open(file_path)[2].header
            ttypes = [
                h[x] for x in h.keys() if 'TTYPE' in x
            ]
        except OSError:
            # not even a fits file
            return False

        except IndexError:
            # A fits file of a different format
            return False

        is_protobuf_zfits_file = (
            (h['XTENSION'] == 'BINTABLE') and
            (h['EXTNAME'] == 'Events') and
            (h['ZTABLE'] is True) and
            (h['ORIGIN'] == 'CTA') and
            (h['PBFHEAD'] == 'R1.CameraEvent')
        )

        is_lst_file = 'lstcam_counters' in ttypes
        return is_protobuf_zfits_file & is_lst_file

    def fill_lst_service_container_from_zfile(self):
        """
        Fill LSTServiceContainer with specific LST service data data
        (from the CameraConfig table of zfit file)

        """

        self.data.lst.tels_with_data = [self.camera_config.telescope_id, ]
        svc_container = self.data.lst.tel[self.camera_config.telescope_id].svc

        svc_container.telescope_id = self.camera_config.telescope_id
        svc_container.cs_serial = self.camera_config.cs_serial
        svc_container.configuration_id = self.camera_config.configuration_id
        svc_container.date = self.camera_config.date
        svc_container.num_pixels = self.camera_config.num_pixels
        svc_container.num_samples = self.camera_config.num_samples
        svc_container.pixel_ids = self.camera_config.expected_pixels_id
        svc_container.data_model_version = self.camera_config.data_model_version

        svc_container.num_modules = self.camera_config.lstcam.num_modules
        svc_container.module_ids = self.camera_config.lstcam.expected_modules_id
        svc_container.idaq_version = self.camera_config.lstcam.idaq_version
        svc_container.cdhs_version = self.camera_config.lstcam.cdhs_version
        svc_container.algorithms = self.camera_config.lstcam.algorithms
        svc_container.pre_proc_algorithms = self.camera_config.lstcam.pre_proc_algorithms

    def fill_lst_event_container_from_zfile(self, event):
        """
        Fill LSTEventContainer with specific LST service data
        (from the Event table of zfit file)

        """

        event_container = self.data.lst.tel[self.camera_config.telescope_id].evt

        event_container.configuration_id = event.configuration_id
        event_container.event_id = event.event_id
        event_container.tel_event_id = event.tel_event_id
        event_container.pixel_status = event.pixel_status
        event_container.ped_id = event.ped_id
        event_container.module_status = event.lstcam.module_status
        event_container.extdevices_presence = event.lstcam.extdevices_presence

        # unpack TIB data
        rec_fmt = '=IHIBB'
        rec_unpack = struct.Struct(rec_fmt).unpack_from
        unpacked_tib = rec_unpack(event.lstcam.tib_data)
        event_container.tib_event_counter = unpacked_tib[0]
        event_container.tib_pps_counter = unpacked_tib[1]
        event_container.tib_tenMHz_counter = unpacked_tib[2]
        event_container.tib_stereo_pattern = unpacked_tib[3]
        event_container.tib_masked_trigger = unpacked_tib[4]
        #event_container.cdts_data = event.lstcam.cdts_data
        event_container.swat_data = event.lstcam.swat_data

        # unpack CDTS data
        rec_fmt = '=IIIQQBBB'
        rec_unpack = struct.Struct(rec_fmt).unpack_from
        unpacked_cdts = rec_unpack(event.lstcam.cdts_data)
        event_container.ucts_event_counter = unpacked_cdts[0]
        event_container.ucts_pps_counter = unpacked_cdts[1]
        event_container.ucts_clock_counter = unpacked_cdts[2]
        event_container.ucts_timestamp = unpacked_cdts[3]
        event_container.ucts_camera_timestamp = unpacked_cdts[4]
        event_container.ucts_trigger_type = unpacked_cdts[5]
        event_container.ucts_white_rabbit_status = unpacked_cdts[6]

        # unpack Dragon counters
        rec_fmt = '=HIIIQ'
        rec_len = struct.calcsize(rec_fmt)
        rec_unpack = struct.Struct(rec_fmt).unpack_from

        event_container.pps_counter = np.zeros(self.camera_config.lstcam.num_modules)
        event_container.tenMHz_counter = np.zeros(self.camera_config.lstcam.num_modules)
        event_container.event_counter = np.zeros(self.camera_config.lstcam.num_modules)
        event_container.trigger_counter = np.zeros(self.camera_config.lstcam.num_modules)
        event_container.local_clock_counter = np.zeros(self.camera_config.lstcam.num_modules)
        for mod in range(self.camera_config.lstcam.num_modules):

            words=event.lstcam.counters[mod*rec_len:(mod+1)*rec_len]
            unpacked_counter = rec_unpack(words)
            event_container.pps_counter[mod] = unpacked_counter[0]
            event_container.tenMHz_counter[mod] = unpacked_counter[1]
            event_container.event_counter[mod] = unpacked_counter[2]
            event_container.trigger_counter[mod] = unpacked_counter[3]
            event_container.local_clock_counter[mod] = unpacked_counter[4]

        event_container.chips_flags = event.lstcam.chips_flags
        event_container.first_capacitor_id = event.lstcam.first_capacitor_id
        event_container.drs_tag_status = event.lstcam.drs_tag_status
        event_container.drs_tag = event.lstcam.drs_tag

    def fill_r0_camera_container_from_zfile(self, r0_container, event):
        """
        Fill with R0CameraContainer

        """

        r0_container.num_samples = self.camera_config.num_samples
        #container.trigger_time = event.trigger_time_s

        # temporary patch to have an event time set
        r0_container.trigger_time = (
            self.data.lst.tel[self.camera_config.telescope_id].evt.tib_pps_counter +
            self.data.lst.tel[self.camera_config.telescope_id].evt.tib_tenMHz_counter * 10**(-7))

        r0_container.trigger_type = event.trigger_type

        # verify the number of gains
        if event.waveform.shape[0] == (self.camera_config.num_pixels *
                                       r0_container.num_samples):
            n_gains = 1
        elif event.waveform.shape[0] == (self.camera_config.num_pixels *
                                         r0_container.num_samples * 2):
            n_gains = 2
        else:
            raise ValueError("Waveform matrix dimension not supported: "
                             "N_chan x N_pix x N_samples= '{}'"
                             .format(event.waveform.shape[0]))

        reshaped_waveform = np.array(
            event.waveform
        ).reshape(
            n_gains,
            self.camera_config.num_pixels,
            r0_container.num_samples
        )

        # initialize the waveform container to zero
        r0_container.waveform = np.zeros([n_gains, self.n_camera_pixels,
                                       r0_container.num_samples])

        # re-order the waveform following the expected_pixels_id values
        # (rank = pixel id)
        r0_container.waveform[:, self.camera_config.expected_pixels_id, :] =\
            reshaped_waveform

    def fill_r0_container_from_zfile(self, event):
        """
        Fill with R0Container

        """
        container = self.data.r0

        container.obs_id = -1
        container.event_id = event.event_id

        container.tels_with_data = [self.camera_config.telescope_id, ]
        r0_camera_container = container.tel[self.camera_config.telescope_id]
        self.fill_r0_camera_container_from_zfile(
            r0_camera_container,
            event
        )

    def fill_mon_container_from_zfile(self, event):
        """
        Fill with MonitoringContainer.
        For the moment, initialize only the PixelStatusContainer

        """
        container = self.data.mon
        container.tels_with_data = [self.camera_config.telescope_id, ]
        mon_camera_container = container.tel[self.camera_config.telescope_id]

        # reorder the array
        pixel_status = np.zeros([self.n_camera_pixels])
        pixel_status[self.camera_config.expected_pixels_id]  = \
            event.pixel_status

        # initalize the container
        status_container = PixelStatusContainer()
        status_container.hardware_mask = pixel_status > 0
        mon_camera_container.pixel_status = status_container


class MultiFiles:

    """
    This class open all the files in file_list and read the events following
    the event_id order
    """

    def __init__(self, file_list):

        self._file = {}
        self._events = {}
        self._events_table = {}
        self._camera_config = {}
        self.camera_config = None

        paths = []
        for file_name in file_list:
            paths.append(file_name)
            Provenance().add_input_file(file_name, role='r0.sub.evt')

        # open the files and get the first fits Tables
        from protozfits import File

        for path in paths:

            try:
                self._file[path] = File(path)
                self._events_table[path] = File(path).Events
                self._events[path] = next(self._file[path].Events)

                # verify where the CameraConfig is present
                if 'CameraConfig' in self._file[path].__dict__.keys():
                    self._camera_config[path] = next(self._file[path].CameraConfig)

                # for the moment it takes the first CameraConfig it finds (to be changed)
                    if(self.camera_config is None):
                        self.camera_config = self._camera_config[path]

            except StopIteration:
                pass

        # verify that somewhere the CameraConfing is present
        assert self.camera_config

    def __iter__(self):
        return self

    def __next__(self):
        return self.next_event()

    def next_event(self):
        # check for the minimal event id
        if not self._events:
            raise StopIteration

        min_path = min(
            self._events.items(),
            key=lambda item: item[1].event_id,
        )[0]

        # return the minimal event id
        next_event = self._events[min_path]
        try:
            self._events[min_path] = next(self._file[min_path].Events)
        except StopIteration:
            del self._events[min_path]

        return next_event

    def __len__(self):
        total_length = sum(
            len(table)
            for table in self._events_table.values()
        )
        return total_length

    def rewind(self):
        for name, file in self._file.items():
            file.Events.protobuf_i_fits.rewind()

    def num_inputs(self):
        return len(self._file)
