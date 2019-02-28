"""
Extract flat field coefficients from flasher data files.
"""
from traitlets import Dict, List, Unicode
import ctapipe.utils.tools as tool_utils
from ctapipe.core import Provenance
from ctapipe.io import HDF5TableWriter
from ctapipe.core import Tool
from ctapipe.io import event_source, EventSource

from ctapipe.image import ChargeExtractor, WaveformCleaner

# to be changed for ctapipe
from lstchain.calib.camera import FlatFieldCalculator
from lstchain.io.containers import FlatFieldContainer


class FlatFieldHDF5Writer(Tool):
    name = "FlatFieldHDF5Writer"
    description = "Generate a HDF5 file with flat field coefficients"

    output_file = Unicode(
        'flatfield.hdf5',
        help='Name of the output flat field file file'
    ).tag(config=True)

    aliases = Dict(dict(
        input_file='EventSource.input_url',
        max_events='EventSource.max_events',
        charge_extractor='ChargeExtractor.product',
        window_width='WindowIntegrator.window_width',
        window_shift='WindowIntegrator.window_shift',
        sig_amp_cut_HG='PeakFindingIntegrator.sig_amp_cut_HG',
        sig_amp_cut_LG='PeakFindingIntegrator.sig_amp_cut_LG',
        t0='SimpleIntegrator.t0',
        lwt='NeighbourPeakIntegrator.lwt',
        cleaner='WaveformCleaner.product',
        baseline_start='WaveformCleaner.baseline_start',
        baseline_end='WaveformCleaner.baseline_end',
        generator='FlatFieldCalculator.product',
        tel_id='FlatFieldCalculator.tel_id',
        sample_duration='FlatFieldCalculator.sample_duration',
        sample_size='FlatFieldCalculator.sample_size',
        n_channels='FlatFieldCalculator.n_channels',
    ))

    classes = List([EventSource,
                    ChargeExtractor,
                    WaveformCleaner,
                    FlatFieldCalculator,
                    FlatFieldContainer,
                    HDF5TableWriter
                    ]+ tool_utils.classes_with_traits(WaveformCleaner)
	                 + tool_utils.classes_with_traits(ChargeExtractor)
                   )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.eventsource = None
        self.flatfield = None
        self.container = None
        self.writer = None
        self.cleaner = None

    def setup(self):
        kwargs = dict(config=self.config, tool=self)
        self.eventsource = EventSource.from_config(**kwargs)
        self.flatfield = FlatFieldCalculator(**kwargs)

        self.cleaner = WaveformCleaner.from_name(
            self.cleaner_product,
            **kwargs
        )

        #self.container = FlatFieldContainer()
        self.writer = HDF5TableWriter(
            filename=self.output_file, group_name='flatfield', overwrite=True
        )

    def start(self):
        '''Flat field coefficient calculator'''

        for count, event in enumerate(self.eventsource):

            ff_data = self.flatfield.calculate_relative_gain(event)

            if ff_data:

                table_name = 'tel_' + str(self.flatfield.tel_id)

                self.log.info("write event in table: /flatfield/%s",
                              table_name)
                self.writer.write(table_name, ff_data)

    def finish(self):
        Provenance().add_output_file(
            self.output_file,
            role='mon.tel.flatfield'
        )
        self.writer.close()


def main():
    exe = FlatFieldHDF5Writer()
    exe.run()


if __name__ == '__main__':
    main()
