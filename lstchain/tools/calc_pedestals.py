"""
Extract pedestals from pedestal events
"""
from traitlets import Dict, List, Unicode
import ctapipe.utils.tools as tool_utils
from ctapipe.core import Provenance
from ctapipe.io import HDF5TableWriter
from ctapipe.core import Tool
from ctapipe.io import EventSource
from ctapipe.image import ChargeExtractor

from lstchain.calib.camera.pedestals import PedestalCalculator
from lstchain.io.containers import PedestalContainer


class PedestalHDF5Writer(Tool):
    name = "PedestalHDF5Writer"
    description = "Generate a HDF5 file with pedestal values"

    output_file = Unicode(
        'pedestal.hdf5',
        help='Name of the output file'
    ).tag(config=True)

    aliases = Dict(dict(
        pedestal_calculator='PedestalHDF5Writer.calculator_product',
        cleaner='PedestaldHDF5Writer.cleaner_product',
        input_file='EventSource.input_url',
        max_events='EventSource.max_events',
        window_width='WindowIntegrator.window_width',
        window_shift='WindowIntegrator.window_shift',
        sig_amp_cut_HG='PeakFindingIntegrator.sig_amp_cut_HG',
        sig_amp_cut_LG='PeakFindingIntegrator.sig_amp_cut_LG',
        t0='SimpleIntegrator.t0',
        lwt='NeighbourPeakIntegrator.lwt',
        baseline_start='BaselineWaveformCleaner.baseline_start',
        baseline_end='BaselineWaveformCleaner.baseline_end',
        charge_extractor='.FlatFieldCalculator.extractor_product',
        tel_id='FlatFieldCalculator.tel_id',
        sample_duration='FlatFieldCalculator.sample_duration',
        sample_size='FlatFieldCalculator.sample_size',
        n_channels='FlatFieldCalculator.n_channels',
    ))

    classes = List([EventSource,
                    ChargeExtractor,
                    PedestalCalculator,
                    PedestalContainer,
                    HDF5TableWriter
                    ])+ tool_utils.classes_with_traits(PedestalCalculator)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.eventsource = None
        self.pedestal = None
        self.container = None
        self.writer = None

    def setup(self):
        kwargs = dict(config=self.config, tool=self)
        self.eventsource = EventSourceFactory.produce(**kwargs)
        self.pedestal = PedestalFactory.produce(**kwargs)

        self.writer = HDF5TableWriter(
            filename=self.output_file, group_name='pedestals', overwrite=True
        )

    def start(self):
        '''Pedestal calculator'''

        for count, event in enumerate(self.eventsource):

            if __name__ == '__main__':
                ped_data = self.pedestal.calculate_pedestals(event)

            if ped_data:

                table_name = 'tel_' + str(self.pedestal.tel_id)

                self.log.info("write event in table: /pedestal/%s",
                              table_name)
                self.writer.write(table_name, ped_data)

    def finish(self):
        Provenance().add_output_file(
            self.output_file,
            role='mon.tel.pedestal'
        )
        self.writer.close()


def main():
    exe = PedestalHDF5Writer()
    exe.run()


if __name__ == '__main__':
    main()
