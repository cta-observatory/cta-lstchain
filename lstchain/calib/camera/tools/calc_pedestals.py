"""
Extract pedestals from pedestal events
"""
from traitlets import Dict, List, Unicode

from ctapipe.core import Provenance
from ctapipe.io import HDF5TableWriter
from ctapipe.core import Tool
from ctapipe.io import EventSourceFactory
from ctapipe.image import ChargeExtractorFactory

from lstchain.calib.camera.pedestals import PedestalFactory
from lstchain.io.containers import PedestalContainer


class PedestalHDF5Writer(Tool):
    name = "PedestalHDF5Writer"
    description = "Generate a HDF5 file with pedestal values"

    output_file = Unicode(
        'pedestal.hdf5',
        help='Name of the output file'
    ).tag(config=True)

    aliases = Dict(dict(
        input_file='EventSourceFactory.input_url',
        max_events='EventSourceFactory.max_events',
        allowed_tels='EventSourceFactory.allowed_tels',
        charge_extractor='ChargeExtractorFactory.product',
        window_width='ChargeExtractorFactory.window_width',
        t0='ChargeExtractorFactory.t0',
        window_shift='ChargeExtractorFactory.window_shift',
        sig_amp_cut_HG='ChargeExtractorFactory.sig_amp_cut_HG',
        sig_amp_cut_LG='ChargeExtractorFactory.sig_amp_cut_LG',
        lwt='ChargeExtractorFactory.lwt',
        generator='PedestalFactory.product',
        tel_id='PedestalFactory.tel_id',
        sample_duration='PedestalFactory.sample_duration',
        sample_size='PedestalFactory.sample_size',
        n_channels='PedestalFactory.n_channels',
    ))

    classes = List([EventSourceFactory,
                    ChargeExtractorFactory,
                    PedestalFactory,
                    PedestalContainer,
                    HDF5TableWriter
                    ])

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
        #self.container = PedestalContainer()
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
