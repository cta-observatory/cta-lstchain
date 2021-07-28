import numpy as np
from tqdm import tqdm
from ctapipe_io_lst import LSTEventSource
from ctapipe_io_lst.calibration import get_spike_A_positions
from ctapipe_io_lst.constants import N_GAINS, N_PIXELS, N_CAPACITORS_PIXEL, N_SAMPLES
import numba
import tables

from ctapipe.io.hdf5tableio import DEFAULT_FILTERS
from ctapipe.core import Tool
from ctapipe.core.traits import Path, Integer

from ..statistics import OnlineStats


@numba.njit(cache=True, inline='always')
def flat_index(gain, pixel, cap):
    return N_PIXELS * N_CAPACITORS_PIXEL * gain + N_CAPACITORS_PIXEL * pixel + cap


@numba.njit(cache=True)
def fill_stats(
    waveform,
    first_cap,
    last_first_cap,
    last_readout_time,
    baseline_stats,
    spike1_stats,
    spike2_stats,
    spike3_stats,
    skip_samples_front,
    skip_samples_end,
):
    for gain in range(N_GAINS):
        for pixel in range(N_PIXELS):
            fc = first_cap[gain, pixel]
            last_fc = last_first_cap[gain, pixel]
            spike_positions = get_spike_A_positions(fc, last_fc)

            for sample in range(skip_samples_front, N_SAMPLES - skip_samples_end):
                cap = (fc + sample) % N_CAPACITORS_PIXEL

                # ignore samples where we don't have a last readout time yet
                if last_readout_time[gain, pixel, cap] == 0:
                    continue

                idx = flat_index(gain, pixel, cap)

                # if sample in spike_positions or (sample - 1) in spike_positions or (sample - 2) in spike_positions:
                if sample in spike_positions:
                    spike1_stats.add_value(idx, waveform[gain, pixel, sample])
                elif sample - 1 in spike_positions:
                    spike2_stats.add_value(idx, waveform[gain, pixel, sample])
                elif sample - 2 in spike_positions:
                    spike3_stats.add_value(idx, waveform[gain, pixel, sample])
                else:
                    baseline_stats.add_value(idx, waveform[gain, pixel, sample])


class DRS4PedestalAndSpikeHeight(Tool):
    name = 'lstchain_create_drs4_pedestal_file'

    output_path = Path(directory_ok=False).tag(config=True)
    skip_samples_front = Integer(default_value=10).tag(config=True)
    skip_samples_end = Integer(default_value=1).tag(config=True)

    aliases = {
        ('i', 'input'): 'LSTEventSource.input_url',
        ('o', 'output'): 'DRS4PedestalAndSpikeHeight.output_path',
        ('m', 'max-events'): 'LSTEventSource.max_events',
    }

    classes = [LSTEventSource]

    def setup(self):
        self.source = LSTEventSource(
            parent=self,
            pointing_information=False,
            trigger_information=False,
        )

        # set some config options, these are necessary for this tool,
        # so we set them here and not via the config system
        self.source.r0_r1_calibrator.r1_sample_start = 0
        self.source.r0_r1_calibrator.r1_sample_end = N_SAMPLES

        self.source.r0_r1_calibrator.offset = 0
        self.source.r0_r1_calibrator.apply_spike_correction = False
        self.source.r0_r1_calibrator.apply_timelapse_correction = True
        self.source.r0_r1_calibrator.apply_drs4_pedestal_correction = False

        n_stats = N_GAINS * N_PIXELS * N_CAPACITORS_PIXEL
        self.baseline_stats = OnlineStats(n_stats)
        self.spike1_stats = OnlineStats(n_stats)
        self.spike2_stats = OnlineStats(n_stats)
        self.spike3_stats = OnlineStats(n_stats)

    def start(self):
        tel_id = self.source.tel_id

        total = len(self.source.multi_file)
        if self.source.max_events is not None:
            total = min(self.source.max_events, total)

        for event in tqdm(self.source, total=total):
            fill_stats(
                event.r1.tel[tel_id].waveform,
                self.source.r0_r1_calibrator.first_cap[tel_id],
                self.source.r0_r1_calibrator.first_cap_old[tel_id],
                self.source.r0_r1_calibrator.last_readout_time[tel_id],
                self.baseline_stats,
                self.spike1_stats,
                self.spike2_stats,
                self.spike3_stats,
                skip_samples_front=self.skip_samples_front,
                skip_samples_end=self.skip_samples_end,
            )

    def finish(self):
        self.log.info('Writing output to %s', self.output_path)
        with tables.open_file(self.output_path, 'w') as f:
            stats = {
                'baseline': self.baseline_stats,
                'spike1': self.spike1_stats,
                'spike2': self.spike2_stats,
                'spike3': self.spike3_stats,
            }
            mean_baseline = self.baseline_stats.mean

            for name, stat in stats.items():
                g = f.create_group('/', name)

                for attr in ('counts', 'mean', 'std'):
                    array = getattr(stat, attr).reshape((N_GAINS, N_PIXELS, N_CAPACITORS_PIXEL))

                    # store mean / std as int32 scaled by 100
                    # aka fixed precision with two decimal digits
                    if attr in ('mean', 'std'):
                        array *= 100
                    array = array.astype(np.int32)

                    f.create_carray(g, attr, obj=array, filters=DEFAULT_FILTERS)

                if name.startswith('spike'):
                    mask = stat.counts > 0
                    n_spikes = stat.counts[mask].sum()
                    spike_height = stat.mean[mask] - mean_baseline[mask]
                    mean_height = np.average(spike_height, weights=stat.counts[mask])
                    self.log.info(
                        "Found %d %s, mean height = %.2f",
                        n_spikes, name, mean_height,
                    )


def main():
    tool = DRS4PedestalAndSpikeHeight()
    tool.run()


if __name__ == '__main__':
    main()
