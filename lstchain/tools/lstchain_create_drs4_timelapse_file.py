import numpy as np
from tqdm import tqdm
from ctapipe_io_lst import LSTEventSource
from ctapipe_io_lst.calibration import get_spike_A_positions
from ctapipe_io_lst.constants import N_GAINS, N_PIXELS, N_CAPACITORS_PIXEL, N_SAMPLES, N_PIXELS_MODULE, CLOCK_FREQUENCY_KHZ
import numba
import tables

from ctapipe.io.hdf5tableio import DEFAULT_FILTERS
from ctapipe.core import Tool
from ctapipe.core.traits import Path, Integer

from ..statistics import OnlineStats


N_BINS_DT = 25
LOG_DT_MIN_MS = -3.0
LOG_DT_MAX_MS = 2.0


@numba.njit(cache=True, inline='always')
def bin_index(x, low, high, n_bins):
    '''Calculates the linear bin index of x for the given bin index
    idx 0 is underflow, index n_bins is overflow
    '''
    idx = 1 + int(n_bins * (x - low) / (high - low))
    idx = min(idx, n_bins + 1)
    idx = max(idx, 0)
    return idx


@numba.njit(cache=True, inline='always')
def flat_index(gain, pixel, cap, dt):
    '''Flattened index of (gain, pixel, cap, dt) in the flat stats array'''
    log_dt = np.log10(dt)
    dt_bin = bin_index(log_dt, LOG_DT_MIN_MS, LOG_DT_MAX_MS, N_BINS_DT)
    return (
        N_PIXELS * N_CAPACITORS_PIXEL * N_BINS_DT * gain
        + N_CAPACITORS_PIXEL * N_BINS_DT * pixel
        + N_BINS_DT * cap
        + dt_bin
    )


@numba.njit(cache=True)
def fill_stats(
    waveform,
    first_cap,
    last_first_cap,
    last_readout_time,
    local_clock_counter,
    expected_pixels_id,
    dt_stats,
    skip_samples_front,
    skip_samples_end,
):
    n_modules = len(expected_pixels_id) // N_PIXELS_MODULE
    for gain in range(N_GAINS):
        for module in range(n_modules):
            time_now = local_clock_counter[module]
            for pixel_in_module in range(N_PIXELS_MODULE):
                pixel_index = module * N_PIXELS_MODULE + pixel_in_module
                pixel = expected_pixels_id[pixel_index]

                fc = first_cap[gain, pixel]
                last_fc = last_first_cap[gain, pixel]
                spike_positions = get_spike_A_positions(fc, last_fc)

                for sample in range(skip_samples_front, N_SAMPLES - skip_samples_end):
                    cap = (fc + sample) % N_CAPACITORS_PIXEL

                    last_read = last_readout_time[gain, pixel, cap]
                    # ignore samples where we don't have a last readout time yet
                    if last_read == 0:
                        continue

                    dt = (time_now - last_read) / CLOCK_FREQUENCY_KHZ
                    idx = flat_index(gain, pixel, cap, dt)

                    # ignore spikes
                    if sample in spike_positions or (sample - 1) in spike_positions or (sample - 2) in spike_positions:
                        continue

                    dt_stats.add_value(idx, waveform[gain, pixel, sample])


class DRS4Timelapse(Tool):
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

        n_stats = N_GAINS * N_PIXELS * N_CAPACITORS_PIXEL * N_BINS_DT
        self.dt_stats = OnlineStats(n_stats)

    def start(self):
        tel_id = self.source.tel_id
        expected_pixels_id = self.source.camera_config.expected_pixels_id

        total = len(self.source.multi_file)
        if self.source.max_events is not None:
            total = min(self.source.max_events, total)

        for event in tqdm(self.source, total=total):
            fill_stats(
                waveform=event.r1.tel[tel_id].waveform,
                first_cap=self.source.r0_r1_calibrator.first_cap[tel_id],
                last_first_cap=self.source.r0_r1_calibrator.first_cap_old[tel_id],
                last_readout_time=self.source.r0_r1_calibrator.last_readout_time[tel_id],
                local_clock_counter=event.lst.tel[tel_id].evt.local_clock_counter,
                expected_pixels_id=expected_pixels_id,
                dt_stats=self.dt_stats,
                skip_samples_front=self.skip_samples_front,
                skip_samples_end=self.skip_samples_end,
            )

    def finish(self):
        self.log.info('Writing output to %s', self.output_path)
        with tables.open_file(self.output_path, 'w') as f:

            g = f.create_group('/', 'dt')
            f.root.attrs['dt_n_bins'] = N_BINS_DT
            f.root.attrs['dt_min_log10'] = LOG_DT_MIN_MS
            f.root.attrs['dt_max_log10'] = LOG_DT_MAX_MS

            for attr in ('counts', 'mean', 'std'):
                array = getattr(self.dt_stats, attr).reshape((N_GAINS, N_PIXELS, N_CAPACITORS_PIXEL))

                # store mean / std as int32 scaled by 100
                # aka fixed precision with two decimal digits
                if attr in ('mean', 'std'):
                    array *= 100
                array = array.astype(np.int32)

                f.create_carray(g, attr, obj=array, filters=DEFAULT_FILTERS)


def main():
    tool = DRS4Timelapse()
    tool.run()


if __name__ == '__main__':
    main()
