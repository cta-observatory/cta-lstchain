import numpy as np
from tqdm import tqdm
from ctapipe_io_lst import LSTEventSource
from ctapipe_io_lst.calibration import get_spike_A_positions
from ctapipe_io_lst.constants import N_GAINS, N_PIXELS, N_CAPACITORS_PIXEL, N_SAMPLES, N_PIXELS_MODULE, CLOCK_FREQUENCY_KHZ
from ctapipe.io import read_table
import numba
from iminuit import Minuit
from iminuit.cost import LeastSquares

from ctapipe.io.hdf5tableio import HDF5TableWriter
from ctapipe.io.tableio import FixedPointColumnTransform
from ctapipe.core import Tool, Container, Field
from ctapipe.core.traits import Path, Integer

from ..statistics import OnlineStats


N_BINS = 100
# for under and overflow, add two bins
N_BINS_TOTAL = N_BINS + 2
LOG_DT_MIN_MS = -2
LOG_DT_MAX_MS = 2


def delta_t_correction(x, scale, exponent, offset):
    return scale * x**(-exponent) + offset


def fit_timelapse(centers, counts, mean, std):
    mean_err = std / np.sqrt(counts)
    mask = (counts > 500) & (centers > 0.05)
    cost = LeastSquares(centers[mask], mean[mask], mean_err[mask], delta_t_correction)
    m = Minuit(cost, scale=32, exponent=0.22, offset=-12)
    m.limits['scale'] = (1e-30, None)
    m.migrad()
    return m.values

class OnlineStatsContainer(Container):
    default_prefix = ""
    counts = Field(0, "Number of samples")
    mean = Field(np.nan, "mean")
    std = Field(np.nan, "standard deviation")

class TimeLapseCoefficients(Container):
    default_prefix = ""
    scale = Field(np.nan, "Number of samples")
    exponent = Field(np.nan, "mean")
    offset = Field(np.nan, "standard deviation")


def read_drs4_baseline(path, tel_id):
    table = read_table(path, f'/r1/monitoring/drs4_baseline/tel_{tel_id:03d}')
    return np.array(table[0]['baseline_mean'])


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
def flat_index(gain, pixel, dt):
    '''Flattened index of (gain, pixel, cap, dt) in the flat stats array'''
    log_dt = np.log10(dt)
    dt_bin = bin_index(log_dt, LOG_DT_MIN_MS, LOG_DT_MAX_MS, N_BINS_TOTAL)
    return (
        N_PIXELS * N_BINS_TOTAL * gain
        + N_BINS_TOTAL * pixel
        + dt_bin
    )


@numba.njit(cache=True)
def fill_stats(
    waveform,
    baseline,
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
                    idx = flat_index(gain, pixel, dt)

                    # ignore spikes
                    if sample in spike_positions or (sample - 1) in spike_positions or (sample - 2) in spike_positions:
                        continue

                    value = waveform[gain, pixel, sample] - baseline[gain, pixel, cap]
                    dt_stats.add_value(idx, value)


class DRS4Timelapse(Tool):
    name = 'lstchain_create_drs4_timelapse_file'

    output_path = Path(directory_ok=False).tag(config=True)
    skip_samples_front = Integer(default_value=10).tag(config=True)
    skip_samples_end = Integer(default_value=1).tag(config=True)
    drs4_baseline_path = Path(directory_ok=False).tag(config=True)

    aliases = {
        ('i', 'input'): 'LSTEventSource.input_url',
        ('b', 'drs4-baseline'): 'DRS4Timelapse.drs4_baseline_path',
        ('o', 'output'): 'DRS4Timelapse.output_path',
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
        self.source.r0_r1_calibrator.apply_timelapse_correction = False
        self.source.r0_r1_calibrator.apply_drs4_pedestal_correction = False

        n_stats = N_GAINS * N_PIXELS * N_BINS_TOTAL
        self.dt_stats = OnlineStats(n_stats)
        self.baseline = read_drs4_baseline(self.drs4_baseline_path, self.source.tel_id)

    def start(self):
        tel_id = self.source.tel_id
        expected_pixels_id = self.source.camera_config.expected_pixels_id

        for event in tqdm(self.source):
            fill_stats(
                waveform=event.r1.tel[tel_id].waveform,
                baseline=self.baseline,
                first_cap=self.source.r0_r1_calibrator.first_cap[tel_id],
                last_first_cap=self.source.r0_r1_calibrator.first_cap_old[tel_id],
                last_readout_time=self.source.r0_r1_calibrator.last_readout_time[tel_id],
                local_clock_counter=event.lst.tel[tel_id].evt.local_clock_counter,
                expected_pixels_id=expected_pixels_id,
                dt_stats=self.dt_stats,
                skip_samples_front=self.skip_samples_front,
                skip_samples_end=self.skip_samples_end,
            )
            self.source.r0_r1_calibrator.update_last_readout_times(event, tel_id)

        bins = np.logspace(LOG_DT_MIN_MS, LOG_DT_MAX_MS, N_BINS)
        centers = 0.5 * (bins[:-1] + bins[1:])
        scale = np.empty((N_GAINS, N_PIXELS), dtype=np.float32)
        exponent = np.empty((N_GAINS, N_PIXELS), dtype=np.float32)
        offset = np.empty((N_GAINS, N_PIXELS), dtype=np.float32)

        shape = (N_GAINS, N_PIXELS, N_BINS_TOTAL)
        counts = self.dt_stats.counts.reshape(shape)
        mean = self.dt_stats.mean.reshape(shape)

        with tqdm(total=N_GAINS * N_PIXELS) as bar:
            for gain in range(N_GAINS):
                for pixel in range(N_PIXELS):
                    result = fit_timelapse(centers, counts, mean)

                    scale[gain, pixel] = result['scale']
                    exponent[gain, pixel] = result['exponent']
                    offset[gain, pixel] = result['offset']
                    bar.update(1)

    def finish(self):
        self.log.info('Writing output to %s', self.output_path)
        shape = (N_GAINS, N_PIXELS, N_BINS_TOTAL)

        transform = FixedPointColumnTransform(scale=100, offset=0, source_dtype=np.float64, target_dtype=np.int32)
        tel_id = self.source.tel_id
        table_name = f"r0/monitoring/timelapse_data/tel_{tel_id:03d}"

        with HDF5TableWriter(self.output_path) as writer:
            for col in ('mean', 'std'):
                writer.add_column_transform(table_name, col, transform)
            
            container = OnlineStatsContainer(
                counts=self.dt_stats.counts.reshape(shape),
                mean=self.dt_stats.mean.reshape(shape),
                std=self.dt_stats.std.reshape(shape),
            )
            writer.write(table_name, container)


def main():
    tool = DRS4Timelapse()
    tool.run()


if __name__ == '__main__':
    main()
