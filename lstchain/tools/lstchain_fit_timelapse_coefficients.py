import re

import numpy as np
from iminuit import Minuit
from iminuit.cost import LeastSquares
import tables
from tqdm.auto import tqdm

from ctapipe.core import Tool, Container, Field
from ctapipe.core.traits import Path
from ctapipe.io import HDF5TableReader, HDF5TableWriter

from ..io.lstcontainers import OnlineStatsContainer

TIMELAPSE_GROUP = "/r0/monitoring/timelapse_data"
TEL_ID_RE = re.compile(r"tel_(\d+)")


def delta_t_correction(x, scale=32.99, exponent=0.22, offset=-11.9):
    return scale * x**(-exponent) + offset


def fit_timelapse(centers, counts, mean, std):
    mean_err = std / np.sqrt(counts)
    mask = (counts > 500) & (centers > 0.05)
    cost = LeastSquares(centers[mask], mean[mask], mean_err[mask], delta_t_correction)
    m = Minuit(cost, scale=32, exponent=0.22, offset=-12)
    m.limits['scale'] = (1e-30, None)
    m.migrad()
    return m.values, cost(*m.values)


class TimeLapseCoefficients(Container):
    default_prefix = ""
    scale = Field(np.nan, "Number of samples")
    exponent = Field(np.nan, "mean")
    offset = Field(np.nan, "standard deviation")
    chi2 = Field(np.nan, "Least Squares value of fit")


class DRS4TimelapseFitter(Tool):
    name = 'lstchain_create_drs4_timelapse_file'

    input_path = Path(directory_ok=False).tag(config=True)
    output_path = Path(directory_ok=False).tag(config=True)

    aliases = {
        ('i', 'input'): 'DRS4TimelapseFitter.input_path',
        ('o', 'output'): 'DRS4TimelapseFitter.output_path',
    }

    def setup(self):
        self.h5file = self.enter_context(tables.open_file(self.input_path))
        self.writer = self.enter_context(HDF5TableWriter(self.output_path))

    def start(self):
        for table in self.h5file.root[TIMELAPSE_GROUP]._f_iter_nodes():
            tel_id = int(TEL_ID_RE.match(table._v_name).group(1))

            with HDF5TableReader(self.h5file) as reader:
                timelapse_data = next(reader.read(table._v_pathname, OnlineStatsContainer))

            n_gains, n_pixels, n_bins_total = timelapse_data.counts.shape
            # remove over / underflow
            n_bins = n_bins_total - 2
            counts = timelapse_data.counts[:, :, 1:-1]
            mean = timelapse_data.mean[:, :, 1:-1]
            std = timelapse_data.std[:, :, 1:-1]

            dt_min = timelapse_data.meta["log10_dt_min_ms"]
            dt_max = timelapse_data.meta["log10_dt_max_ms"]
            bins = np.logspace(dt_min, dt_max, n_bins + 1)
            centers = 0.5 * (bins[:-1] + bins[1:])
            scale = np.empty((n_gains, n_pixels), dtype=np.float32)
            exponent = np.empty((n_gains, n_pixels), dtype=np.float32)
            offset = np.empty((n_gains, n_pixels), dtype=np.float32)
            chi2 = np.empty((n_gains, n_pixels), dtype=np.float32)

            with tqdm(total=n_gains * n_pixels) as bar:
                for gain in range(n_gains):
                    for pixel in range(n_pixels):
                        result, cost = fit_timelapse(centers, counts[gain, pixel], mean[gain, pixel], std[gain, pixel])

                        scale[gain, pixel] = result['scale']
                        exponent[gain, pixel] = result['exponent']
                        offset[gain, pixel] = result['offset']
                        chi2[gain, pixel] = cost
                        bar.update(1)

            container = TimeLapseCoefficients(scale=scale, exponent=exponent, offset=offset, chi2=chi2)
            self.writer.write(f"r1/service/timelapse_coefficients/tel_{tel_id:03d}", container)


def main():
    DRS4TimelapseFitter().run()



if __name__ == "__main__":
    main()

