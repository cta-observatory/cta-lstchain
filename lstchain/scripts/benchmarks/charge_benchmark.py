import argparse
import logging
import os
import sys
from pathlib import Path

import ctaplot
import matplotlib.pyplot as plt
import tables
from astropy.table import Table
from ctaplot.plots.calib import (
    plot_charge_resolution,
    plot_photoelectron_true_reco,
    plot_pixels_pe_spectrum,
)
from matplotlib.backends.backend_pdf import PdfPages

from lstchain.io.config import (
    get_standard_config,
    read_configuration_file,
)
from lstchain.paths import r0_to_dl1_filename
from lstchain.reco import r0_to_dl1

log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="R0 to DL1")

# Required arguments
parser.add_argument('--input-file', '-f', type=Path,
                    dest='input_file',
                    help='Path to the simtelarray file',
                    required=True
                    )

# Optional arguments
parser.add_argument('--output-dir', '-o', action='store', type=Path,
                    dest='output_dir',
                    help='Path where to store the reco dl2 events',
                    default='./benchs_results/')

parser.add_argument('--config', '-c', action='store', type=Path,
                    dest='config_file',
                    help='Path to a configuration file. If none is given, a standard configuration is applied',
                    default=None
                    )

args = parser.parse_args()


def main():
    ctaplot.set_style()

    output_dir = args.output_dir.absolute()
    output_dir.mkdir(exist_ok=True, parents=True)
    output_file = output_dir / r0_to_dl1_filename(args.input_file.name)

    r0_to_dl1.allowed_tels = {1, 2, 3, 4}

    if args.config_file is not None:
        try:
            config = read_configuration_file(args.config_file.absolute())
        except Exception as e:
            log.error(f'Config file {args.config_file} could not be read: {e}')
            sys.exit(1)
    else:
        config = get_standard_config()

    # This benchmark needs true pe image
    config['write_pe_image'] = True

    # directly jump to the benchmarks if the dl1 file already exists
    if not os.path.exists(output_file):
        r0_to_dl1.r0_to_dl1(
            args.input_file,
            output_filename=output_file,
            custom_config=config,
        )

    with tables.open_file(output_file) as f:
        sim_table = Table(f.root.dl1.event.simulation.LST_LSTCam.read())
        im_table = Table(f.root.dl1.event.telescope.image.LST_LSTCam.read())

    if len(sim_table) != len(im_table):
        raise ValueError('the number of events with simulation info is not equal to the number of dl1 events')

    pdf_filename = os.path.join(args.output_dir, f"charge_bench_{os.path.basename(output_file).replace('.h5', '')}.pdf")
    with PdfPages(pdf_filename) as pdf:

        plot_pixels_pe_spectrum(sim_table['true_image'], im_table['image'])
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        plot_photoelectron_true_reco(sim_table['true_image'], im_table['image'])
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        ax = plot_charge_resolution(sim_table['true_image'], im_table['image'])
        ax.set_ylim(-1, 10)
        plt.tight_layout()
        pdf.savefig()
        plt.close()


if __name__ == '__main__':
    main()
