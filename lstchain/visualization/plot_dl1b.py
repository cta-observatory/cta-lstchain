import argparse
from distutils.util import strtobool
from lstchain.io.io import dl1_params_lstcam_key, dl2_params_lstcam_key
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import os
import pandas as pd


parser = argparse.ArgumentParser(description="Plots of DL1 b image parameters")

parser.add_argument("--infile", '-f', type=str,
                    dest='infile',
                    help="Path to the DL1 input file")

parser.add_argument("--cuts", type=lambda x: bool(strtobool(x)),
                    dest='cuts',
                    help='Apply cuts',
                    default=False)

parser.add_argument("--min_intensity", type=float,
                    dest='min_intensity',
                    help="Minimum value of intensity",
                    default="1e2")

parser.add_argument("--max_intensity", type=float,
                    dest='max_intensity',
                    help="Maximum value of intensity",
                    default="1e5")

parser.add_argument("--leakage_cut", action='store', type=float,
                    dest='leakage_cut',
                    help="Maximum value of leakage parameter",
                    default="0.2")

parser.add_argument("--wl_cut", '-wl', action='store', type=float,
                    dest='wl_cut',
                    help="Minimum value of width/length parameter",
                    default="0.1")

args = parser.parse_args()


def main():
    input_directory = os.path.dirname(args.infile)
    output_filename = os.path.basename(args.infile) + '.pdf'

    df_data = pd.read_hdf(args.infile, key=dl1_params_lstcam_key)

    # Sort events using dragon timestamps
    df_data.sort_values('dragon_time')

    # Apply cuts
    if args.cuts:
        df_data = df_data[(df_data['intensity'] > args.min_intensity) &
                          (df_data['intensity'] < args.max_intensity) &
                          (df_data['leakage'] < args.leakage_cut) &
                          (df_data['wl'] > args.wl_cut)]

    # Determine the duration of the total number of events in the file
    timestamps = df_data['dragon_time'][df_data['dragon_time'] > 0]
    duration = timestamps.iloc[-1] - timestamps.iloc[0]
    e_bins = int(np.round(duration) / 10)  # 10 secs bin
    weight_time = np.ones_like(timestamps) * 0.1  # Beware the factor 10
    if e_bins <= 0:
        e_bins = 1

    plt.rcParams['figure.figsize'] = (15, 17)
    plt.rcParams['font.size'] = 15
    plt.rcParams['patch.linewidth'] = 2
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['hist.bins'] = 100

    with PdfPages(output_filename) as pdf:

        fig, axes = plt.subplots(nrows=3, ncols=2)

        if args.cuts:
            plt.suptitle(f'Input file name: {args.infile}' + '\n' +
                         f'Cuts: ' + f'wl > {args.wl_cut}, ' +
                         f'{args.min_intensity:.0e} < intensity <' +
                         f'{args.max_intensity:.0e}, ' +
                         f'leakage < {args.leakage_cut}')
        else:
            plt.suptitle(f'Input file name: {args.infile}' + '\n' +
                         f'No cuts applied')

        # Intensity distribution
        ax = axes[0][0]
        ax.hist(df_data['log_intensity'], histtype='step')
        ax.set_ylabel('Number of events')
        ax.set_xlabel(r'$log_{10}$ Intensity')
        ax.set_yscale('log')

        # Width distribution
        ax = axes[0][1]
        ax.hist(df_data['width'], histtype='step')
        ax.set_ylabel('Number of events')
        ax.set_xlabel('Width (deg)')
        ax.set_yscale('log')

        # Length distribution
        ax = axes[1][0]
        ax.hist(df_data['length'], histtype='step')
        ax.set_ylabel('Number of events')
        ax.set_xlabel('Length (deg)')
        ax.set_yscale('log')

        # Psi distribution
        ax = axes[1][1]
        ax.hist(df_data['psi'] * 180/np.pi, histtype='step')
        ax.set_ylabel('Number of events')
        ax.set_xlabel('Psi (deg)')

        # Rates
        ax = axes[2][0]
        ax.hist(timestamps, bins=e_bins, histtype='step', weights=weight_time)
        ax.set_xlabel('Timestamp (sec)')
        ax.set_ylabel('Rate (Hz)')

        # Center of gravity
        ax = axes[2][1]
        cog = ax.hist2d(df_data['x'], df_data['y'], bins=100)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        fig.colorbar(cog[-1], ax=ax)

        plt.subplots_adjust(wspace=0.25)
        plt.tight_layout(rect=[0, 0, 0.95, 0.9])
        pdf.savefig()

        # Second page

        fig, axes = plt.subplots(nrows=2, ncols=1)

        # TODO: indicate timestamps using datetime format
        # Timestamps
        ax = axes[0]
        ax.plot(df_data['event_id'][df_data['ucts_time'] > 0],
                df_data['ucts_time'][df_data['ucts_time'] > 0],
                label='UCTS')
        ax.plot(df_data['event_id'][df_data['tib_time'] > 0],
                df_data['tib_time'][df_data['tib_time'] > 0],
                label='TIB')
        ax.plot(df_data['event_id'], df_data['dragon_time'],
                '--', label='Dragon')
        ax.set_ylabel('Timestamps (sec)')
        ax.set_xlabel('Event Id')
        ax.tick_params(axis='x', labelrotation=30)
        ax.legend(loc=0)

        # Telescope altitude
        ax = axes[1]
        # plt.plot(df_data['ucts_time'], df_data['alt_tel'] * 180/np.pi)
        ax.plot(df_data['dragon_time'], df_data['alt_tel'] * 180/np.pi)
        ax.set_ylabel('Altitude (deg)')
        ax.set_xlabel('Timestamps (sec)')

        plt.tight_layout(rect=[0, 0.5, 0.95, 1])

        pdf.savefig()


if __name__ == '__main__':
    main()
