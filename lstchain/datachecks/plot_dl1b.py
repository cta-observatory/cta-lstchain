"""
Module for plotting image parameters from real data DL1 files.
"""

import argparse
from distutils.util import strtobool
from lstchain.io.io import dl1_params_lstcam_key, dl2_params_lstcam_key
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import os
import pandas as pd


__all__ = ['plot_dl1_params']


def plot_dl1_params(infile, cuts, min_intensity, max_intensity, leakage, wl):
    input_directory = os.path.dirname(infile)
    output_filename = os.path.basename(infile) + '.pdf'

    df_data = pd.read_hdf(infile, key=dl1_params_lstcam_key)

    # Sort events using dragon timestamps
    df_data.sort_values('dragon_time')

    # Apply cuts
    if cuts:
        df_data = df_data[(df_data['intensity'] > min_intensity) &
                          (df_data['intensity'] < max_intensity) &
                          (df_data['leakage'] < leakage) &
                          (df_data['wl'] > wl)]

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

        if cuts:
            plt.suptitle(f'Input file name: {infile}' + '\n' +
                         f'Cuts: ' + f'wl > {wl}, ' +
                         f'{min_intensity:.0e} < intensity (phe) <' +
                         f'{max_intensity:.0e}, ' +
                         f'leakage < {leakage}')
        else:
            plt.suptitle(f'Input file name: {infile}' + '\n' +
                         f'No cuts applied')

        # Intensity
        ax = axes[0][0]
        ax.hist(df_data['log_intensity'], histtype='step')
        ax.set_ylabel('Number of events')
        ax.set_xlabel(r'$log_{10}$ (intensity/phe)')
        ax.set_yscale('log')

        # Width
        ax = axes[0][1]
        ax.hist(df_data['width'], histtype='step')
        ax.set_ylabel('Number of events')
        ax.set_xlabel('Width (deg)')
        ax.set_yscale('log')

        # Length
        ax = axes[1][0]
        ax.hist(df_data['length'], histtype='step')
        ax.set_ylabel('Number of events')
        ax.set_xlabel('Length (deg)')
        ax.set_yscale('log')

        # Width/length
        ax = axes[1][1]
        ax.hist(df_data['wl'], histtype='step')
        ax.set_ylabel('Number of events')
        ax.set_xlabel('Width/Length')

        # Psi
        ax = axes[2][0]
        ax.hist(df_data['psi'] * 180/np.pi, histtype='step')
        ax.set_ylabel('Number of events')
        ax.set_xlabel('Psi (deg)')

        # Phi
        ax = axes[2][1]
        ax.hist(df_data['phi'] * 180/np.pi, histtype='step')
        ax.set_ylabel('Number of events')
        ax.set_xlabel('Phi (deg)')

        plt.subplots_adjust(wspace=0.25)
        plt.tight_layout(rect=[0, 0, 0.95, 0.9])
        pdf.savefig()

        # Second page

        fig, axes = plt.subplots(nrows=3, ncols=2)

        # Leakage
        ax = axes[0][0]
        ax.hist(df_data['leakage'], histtype='step')
        ax.set_ylabel('Number of events')
        ax.set_xlabel('Leakage')
        ax.set_yscale('log')

        # Number of islands
        ax = axes[0][1]
        ax.hist(df_data['n_islands'], histtype='step')
        ax.set_ylabel('Number of events')
        ax.set_xlabel('Number of islands')
        ax.set_yscale('log')

        # Intercept
        ax = axes[1][0]
        ax.hist(df_data['intercept'], histtype='step')
        ax.set_ylabel('Number of events')
        ax.set_xlabel('Intercept')
        ax.set_yscale('log')

        # Skewness
        ax = axes[1][1]
        ax.hist(df_data['skewness'], histtype='step')
        ax.set_ylabel('Number of events')
        ax.set_xlabel('Skewness')
        ax.set_yscale('log')

        # R coordinate of cog
        ax = axes[2][0]
        ax.hist(df_data['r'], histtype='step')
        ax.set_ylabel('Number of events')
        ax.set_xlabel(r'$r$ (m)')
        ax.set_yscale('log')

        # Camera (cog)
        ax = axes[2][1]
        cog = ax.hist2d(df_data['x'], df_data['y'], bins=50, norm=LogNorm())
        ax.set_xlabel(r'$x$ (m)')
        ax.set_ylabel(r'$y$ (m)')
        fig.colorbar(cog[-1], ax=ax)

        plt.subplots_adjust(wspace=0.25)
        plt.tight_layout(rect=[0, 0, 0.95, 0.9])
        pdf.savefig()

        # Third page

        fig, axes = plt.subplots(nrows=3, ncols=2)

        # Time gradient
        ax = axes[0][0]
        ax.hist(df_data['time_gradient'], histtype='step')
        ax.set_ylabel('Number of events')
        ax.set_xlabel('Time gradient')
        ax.set_yscale('log')

        # Kurtosis
        ax = axes[0][1]
        ax.hist(df_data['kurtosis'], histtype='step')
        ax.set_ylabel('Number of events')
        ax.set_xlabel('Kurtosis')
        ax.set_yscale('log')

        plt.subplots_adjust(wspace=0.25)
        plt.tight_layout(rect=[0, 0, 0.95, 0.9])
        pdf.savefig()

        # Fourth page

        fig = plt.figure()

        # TODO: indicate timestamps using datetime format
        # Timestamps
        ax1 = fig.add_subplot(3, 1, 1)
        ax1.plot(df_data['event_id'][df_data['ucts_time'] > 0],
                 df_data['ucts_time'][df_data['ucts_time'] > 0],
                 label='UCTS')
        ax1.plot(df_data['event_id'][df_data['tib_time'] > 0],
                 df_data['tib_time'][df_data['tib_time'] > 0],
                 label='TIB')
        ax1.plot(df_data['event_id'], df_data['dragon_time'],
                 '--', label='Dragon')
        ax1.set_ylabel('Timestamps (sec)')
        ax1.set_xlabel('Event Id')
        ax1.tick_params(axis='x', labelrotation=30)
        ax1.legend(loc=0)

        # Telescope altitude
        ax2 = fig.add_subplot(3, 1, 2)
        # plt.plot(df_data['ucts_time'], df_data['alt_tel'] * 180/np.pi)
        ax2.plot(df_data['dragon_time'], df_data['alt_tel'] * 180/np.pi)
        ax2.set_ylabel('Altitude (deg)')

        # Rates
        ax3 = fig.add_subplot(3, 1, 3, sharex=ax2)
        # plt.plot(df_data['ucts_time'], df_data['alt_tel'] * 180/np.pi)
        ax3.hist(timestamps, bins=e_bins, histtype='step', weights=weight_time)
        ax3.set_xlabel('Timestamp (sec)')
        ax3.set_ylabel('Rate (Hz)')

        plt.tight_layout(rect=[0, 0.25, 0.95, 0.95])

        pdf.savefig()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot DL1 image parameters")

    parser.add_argument("--infile", '-f', type=str,
                        dest='infile',
                        help="Path to the DL1 input file")

    parser.add_argument("--cuts", type=lambda x: bool(strtobool(x)),
                        dest='cuts',
                        help='Apply cuts',
                        default=False)

    parser.add_argument("--min_intensity", type=float,
                        dest='min_intensity',
                        help="Minimum value of intensity (in phe)",
                        default="1e2")

    parser.add_argument("--max_intensity", type=float,
                        dest='max_intensity',
                        help="Maximum value of intensity (in phe)",
                        default="1e5")

    parser.add_argument("--leakage", action='store', type=float,
                        dest='leakage',
                        help="Maximum value of leakage parameter",
                        default="0.2")

    parser.add_argument("--wl", '-wl', action='store', type=float,
                        dest='wl',
                        help="Minimum value of width/length parameter",
                        default="0.1")

    args = parser.parse_args()

    plot_dl1_params(args.infile, args.cuts, args.min_intensity,
            args.max_intensity, args.leakage, args.wl)
