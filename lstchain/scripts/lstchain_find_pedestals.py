#!/usr/bin/env python

"""
This script reads all DL1 subrun files (i.e. not merged) in the input
directory, and finds the event_id's of interleaved pedestals. It writes them
out into one hdf5 file for each sub-run, called pedestal_ids_RunXXXXX.YYYY.h5,
which contains a single table with one column, 'event_id'
"""

import argparse
from pathlib import Path

import numpy as np
import tables
from ctapipe.io import read_table

from lstchain.paths import parse_dl1_filename

parser = argparse.ArgumentParser(description="Interleaved Pedestal Finder")

parser.add_argument('-d', '--input-dir', dest='srcdir',
                    type=Path, default='./',
                    help='path to the directory of the DL1 files'
                    )

parser.add_argument('output_dir', metavar='output-dir', type=Path,
                    help='Path where to store the output hdf5 files'
                    )


def main():
    args = parser.parse_args()

    files = sorted(args.srcdir.glob('dl1_LST-1.Run?????.????.h5'))
    if not files:
        raise IOError("No input dl1 files found")

    output_dir = args.output_dir.absolute()
    output_dir.mkdir(exist_ok=True, parents=True)

    for file in files:
        run_info = parse_dl1_filename(file)
        run_number, subrun_index = run_info.run, run_info.subrun

        # Approximate interleaved pedestal frequency in Hz. This changed once (so far) in the life of LST.
        # We just use the run number to decide which frequency we should expect:
        approximate_frequency = 50
        if run_number > 2708:
            approximate_frequency = 100

        table = read_table(file, '/dl1/event/telescope/parameters/LST_LSTCam')
        all_times = np.array(table['dragon_time'])
        # probable interleaved flat-field events (just in case tag is
        # faulty!):
        ffmask = ((table['intensity'] > 3e4) &
                  (table['concentration_pixel'] < 0.005))

        pedmask = np.zeros(len(table), dtype=bool)
        pedmask[~ffmask] = find_pedestals(all_times[~ffmask],
                                          approximate_frequency)

        # Now remove the 10 brightest events (might be cosmics accidentally
        # falling in the time windows determined by time_pedestals). The
        # expected value of cosmics is smaller that, so we are probably
        # removing some pedestals, but it does not harm.

        intensity = np.array(table['intensity'])
        # set nans to 0, to avoid problems with sorting:
        intensity[np.isnan(intensity)] = 0

        decreasing_intensity_ordered_indices = np.flip(np.argsort(intensity))
        pedmask2 = pedmask[decreasing_intensity_ordered_indices]
        # indices of likely pedestals, ordered from brightest to dimmest:
        remove_indices = decreasing_intensity_ordered_indices[pedmask2]
        pedmask[remove_indices[:10]] = False

        n_pedestals = np.count_nonzero(pedmask)
        pedestal_rate = n_pedestals / (all_times[-1] - all_times[0])
        print(file)
        print(f'  Rate of identified pedestals: {pedestal_rate:.3f} Hz')

        if n_pedestals > 0:
            print(f'  Maximum intensity: {np.nanmax(intensity[pedmask]):.1f} pe')
        else:
            print('  Did not find any pedestal events')

        output_file = Path(
            output_dir,
            f'pedestal_ids_Run{run_number:0>5}.{subrun_index:0>4}.h5'
        )

        data = table[pedmask][['obs_id', 'event_id']].as_array()
        with tables.open_file(output_file, "w") as outfile:
            outfile.create_table(
                "/", "interleaved_pedestal_ids",
                obj=data,
            )


def find_pedestals(timestamps, expected_frequency=50):
    """
    Parameters
    ----------
    timestamps: ndarray, series of timestamps of LST1 events from a sub-run
    expected_frequency: expected approximate frequency (Hz) of the interleaved
    pedestals for the epoch of the observations

    Returns
    -------
    a mask which is True for the elements of timestamps which are most evenly
    distributed (and with the given approximate frequency)

    The function finds the set of events which are most evenly distributed
    in time with the approximate frequency expected_frequency. This should be the
    interleaved pedestals, which are very regularly spaced in time to better
    (typically) than microsec precision. Better to pass an array of
    timestamps in which flatfield events (easily recognizable) have been
    excluded.


    """
    period_mean = 1 / expected_frequency
    period_step_width = 1e-7  # s
    period_step_range = np.linspace(-50, 50, 101)

    phase = None
    best_contents = None
    best_bins = None
    best_period = None

    average_nevents_per_bin = 1
    # number of bins for the phase-folded histogram:
    nbins = int(len(timestamps) / average_nevents_per_bin)
    # approximately average_nevents_per_bin cosmics will remain per bin
    # (i.e. a contamination for the genuine pedestals we are looking for)

    # Find the period:
    max_peak = 0
    for i in period_step_range:
        tmod = timestamps % (period_mean + i * period_step_width)
        contents, bins = np.histogram(tmod, bins=nbins, range=(0, period_mean))
        if np.max(contents) < max_peak:
            continue
        max_peak = np.max(contents)
        best_bins = bins
        best_contents = contents
        best_period = period_mean + i * period_step_width
    # print('best_period =', best_period, ", best freq = ", 1/best_period)

    # adjust phase:
    max_peak = 0
    best_phase = 0
    steps = 1000  # full phase will be divided in this number of steps

    for phase in np.linspace(0, best_period, steps):
        tmod = (timestamps + phase) % best_period
        contents, bins = np.histogram(tmod, bins=nbins, range=(0, best_period))
        if np.max(contents) < max_peak:
            continue
        max_peak = np.max(contents)
        best_phase = phase
        best_contents = contents
        best_bins = bins

    tmod = (timestamps + best_phase) % best_period

    ibin = np.argmax(best_contents)
    # check the two neighboring bins, which may also contain pedestals.
    # Include them if content is at least 10% of the max:
    ifirst = ibin
    ilast = ibin

    bin_before = (ibin + len(best_contents) - 1) % len(best_contents)
    bin_after = (ibin + 1) % len(best_contents)

    if best_contents[bin_before] > 0.1 * best_contents[ibin]:
        ifirst = bin_before
    if best_contents[bin_after] > 0.1 * best_contents[ibin]:
        ilast = bin_after

    minval = best_bins[ifirst]
    maxval = best_bins[ilast + 1]

    if ilast >= ifirst:
        return ((tmod > minval) & (tmod < maxval))
    else:
        return ((tmod < maxval) | (tmod > minval))


if __name__ == '__main__':
    main()
