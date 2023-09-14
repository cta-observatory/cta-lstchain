"""
Create a summary of all runs from daily run summaries,
adding pointing information.

It is also possible to append a single night to an already
existing merged summary file when date is especified.
"""

import logging
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.table import Table, vstack
from astropy.time import Time
from ctapipe_io_lst import LSTEventSource
from ctapipe_io_lst.event_time import read_run_summary
from ctapipe_io_lst.pointing import PointingSource

log = logging.getLogger('create_run_overview')

RUN_SUMMARY_COLUMNS = [
    'run_id',
    'n_subruns',
    'run_type',
    'run_start',
]

base_path = Path('/fefs/aswg/data/real/monitoring')

parser = ArgumentParser()
parser.add_argument('output_file', type=Path)
parser.add_argument('-m', '--monitoring-path', type=Path, default=base_path)
parser.add_argument('-v', '--verbose', action='store_true')
# Option to only append a single night to an existing merged summary file.
# By default, the whole directory with nightly summaries is processed.
parser.add_argument(
    '-d',
    '--date',
    type=str,
    default=None,
    help='Date in YYYYMMDD format. When the date is given append only '
    'the summary of that night to an existing merged summary file',
)


SUBARRAY = LSTEventSource.create_subarray()


def get_pointing_info(times, drive_report):
    pointing_source = PointingSource(
        SUBARRAY,
        drive_report_path=drive_report,
    )
    try:
        pointing_source._read_drive_report_for_tel(1)
        valid = pointing_source.drive_log[1]['unix_time'] != 0
        pointing_source.drive_log[1] = pointing_source.drive_log[1][valid]
        if np.count_nonzero(valid) < 2:
            raise ValueError('Not enough values')

    except:
        return {
            'ra': np.full(len(times), np.nan) * u.deg,
            'dec': np.full(len(times), np.nan) * u.deg,
            'alt': np.full(len(times), np.nan) * u.rad,
            'az': np.full(len(times), np.nan) * u.rad,
        }

    pointing_info = {k: [] for k in ('ra', 'dec', 'alt', 'az')}

    for time in times:
        try:
            ra, dec = pointing_source.get_pointing_position_icrs(tel_id=1, time=time)
        except ValueError:
            ra = dec = np.nan * u.deg

        pointing_info['ra'].append(ra)
        pointing_info['dec'].append(dec)

        try:
            altaz = pointing_source.get_pointing_position_altaz(tel_id=1, time=time)
            alt = altaz.altitude
            az = altaz.azimuth
        except ValueError:
            alt = az = np.nan * u.rad

        pointing_info['alt'].append(alt)
        pointing_info['az'].append(az)

    return pointing_info


def merge_run_summary_with_pointing(run_summary, drive_report):
    table = read_run_summary(run_summary)[RUN_SUMMARY_COLUMNS]

    if len(table) == 0:
        return None

    table['run_start'] = Time(table['run_start'] / 1e9, format='unix_tai', scale='utc')

    pointing_info = get_pointing_info(table['run_start'], drive_report)
    for k, v in pointing_info.items():
        table[k] = u.Quantity(v)

    # add date as column and remove from meta date so merging does not complain
    table['date'] = table.meta['date']
    del table.meta['date']
    del table.meta['lstchain_version']

    table['run_start'].format = 'isot'
    # reorder columns
    table = table[
        [
            'date',
            'run_id',
            'run_type',
            'n_subruns',
            'run_start',
            'ra',
            'dec',
            'alt',
            'az',
        ]
    ]
    return table


def main():
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.date is not None:
        # Append only the run summary corresponding to this date
        night = datetime.strptime(args.date, '%Y%m%d')
        if not args.output_file.is_file():
            raise ValueError(f'Output file {args.output_file} does not exist.')

        log.info(f'Appending {args.date} to {args.output_file}')
        run_summary_dir = args.monitoring_path / 'RunSummary'
        drive_report_dir = args.monitoring_path / 'DrivePositioning'
        run_summary = run_summary_dir / f'RunSummary_{args.date}.ecsv'
        drive_report = drive_report_dir / f'DrivePosition_log_{args.date}.txt'
        summary = merge_run_summary_with_pointing(run_summary, drive_report)
        if summary is not None:
            old_summary = Table.read(args.output_file)
            new_summary = vstack([old_summary, summary])
            new_summary.write(args.output_file, overwrite=True)
        return

    # Otherwise, merge all summary files found
    run_summary_dir = args.monitoring_path / 'RunSummary'
    drive_report_dir = args.monitoring_path / 'DrivePositioning'

    run_summaries = sorted(run_summary_dir.glob('RunSummary*.ecsv'))
    log.info('Found %d run summaries', len(run_summaries))

    summaries = []

    for run_summary in run_summaries:
        log.debug('Processing %s', run_summary)
        night = datetime.strptime(run_summary.stem, 'RunSummary_%Y%m%d')

        drive_report = drive_report_dir / f'DrivePosition_log_{night:%Y%m%d}.txt'

        if not drive_report.is_file():
            log.error(f'No drive report found for {night:%Y-%m-%d}')
            continue

        summary = merge_run_summary_with_pointing(
            run_summary,
            drive_report,
        )
        if summary is not None:
            summaries.append(summary)

    vstack(summaries).write(args.output_file, overwrite=True)


if __name__ == '__main__':
    main()
