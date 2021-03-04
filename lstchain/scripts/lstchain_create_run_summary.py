"""
Create a run summary for a given date containing the number of subruns,
the start time of the run, type pf the run: DATA, DRS4, CALI, and
the reference timestamp and counter of the run.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import protozfits
from astropy.table import Table
from astropy.time import Time
from ctapipe.containers import EventType
from ctapipe.io import EventSource
from lstchain.paths import parse_r0_filename

# FIXME: take it from somewhere else
tel_id = 1


log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Create run summary file")

parser.add_argument(
    "-d",
    "--date",
    type=str,
    help="Date for the creation of the run summary in format YYYYMMDD",
    required=True,
)
# TODO: default today()

parser.add_argument(
    "--r0-path",
    type=Path,
    dest="R0_PATH",
    help="Path to the R0 files. Default is /fefs/aswg/data/real/R0",
    default=Path("/fefs/aswg/data/real/R0"),
)

parser.add_argument(
    "-o",
    "--output-dir",
    type=Path,
    dest="output_dir",
    help="Directory in which Run Summary file is written",
    default="./",
)


args = parser.parse_args()


def get_list_of_files(date):
    """Get the list of R0 files from a given date."""
    path_r0 = args.R0_PATH / date
    # FIXME: use regular expressions from lstchain.paths.R0_RE
    list_of_files = path_r0.glob("LST*.fits.fz")
    return list_of_files


def get_list_of_runs(list_of_files):
    """Get the sorted list of run objects from R0 filenames."""
    return sorted(parse_r0_filename(file) for file in list_of_files)


def get_runs_and_subruns(list_of_run_objects):
    """
    Get the list of run numbers and the number of sequenctial files (subruns)
    of each run for a given date assuming 4 ZFW streams.
    """
    NUMBER_OF_STREAMS = 4

    run, number_of_files = np.unique(
        list(map(lambda x: x.run, list_of_run_objects)), return_counts=True
    )

    n_subruns = number_of_files / NUMBER_OF_STREAMS
    return run, n_subruns.astype(int)


def type_of_run(run_number, n_events=500):
    """
    Get empirically the type of run based on the percentage of
    pedestals/mono trigger types from the first n_events:
    100% mono events (trigger 1): DRS4 pedestal run
    <10% pedestal events (trigger 32): cosmic DATA run
    ~50% mono, ~50% pedestal events: PEDESTAL-CALIBRATION run
    First subrun needs to be open.
    """
    filename = args.R0_PATH / args.date / f"LST-1.1.Run{run_number:05d}.0000.fits.fz"

    with EventSource(input_url=filename, max_events=n_events) as source:
        n_pedestal_events = sum(
            1 for event in source if event.trigger.event_type == EventType.SKY_PEDESTAL
        )
        n_sky_events = sum(1 for event in source if event.trigger.event_type == EventType.SUBARRAY)

    # FIXME: Do this classification in some other way?
    if n_sky_events / n_events > 0.999:
        run_type = "DRS4"
    elif n_pedestal_events / n_events > 0.1:
        run_type = "CALI"
    elif n_pedestal_events / n_events < 0.1:
        run_type = "DATA"
    else:
        run_type = "UNKW"

    return run_type


def initial_timestamps(run_number):
    """
    Get initial valid timestamps from the first subrun.
    Write down the reference Dragon module used, reference event_id.
    Taken from ctapipe_io_lst.event_time.
    """
    filename = args.R0_PATH / args.date / f"LST-1.1.Run{run_number:05d}.0000.fits.fz"

    with EventSource(input_url=filename) as source:

        for event in source:
            lst = event.lst.tel[tel_id]
            # Start of the run timestamp (nanoseconds) in UNIX TAI
            run_date = int(round(Time(lst.svc.date, format="unix").unix_tai)) * int(1e9)
            # Get the first available module
            module_index = np.where(lst.evt.module_status != 0)[0][0]
            ucts_available = lst.evt.extdevices_presence & 2

            # Look for the first ucts timestamp available
            if ucts_available:
                dragon_reference_source = "ucts"
                dragon_reference_timestamp = lst.evt.ucts_timestamp
                initial_dragon_counter = (
                    int(1e9) * lst.evt.pps_counter[module_index]
                    + 100 * lst.evt.tenMHz_counter[module_index]
                )
                break

            else:
                # No UCTS available in the first subrun
                dragon_reference_source = "run_date"
                dragon_reference_timestamp = -1
                initial_dragon_counter = (
                    int(1e9) * lst.evt.pps_counter[module_index]
                    + 100 * lst.evt.tenMHz_counter[module_index]
                )
                break

    return {
        "run": run_number,
        "start_of_the_run": run_date,
        "dragon_reference_source": dragon_reference_source,
        "event_id": event.index.event_id,
        "dragon_reference_timestamp": dragon_reference_timestamp,
        "initial_dragon_counter": initial_dragon_counter,
        "module_id": module_index,
    }


def main(date):
    """
    Write run summary to a file the following information per run:
    Run number
    Number of subruns
    Type of run
    Start_of_the_run
    Event ID used to take time reference
    Reference_source ("ucts" or "run_date")
    Reference_timestamp
    Initial dragon counter
    Dragon module ID used to take the counter values
    """
    # TODO: Be able to create file incrementally run-by-run

    list_of_files = get_list_of_files(date)
    list_of_run_objects = get_list_of_runs(list_of_files)
    run_numbers, n_subruns = get_runs_and_subruns(list_of_run_objects)
    list_type_of_runs = [type_of_run(run) for run in run_numbers]
    dict_run_timestamps = [initial_timestamps(run) for run in run_numbers]

    run_summary = Table(dict_run_timestamps)
    run_summary.add_column(n_subruns, name="n_subruns", index=1)
    run_summary.add_column(list_type_of_runs, name="type_of_run", index=2)
    run_summary.write(args.output_dir / f"RunSummary_{date}.txt", format="ascii.csv")


if __name__ == "__main__":
    main(args.date)
