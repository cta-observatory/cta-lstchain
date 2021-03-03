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
from ctapipe.containers import EventType
from ctapipe.io import EventSource
from ctapipe_io_lst.event_time import CENTRAL_MODULE
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


def get_run_numbers(list_of_runs):
    """Get the list of run numbers of a given date."""
    return np.unique([run_info.run for run_info in list_of_runs])


def get_number_of_subruns(run_number, list_of_run_objects):
    """Obtain the number of sequential files (subruns) of a given run."""
    filtered_run = filter(lambda x: x.run == run_number, list_of_run_objects)
    last_subrun = max(list(filtered_run), key=lambda x: x.subrun)
    # Since subruns are counted from 0, the number of files is increased by 1.
    return last_subrun.subrun + 1


def start_of_run_datetime(run_number):
    """
    Get datetime of the start of the run in ISOT format and UTC scale.
    """
    filename = args.R0_PATH / args.date / f"LST-1.1.Run{run_number:05d}.0000.fits.fz"
    file = protozfits.File(str(filename))
    return file.CameraConfig.header["DATE"]


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
        n_pedestal = sum(
            1 for event in source if event.trigger.event_type == EventType.SKY_PEDESTAL
        )
        n_sky = sum(
            1 for event in source if event.trigger.event_type == EventType.SUBARRAY
        )

    # FIXME: Do this classification in some other way?
    if n_sky / n_events > 0.999:
        run_type = "DRS4"
    elif n_pedestal / n_events > 0.1:
        run_type = "CALI"
    elif n_pedestal / n_events < 0.1:
        run_type = "DATA"
    else:
        run_type = "UNKW"

    return run_type


def get_initial_timestamps(run_number):
    """
    Get initial valid timestamps. Write down the Dragon module used.
    First subrun needs to be open. Taken from ctapipe_io_lst.event_time
    """
    filename = args.R0_PATH / args.date / f"LST-1.1.Run{run_number:05d}.0000.fits.fz"

    with EventSource(input_url=filename) as source:

        for event in source:
            lst = event.lst.tel[tel_id]
            central_module_index = np.where(lst.svc.module_ids == CENTRAL_MODULE)[0][0]
            module_index = central_module_index
            ucts_available = lst.evt.extdevices_presence & 2

            # Look for the first ucts timestamp available
            if ucts_available:
                dragon_reference_source = "ucts"
                dragon_reference_timestamp = lst.evt.ucts_timestamp
                initial_dragon_counter = (
                    int(1e9) * lst.evt.pps_counter[module_index]
                    + 100 * lst.evt.tenMHz_counter[module_index]
                )

                return (
                    dragon_reference_source,
                    dragon_reference_timestamp,
                    initial_dragon_counter,
                    module_index,
                )

            else:
                # No UCTS available in the first subrun
                dragon_reference_source = "run_date"
                dragon_reference_timestamp = lst.svc.date
                initial_dragon_counter = (
                    int(1e9) * lst.evt.pps_counter[module_index]
                    + 100 * lst.evt.tenMHz_counter[module_index]
                )

                return (
                    dragon_reference_source,
                    dragon_reference_timestamp,
                    initial_dragon_counter,
                    module_index,
                )


def write_run_summary_to_file(date, run_numbers, list_of_run_objects):
    """
    Write to a .txt file the following information per run:
    Run number
    Number of subruns starting from 0
    Type of run
    Start_of_the_run
    Reference_source ("ucts" or "run_date")
    Reference_timestamp
    Initial dragon counter
    Dragon module ID used to take the counter values
    """
    # TODO: Be able to create file incrementally run-by-run
    with open(args.output_dir / f"RunSummary_{date}.txt", "a+") as summary_file:
        for run in run_numbers:
            n_subruns = get_number_of_subruns(run, list_of_run_objects)

            (
                dragon_reference_source,
                dragon_reference_timestamp,
                initial_dragon_counter,
                module_index,
            ) = get_initial_timestamps(run)

            run_information = (
                f"{run:05d},"
                f"{n_subruns},"
                f"{type_of_run(run)},"
                f"{start_of_run_datetime(run)},"
                f"{dragon_reference_source},"
                f"{dragon_reference_timestamp},"
                f"{initial_dragon_counter},"
                f"{module_index}\n"
            )
            summary_file.write(run_information)


def main():
    list_of_files = get_list_of_files(args.date)
    list_of_run_objects = get_list_of_runs(list_of_files)
    run_numbers = get_run_numbers(list_of_run_objects)
    write_run_summary_to_file(args.date, run_numbers, list_of_run_objects)


if __name__ == "__main__":
    main()
