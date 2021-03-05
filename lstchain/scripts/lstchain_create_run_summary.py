"""
Create a run summary for a given date containing the number of subruns,
the start time of the run, type pf the run: DATA, DRS4, CALI, and
the reference timestamp and counter of the run.
"""

import argparse
import logging
from glob import glob
from pathlib import Path

import numpy as np
from astropy.table import Table
from astropy.time import Time
from ctapipe.containers import EventType
from ctapipe_io_lst import (
    CDTS_AFTER_37201_DTYPE,
    CDTS_BEFORE_37201_DTYPE,
    DRAGON_COUNTERS_DTYPE,
    LSTEventSource,
    MultiFiles,
)
from ctapipe_io_lst.event_time import combine_counters
from traitlets.config import Config

from lstchain.paths import parse_r0_filename

log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Create run summary file")

parser.add_argument(
    "-d",
    "--date",
    help="Date for the creation of the run summary in format YYYYMMDD",
    required=True,
)

parser.add_argument(
    "--r0-path",
    type=Path,
    help="Path to the R0 files. Default is /fefs/aswg/data/real/R0",
    default=Path("/fefs/aswg/data/real/R0"),
)

parser.add_argument(
    "-o",
    "--output-dir",
    type=Path,
    help="Directory in which Run Summary file is written",
    default="./",
)


def get_list_of_files(r0_path):
    """Get the list of R0 files from a given date."""
    # FIXME: use regular expressions from lstchain.paths.R0_RE
    list_of_files = r0_path.glob("LST*.fits.fz")
    return list_of_files


def get_list_of_runs(list_of_files):
    """Get the sorted list of run objects from R0 filenames."""
    return sorted(parse_r0_filename(file) for file in list_of_files)


def get_runs_and_subruns(list_of_run_objects, stream=1):
    """
    Get the list of run numbers and the number of sequenctial files (subruns)
    of each run.
    """
    list_filtered_stream = filter(lambda x: x.stream == stream, list_of_run_objects)

    run, number_of_files = np.unique(
        list(map(lambda x: x.run, list_filtered_stream)), return_counts=True
    )

    return run, number_of_files


def type_of_run(date_path, run_number, counters, n_events=500):
    """
    Get empirically the type of run based on the percentage of
    pedestals/mono trigger types from the first n_events:
    100% mono events (trigger 1): DRS4 pedestal run
    <10% pedestal events (trigger 32): cosmic DATA run
    ~50% mono, ~50% pedestal events: PEDESTAL-CALIBRATION run
    First subrun needs to be open.
    """
    filename = date_path / f"LST-1.1.Run{run_number:05d}.0000.fits.fz"

    config = Config()
    config.EventTimeCalculator.dragon_reference_time = int(counters["dragon_reference_time"])
    config.EventTimeCalculator.dragon_reference_counter = int(counters["dragon_reference_counter"])

    try:
        with LSTEventSource(filename, config=config, max_events=n_events) as source:
            n_pedestal_events = sum(
                event.trigger.event_type == EventType.SKY_PEDESTAL for event in source
            )
            n_sky_events = sum(event.trigger.event_type == EventType.SUBARRAY for event in source)

        if n_sky_events / n_events > 0.999:
            run_type = "DRS4"
        elif n_pedestal_events / n_events > 0.1:
            run_type = "CALI"
        elif n_pedestal_events / n_events < 0.1:
            run_type = "DATA"
        else:
            run_type = "CONF"

    except AttributeError:
        log.error(f"Files {filename} do not contain events")

        run_type = "CONF"

    return run_type


def read_counters(date_path, run_number):
    """
    Get initial valid timestamps from the first subrun.
    Write down the reference Dragon module used, reference event_id.
    """
    pattern = date_path / f"LST-1.*.Run{run_number:05d}.0000.fits.fz"
    try:
        f = MultiFiles(glob(str(pattern)))
        first_event = next(f)

        if first_event.event_id != 1:
            raise ValueError("Must be used on first file streams (subrun)")

        module_index = np.where(first_event.lstcam.module_status)[0][0]
        module_id = np.where(f.camera_config.lstcam.expected_modules_id == module_index)[0][0]
        dragon_counters = first_event.lstcam.counters.view(DRAGON_COUNTERS_DTYPE)
        dragon_reference_counter = combine_counters(
            dragon_counters["pps_counter"][module_index],
            dragon_counters["tenMHz_counter"][module_index],
        )

        ucts_available = bool(first_event.lstcam.extdevices_presence & 2)
        run_start = int(round(Time(f.camera_config.date, format="unix").unix_tai)) * int(1e9)

        if ucts_available:
            if int(f.camera_config.lstcam.idaq_version) > 37201:
                cdts = first_event.lstcam.cdts_data.view(CDTS_AFTER_37201_DTYPE)
            else:
                cdts = first_event.lstcam.cdts_data.view(CDTS_BEFORE_37201_DTYPE)

            ucts_timestamp = cdts["timestamp"][0]
            dragon_reference_time = ucts_timestamp
            dragon_reference_source = "ucts"
        else:
            ucts_timestamp = -1
            dragon_reference_time = run_start
            dragon_reference_source = "run_start"

        return dict(
            ucts_timestamp=ucts_timestamp,
            run_start=run_start,
            dragon_reference_time=dragon_reference_time,
            dragon_reference_module_id=module_id,
            dragon_reference_module_index=module_index,
            dragon_reference_counter=dragon_reference_counter,
            dragon_reference_source=dragon_reference_source,
        )
    except AttributeError:
        log.error(f"Files {pattern} do not contain events")

        return dict(
            ucts_timestamp=-1,
            run_start=-1,
            dragon_reference_time=-1,
            dragon_reference_module_id=-1,
            dragon_reference_module_index=-1,
            dragon_reference_counter=-1,
            dragon_reference_source=None,
        )


def main():
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

    args = parser.parse_args()

    date_path = args.r0_path / args.date

    list_of_files = get_list_of_files(date_path)
    list_of_run_objects = get_list_of_runs(list_of_files)
    run_numbers, n_subruns = get_runs_and_subruns(list_of_run_objects)
    reference_counters = [read_counters(date_path, run) for run in run_numbers]
    list_type_of_runs = [
        type_of_run(date_path, run, counters)
        for run, counters in zip(run_numbers, reference_counters)
    ]

    run_summary = Table(reference_counters)
    run_summary.add_column(run_numbers, name="run_numbers", index=0)
    run_summary.add_column(n_subruns, name="n_subruns", index=1)
    run_summary.add_column(list_type_of_runs, name="type_of_run", index=2)
    run_summary.write(
        args.output_dir / f"RunSummary_{args.date}.ecsv", format="ascii.ecsv", delimiter=","
    )


if __name__ == "__main__":
    main()
