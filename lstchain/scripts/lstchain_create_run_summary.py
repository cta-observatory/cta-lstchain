"""
Create a run summary for a given date containing the number of subruns,
the start time of the run, type pf the run: DATA, DRS4, CALI, and
the reference timestamp and counter of the run.
"""

import argparse
import logging
from collections import Counter
from datetime import datetime
from glob import glob
from pathlib import Path

import numpy as np
from astropy.table import Table
from astropy.time import Time
from ctapipe.containers import EventType
from ctapipe_io_lst import (CDTS_AFTER_37201_DTYPE, CDTS_BEFORE_37201_DTYPE,
                            DRAGON_COUNTERS_DTYPE, LSTEventSource, MultiFiles)
from ctapipe_io_lst.event_time import combine_counters
from traitlets.config import Config

from lstchain import __version__
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
    default="/fefs/aswg/data/real/monitoring/RunSummary",
)

dtypes = {
    "ucts_timestamp": np.int64,
    "run_start": np.int64,
    "dragon_reference_time": np.int64,
    "dragon_reference_module_id": np.int16,
    "dragon_reference_module_index": np.int16,
    "dragon_reference_counter": np.uint64,
    "dragon_reference_source": str,
}


def get_list_of_files(r0_path):
    """
    Get the list of R0 files from a given date.

    Parameters
    ----------
    r0_path : pathlib.Path
        Path to the R0 files

    Returns
    -------
    list_of_files: pathlib.Path.glob
        List of files
    """
    return r0_path.glob("LST*.fits.fz")


def get_list_of_runs(list_of_files):
    """
    Get the sorted list of run objects from R0 filenames.

    Parameters
    ----------
    list_of_files : pathlib.Path.glob
        List of files
    Returns
    -------
    list_of_run_objects
    """
    return sorted(parse_r0_filename(file) for file in list_of_files)


def get_runs_and_subruns(list_of_run_objects, stream=1):
    """
    Get the list of run numbers and the number of sequential files (subruns)
    of each run.

    Parameters
    ----------
    list_of_run_objects
    stream: int, optional
        Number of the stream to obtain the number of sequential files (default is 1).

    Returns
    -------
    (run, number_of_files) : tuple
        Run numbers and corresponding subrun of each run.
    """
    list_filtered_stream = filter(lambda x: x.stream == stream, list_of_run_objects)

    run, number_of_files = np.unique(
        list(map(lambda x: x.run, list_filtered_stream)), return_counts=True
    )

    return run, number_of_files


def type_of_run(date_path, run_number, counters, n_events=500):
    """
    Guessing empirically the type of run based on the percentage of
    pedestals/mono trigger types from the first n_events:
    DRS4 pedestal run (DRS4): 100% mono events (trigger_type == 1)
    cosmic data run (DATA): <10% pedestal events (trigger_type == 32)
    pedestal-calibration run (PEDCALIB): ~50% mono, ~50% pedestal events
    Otherwise (ERROR) the run is not expected to be processed.
    This method may not give always the correct type.
    At some point this should be taken directly from TCU.

    Parameters
    ----------
    date_path : pathlib.Path
        Path to the R0 files
    run_number : int
        Run id
    counters : dict
        Dict containing the reference counters and timestamps
    n_events : int
        Number of events used to infer the type of the run

    Returns
    -------
    run_type: str
        Type of run (DRS4, PEDCALIB, DATA, ERROR)
    """

    pattern = f"LST-1.1.Run{run_number:05d}.0000*.fits.fz"
    list_of_files = sorted(date_path.glob(pattern))
    if len(list_of_files) == 0:
        log.error(f"First subrun not found for {pattern}")
        return "ERROR"

    filename = list_of_files[0]

    config = Config()
    config.EventTimeCalculator.dragon_reference_time = int(counters["dragon_reference_time"])
    config.EventTimeCalculator.dragon_reference_counter = int(counters["dragon_reference_counter"])
    config.EventTimeCalculator.dragon_module_id = int(counters["dragon_reference_module_id"])

    try:
        with LSTEventSource(filename, config=config, max_events=n_events) as source:
            source.log.setLevel(logging.ERROR)

            event_type_counts = Counter(event.trigger.event_type for event in source)
            n_pedestals = event_type_counts[EventType.SKY_PEDESTAL]
            n_subarray = event_type_counts[EventType.SUBARRAY]

        if n_subarray / n_events > 0.999:
            run_type = "DRS4"
        elif n_pedestals / n_events > 0.1:
            run_type = "PEDCALIB"
        elif n_pedestals / n_events < 0.1:
            run_type = "DATA"
        else:
            run_type = "ERROR"

    except (AttributeError, ValueError, IOError, IndexError) as err:
        log.error(f"File {filename} has error: {err!r}")

        run_type = "ERROR"

    return run_type


def read_counters(date_path, run_number):
    """
    Get initial valid timestamps from the first subrun.
    Write down the reference Dragon module used, reference event_id.

    Parameters
    ----------
    date_path: pathlib.Path
        Directory that contains the R0 files
    run_number: int
        Number of the run

    Returns
    -------
    dict: reference counters and timestamps
    """
    pattern = date_path / f"LST-1.*.Run{run_number:05d}.0000*.fits.fz"
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

            ucts_timestamp = np.int64(cdts["timestamp"][0])
            dragon_reference_time = ucts_timestamp
            dragon_reference_source = "ucts"
        else:
            ucts_timestamp = np.int64(-1)
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

    except Exception as err:
        log.error(f"Files {pattern} have error: {err}")

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
    Build an astropy Table with run summary information and write it
    as ECSV file with the following information (one row per run):
     - run_id
     - number of subruns
     - type of run (DRS4, CALI, DATA, CONF)
     - start of the run
     - dragon reference UCTS timestamp if available (-1 otherwise)
     - dragon reference time source ("ucts" or "run_date")
     - dragon_reference_module_id
     - dragon_reference_module_index
     - dragon_reference_counter
    """

    args = parser.parse_args()

    date_path = args.r0_path / args.date

    file_list = get_list_of_files(date_path)
    runs = get_list_of_runs(file_list)
    run_numbers, n_subruns = get_runs_and_subruns(runs)

    reference_counters = [read_counters(date_path, run) for run in run_numbers]

    run_types = [
        type_of_run(date_path, run, counters)
        for run, counters in zip(run_numbers, reference_counters)
    ]

    run_summary = Table(
        {
            col: np.array([d[col] for d in reference_counters], dtype=dtype)
            for col, dtype in dtypes.items()
        }
    )
    run_summary.meta["date"] = datetime.strptime(args.date, "%Y%m%d").date().isoformat()
    run_summary.meta["lstchain_version"] = __version__
    run_summary.add_column(run_numbers, name="run_id", index=0)
    run_summary.add_column(n_subruns, name="n_subruns", index=1)
    run_summary.add_column(run_types, name="run_type", index=2)
    run_summary.write(
        args.output_dir / f"RunSummary_{args.date}.ecsv", format="ascii.ecsv", delimiter=","
    )


if __name__ == "__main__":
    main()
