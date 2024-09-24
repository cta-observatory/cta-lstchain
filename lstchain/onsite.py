from glob import glob
import logging
from pathlib import Path
from enum import Enum, auto
from datetime import datetime
import tempfile
from astropy.time import Time
import pymongo

from .paths import parse_calibration_name
from .io.io import get_resource_path

log = logging.getLogger(__name__)

DEFAULT_BASE_PATH = Path('/fefs/aswg/data/real')
DEFAULT_R0_PATH = DEFAULT_BASE_PATH / 'R0'
DEFAULT_DL1_PATH = DEFAULT_BASE_PATH / 'DL1'
CAT_A_PIXEL_DIR = 'monitoring/PixelCalibration/Cat-A'
CAT_B_PIXEL_DIR = 'monitoring/PixelCalibration/Cat-B'


DEFAULT_CONFIG = get_resource_path("data/onsite_camera_calibration_param.json")

DEFAULT_CONFIG_CAT_B_CALIB = get_resource_path("data/catB_camera_calibration_param.json")

class DataCategory(Enum):
    #: Real-Time data processing
    A = auto()
    #: Onsite processing
    B = auto()
    #: Offsite processing
    C = auto()

def is_date(s):
    try:
        datetime.strptime(s, '%Y%m%d')
        return True
    except ValueError:
        return False


def create_symlink_overwrite(link, target):
    '''
    Create a symlink from link to target, replacing an existing link atomically.
    '''
    if not link.exists():
        link.symlink_to(target)
        return

    if link.resolve() == target:
        # nothing to do
        return

    # create the symlink in a tempfile, then replace the original one
    # in one step to avoid race conditions
    tmp = tempfile.NamedTemporaryFile(
        prefix='tmp_symlink_',
        delete=True,
        # use same directory as final link to assure they are on the same device
        # avoids "Invalid cross-device link error"
        dir=link.parent,
    )
    tmp.close()
    tmp = Path(tmp.name)
    tmp.symlink_to(target)
    tmp.replace(link)


def create_pro_symlink(output_dir):
    '''Create a "pro" symlink to the dir in the same directory'''
    output_dir = Path(output_dir)
    pro_link = output_dir.parent / 'pro'

    # the pro-link should be relative to make moving / copying a tree easy
    create_symlink_overwrite(pro_link, output_dir.relative_to(pro_link.parent))


def rglob_symlinks(path, pattern):
    """
    Same as Path.rglob, but actually following symlinks, needed to find R0G files
    """
    # convert results back to path
    return (Path(p) for p in glob(f'{path}/**/{pattern}'))


def find_r0_subrun(run, sub_run, r0_dir=DEFAULT_R0_PATH):
    '''
    Find the given subrun R0 file (i.e. globbing for the date part)
    '''

    file_list = rglob_symlinks(r0_dir, f'LST-1.1.Run{run:05d}.{sub_run:04d}*.fits.fz')
    # ignore directories that are not a date, e.g. "Trash"
    file_list = [p for p in file_list if is_date(p.parent.name)]


    if len(file_list) == 0:
        raise IOError(f"Run {run} not found in r0_dir {r0_dir} \n")

    if len(file_list) > 1:
        raise IOError(f"Found more than one file for run {run}.{sub_run}: {file_list}")

    return file_list[0]


def find_pedestal_file(pro, pedestal_run=None, date=None, base_dir=DEFAULT_BASE_PATH):
    # pedestal base dir
    ped_dir = Path(base_dir) / CAT_A_PIXEL_DIR / "drs4_baseline"

    if pedestal_run is None and date is None:
        raise ValueError('Must give at least `date` or `run`')

    if pedestal_run is not None:
        # search a specific pedestal run
        file_list = sorted(ped_dir.rglob(f'*/{pro}/drs4_pedestal.Run{pedestal_run:05d}.0000.h5'))

        if len(file_list) == 0:
            raise IOError(f"Pedestal file from run {pedestal_run} not found\n")

        return file_list[0].resolve()

    # search for a unique pedestal file from the same date
    file_list = sorted((ped_dir / date / pro).glob('drs4_pedestal*.0000.h5'))
    if len(file_list) == 0:
        raise IOError(f"No pedestal file found for date {date}")

    if len(file_list) > 1:
        raise IOError(f"Too many pedestal files found for date {date}: {file_list}, choose one run\n")

    return file_list[0].resolve()


def find_run_summary(date, base_dir=DEFAULT_BASE_PATH):
    run_summary_path = base_dir / f"monitoring/RunSummary/RunSummary_{date}.ecsv"
    if not run_summary_path.exists():
        raise IOError(f"Night summary file {run_summary_path} does not exist\n")
    return run_summary_path


def find_time_calibration_file(pro, run, time_run=None, base_dir=DEFAULT_BASE_PATH):
    '''Find a time calibration file for given run
    '''
    time_dir = Path(base_dir) / CAT_A_PIXEL_DIR / "drs4_time_sampling_from_FF"


    # search the last time run before or equal to the calibration run
    if time_run is None:
        file_list = sorted(time_dir.rglob(f'*/{pro}/time_calibration.Run*.0000.h5'))

        if len(file_list) == 0:
            raise IOError(f"No time calibration file found in the data tree for prod {pro}\n")

        time_file = None
        for path in file_list:
            run_in_list = parse_calibration_name(path)
            if run_in_list.run <= run:
                time_file = path.resolve()
            else:
                break

        if time_file is None:
            raise IOError(f"No time calibration file found before run {run} for prod {pro}\n")

        return time_file

    # if given, search a specific time file
    file_list = sorted(time_dir.rglob(f'*/{pro}/time_calibration.Run{time_run:05d}.0000.h5'))
    if len(file_list) == 0:
        raise IOError(f"Time calibration file from run {time_run} not found\n")

    return file_list[0].resolve()


def find_systematics_correction_file(pro, date, sys_date=None, base_dir=DEFAULT_BASE_PATH):
    sys_dir = Path(base_dir) / CAT_A_PIXEL_DIR / "ffactor_systematics"

    if sys_date is not None:
        path =  (sys_dir / sys_date / pro / f"ffactor_systematics_{sys_date}.h5").resolve()
        if not path.exists():
            raise IOError(f"F-factor systematics correction file {path} does not exist")
        return path

    dir_list = sorted(sys_dir.rglob(f"*/{pro}/ffactor_systematics*"))
    if len(dir_list) == 0:
        raise IOError(f"No systematic correction file found for production {pro} in {sys_dir}\n")

    sys_date_list = sorted([path.parts[-3] for path in dir_list], reverse=True)
    selected_date = next((day for day in sys_date_list if day <= date), sys_date_list[-1])

    return (sys_dir / selected_date / pro / f"ffactor_systematics_{selected_date}.h5").resolve()

def find_calibration_file(pro, calibration_run=None, date=None, category=DataCategory.A, base_dir=DEFAULT_BASE_PATH):

    if category == DataCategory.A:
        cal_dir = Path(base_dir) / CAT_A_PIXEL_DIR / "calibration"
    elif category == DataCategory.B:
        cal_dir = Path(base_dir) / CAT_B_PIXEL_DIR / "calibration"
    else:
        raise ValueError(f'Argument \'category\' can be only \'DataCategory.A\' or \'DataCategory.B\', not {category}')    

    if calibration_run is None and date is None:
        raise ValueError('Must give at least `date` or `run`')

    if calibration_run is not None:
        # search a specific calibration run
        file_list = sorted(cal_dir.rglob(f'{pro}/calibration*.Run{calibration_run:05d}.0000.h5'))

        if len(file_list) == 0:
            raise IOError(f"Calibration file from run {calibration_run} not found\n")

        return file_list[0].resolve()

    # search for a unique calibration file from the same date
    file_list = sorted((cal_dir / date / pro).glob('calibration*.0000.h5'))
    if len(file_list) == 0:
        raise IOError(f"No calibration file found for date {date}")

    if len(file_list) > 1:
        raise IOError(f"Too many calibration files found for date {date}: {file_list}, choose one run\n")

    return file_list[0].resolve()

def find_DL1_subrun(run, sub_run, dl1_dir=DEFAULT_DL1_PATH):
    '''
    Find the given subrun DL1 file (i.e. globbing for the date part)
    '''
    file_list = rglob_symlinks(dl1_dir, f'dl1_LST-1.Run{run:05d}.{sub_run:04d}*.h5')
    
    # ignore directories that are not a date, e.g. "Trash"
    file_list = [p for p in file_list if is_date(p.parent.name)]

    if len(file_list) == 0:
        raise IOError(f"Run {run} not found\n")

    if len(file_list) > 1:
        raise IOError(f"Found more than one file for run {run}.{sub_run}: {file_list}")

    return file_list[0]

def find_interleaved_subruns(run, interleaved_dir):
    '''
    Return the list of interleaved files for a given run
    '''

    file_list = sorted(interleaved_dir.rglob(f'interleaved_LST-1.Run{run:05d}.*.h5'))
    
    if len(file_list) == 0:
        raise IOError(f"Run {run} not found in interleaved dir {interleaved_dir}\n")
    
    return file_list

def find_filter_wheels(run, database_url):
    """read the employed filters from mongodb"""

    # there was a change of Mongo DB data names on 5/12/2022
    NEW_DB_NAMES_DATE = Time("2022-12-04T00:00:00")

    filters = None
    try:

        myclient = pymongo.MongoClient(database_url)

        mydb = myclient["CACO"]
        mycol = mydb["RUN_INFORMATION"]
        mydoc = mycol.find({"run_number": {"$eq": run}})
        for x in mydoc:
            date =  Time(x["start_time"])
            if date < NEW_DB_NAMES_DATE:
                w1 = int(x["cbox"]["wheel1 position"])
                w2 = int(x["cbox"]["wheel2 position"])
            else:
                w1 = int(x["cbox"]["CBOX_WheelPosition1"])
                w2 = int(x["cbox"]["CBOX_WheelPosition2"])

            filters = f"{w1:1d}{w2:1d}"

    except Exception as e:  # In the case the entry says 'No available'
        log.exception(f"\n >>> Exception: {e}")
        raise IOError(
            "--> No mongo DB filter information."
            " You must pass the filters by argument: -f [filters]"
        )

    if filters is None:  # In the case the entry is missing
        raise IOError(
            "--> No filter information in mongo DB."
            " You must pass the filters by argument: -f [filters]"
        )

    return filters
 