import re
from collections import namedtuple
import os


Run = namedtuple('R0Path', 'tel_id stream run subrun')
R0_RE = re.compile(r'LST-(\d+).(\d+).Run(\d+).(\d+).fits.fz')

DL1_RE = re.compile(
    r'dl1_LST-(\d+)'     # tel_id
    r'(?:.(\d+))?'       # stream is optional
    r'.Run(\d+)'         # run number
    r'.(\d+)'            # subrun number
    r'(?:.fits)?'        # stream is optional
    r'.(?:h5|hdf5|hdf)'  # usual extensions for hdf5 files
)


def parse_int(string):
    if string is None:
        return None
    return int(string)


def parse_r0_filename(filename):
    '''Parse an raw data file name and return a nametuple of it's components'''
    m = R0_RE.match(os.path.basename(filename))

    if m is None:
        raise ValueError(f'Filename {filename} does not match pattern {R0_RE}')

    return Run(*map(parse_int, m.groups()))


def parse_dl1_filename(filename):
    '''Parse an raw data file name and return a nametuple of it's components'''
    m = DL1_RE.match(os.path.basename(filename))

    if m is None:
        raise ValueError(f'Filename {filename} does not match pattern {R0_RE}')

    return Run(*map(parse_int, m.groups()))


def r0_run_to_filename(run):
    return f'LST-{run.tel_id}.{run.stream}.Run{run.run:05d}.{run.subrun:04d}.fits.fz'


def dl1_run_to_filename(run):
    if run.stream is None:
        return f'dl1_LST-{run.tel_id}.Run{run.run:05d}.{run.subrun:04d}.h5'
    return f'dl1_LST-{run.tel_id}.{run.stream}.Run{run.run:05d}.{run.subrun:04d}.h5'
