import os
import re
from collections import namedtuple
from pathlib import Path

Run = namedtuple('Run', 'tel_id run subrun stream', defaults=(None, ))
R0_RE = re.compile(r'LST-(\d+).(\d+).Run(\d+).(\d+)(?:.*).fits.fz')

DL1_RE = re.compile(
    r'dl1_LST-(\d+)'     # tel_id
    r'(?:.(\d+))?'       # stream is optional
    r'.Run(\d+)'         # run number
    r'.(\d+)'            # subrun number
    r'(?:.*)'            # allow arbitrary stuff between the run part and the extension
    r'.(?:h5|hdf5|hdf)'  # usual extensions for hdf5 files
)
DC_DL1_RE = re.compile(
    r'datacheck_dl1_LST-(\d+)'  # tel_id
    r'.Run(\d+)'                # run number
    r'(?:.(\d+))?'              # subrun number is optional
    r'(?:.*)'                   # allow arbitrary stuff between the run part and the extension
    r'.(?:h5|hdf5|hdf)'         # usual extensions for hdf5 files
)

GENERAL_RE = re.compile(
    r'LST-(\d+)'         # tel_id
    r'(?:.(\d+))?'       # stream is optional
    r'.Run(\d+)'         # run number
    r'.(\d+)'            # subrun number
)

EXTENSIONS_TO_REMOVE = {
    '.fits',
    '.fits.fz',
    '.simtel',
    '.simtel.gz',
    '.simtel.zst',
}


def parse_int(string):
    if string is None:
        return None
    return int(string)


def _parse_match(match):
    values = [parse_int(v) for v in match.groups()]
    return Run(tel_id=values[0], run=values[2], subrun=values[3], stream=values[1])


def run_info_from_filename(filename):
    '''Generic function to search a filename for the LST-t.s.Runxxxxx.yyyy'''
    m = GENERAL_RE.search(os.path.basename(filename))
    if m is None:
        raise ValueError(f'Filename {filename} does not include pattern {GENERAL_RE}')

    return _parse_match(m)


def parse_r0_filename(filename):
    '''
    Parse a canonical r0 file name
    and return a ``Run`` namedtuple of it's components

    Parameters
    ----------
    filename: str or pathlib.Path
        the filename to parse

    Returns
    -------
    run: Run
        namedtuple with fields tel_id, run, subrun and stream

    Raises
    ------
    ValueError: when the filename does not match the expected pattern
    '''
    m = R0_RE.match(os.path.basename(filename))

    if m is None:
        raise ValueError(f'Filename {filename} does not match pattern {R0_RE}')

    return _parse_match(m)


def parse_dl1_filename(filename):
    '''
    Parse canonical dl1 file name and return a ``Run`` namedtuple of it's components

    Parameters
    ----------
    filename: str or pathlib.Path
        the filename to parse

    Returns
    -------
    run: Run
        namedtuple with fields tel_id, run, subrun and stream

    Raises
    ------
    ValueError: when the filename does not match the expected pattern
    '''
    m = DL1_RE.match(os.path.basename(filename))

    if m is None:
        raise ValueError(f'Filename {filename} does not match pattern {DL1_RE}')

    return _parse_match(m)


def parse_datacheck_dl1_filename(filename):
    '''
    Parse canonical datacheck dl1 file name and return a ``Run``
    namedtuple of its components

    Parameters
    ----------
    filename: str or pathlib.Path
        the filename to parse

    Returns
    -------
    run: Run
        namedtuple with fields tel_id, run, subrun
    Raises
    ------
    ValueError: when the filename does not match the expected pattern

    '''

    m = DC_DL1_RE.match(os.path.basename(filename))

    if m is None:
        raise ValueError(f'Filename {filename} does not match pattern '
                         f'{DC_DL1_RE}')

    values = [parse_int(v) for v in m.groups()]
    return Run(tel_id=values[0], run=values[1], subrun=values[2])


def run_to_r0_filename(tel_id, run, subrun, stream=None):
    '''
    Create the filename for an r0 file from telescope / run info.
    If you have a `Run` tuple, use like this: ``r0_run_to_filename(*run)``
    '''
    return f'LST-{tel_id}.{stream}.Run{run:05d}.{subrun:04d}.fits.fz'


def run_to_filename(prefix, tel_id, run, subrun, stream=None, ext='.h5'):
    name = f'{prefix}_LST-{tel_id}'
    if stream is not None:
        name += f'.{stream}'
    name += f'.Run{run:05d}'
    if subrun is not None:
        name += f'.{subrun:04d}'
    name += f'{ext}'
    return name


def run_to_dl1_filename(tel_id, run, subrun, stream=None):
    '''
    Create the filename for a dl1 file from telescope / run info.
    If you have a `Run` tuple, use like this: ``r0_run_to_filename(*run)``
    '''
    return run_to_filename('dl1', tel_id, run, subrun, stream)


def run_to_datacheck_dl1_filename(tel_id, run, subrun, stream=None):
    '''
    Create the filename for a datacheck dl1 file from telescope / run info.
    If you have a `Run` tuple, use like this:
    ``run_to_datacheck_dl1_filename(*run)``
    '''
    return run_to_filename('datacheck_dl1', tel_id, run, subrun, stream)


def run_to_dl2_filename(tel_id, run, subrun, stream=None):
    '''
    Create the filename for a dl1 file from telescope / run info.
    If you have a `Run` tuple, use like this: ``r0_run_to_filename(*run)``
    '''
    return run_to_filename('dl2', tel_id, run, subrun, stream)


def run_to_muon_filename(tel_id, run, subrun, stream=None, gzip=True):
    '''
    Create the filename for a muon output file from telescope / run info.
    If you have a `Run` tuple, use like this: ``r0_run_to_filename(*run)``
    '''
    ext = '.fits.gz' if gzip else '.fits'
    return run_to_filename('muons', tel_id, run, subrun, stream, ext=ext)


def r0_to_dl1_filename(r0_path):
    '''Function to add dl1_ in front and replace extension with h5'''
    for ext in EXTENSIONS_TO_REMOVE:
        r0_path, *exts = r0_path.rsplit(ext, 1)

    p = Path(r0_path)
    return p.with_name('dl1_' + p.name + '.h5')
