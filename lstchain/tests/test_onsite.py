import pytest
import os
from pathlib import Path


test_data = Path(os.getenv('LSTCHAIN_TEST_DATA', 'test_data'))
test_r0_path = test_data / 'real/R0/'
test_subrun = test_r0_path / '20200218/LST-1.1.Run02008.0000_first50.fits.fz'
PRO = 'v0.8.2.post2.dev48+gb1343281'
BASE_DIR = test_data / 'real'


def test_create_pro_link(tmp_path: Path):
    from lstchain.onsite import create_pro_symlink

    v1 = tmp_path / 'v1'
    v2 = tmp_path / 'v2'
    pro = tmp_path / 'pro'

    v1.mkdir()
    v2.mkdir()

    # test pro does not yet exist
    create_pro_symlink(v1)
    assert pro.exists()
    assert pro.resolve() == v1

    # test pro exists and points to older version
    create_pro_symlink(v2)
    assert pro.exists()
    assert pro.resolve() == v2


@pytest.mark.private_data
def test_find_r0_subrun():
    from lstchain.onsite import find_r0_subrun

    path = find_r0_subrun(2008, 0, test_r0_path)
    assert path.resolve() == test_subrun.resolve()


@pytest.mark.private_data
def test_find_pedestal_path():
    from lstchain.onsite import find_pedestal_file

    # find by run_id
    path = find_pedestal_file(pro=PRO, pedestal_run=2005, base_dir=BASE_DIR)
    assert path.name == 'drs4_pedestal.Run02005.0000.h5'

    # find by night
    path = find_pedestal_file(pro=PRO, date='20191124', base_dir=BASE_DIR)
    assert path.name == 'drs4_pedestal.Run01623.0000.h5'

    # if both are given, run takes precedence
    path = find_pedestal_file(pro=PRO, pedestal_run=2005, date='20191124', base_dir=BASE_DIR)
    assert path.name == 'drs4_pedestal.Run02005.0000.h5'


    with pytest.raises(IOError):
        # wrong run
        find_pedestal_file(pro=PRO, pedestal_run=2010, base_dir=BASE_DIR)


@pytest.mark.private_data
def test_find_run_summary():
    from lstchain.onsite import find_run_summary

    # find by run_id
    path = find_run_summary(date='20200218', base_dir=BASE_DIR)
    assert path.name == 'RunSummary_20200218.ecsv'

    path = find_run_summary(date='20201120', base_dir=BASE_DIR)
    assert path.name == 'RunSummary_20201120.ecsv'

    with pytest.raises(IOError):
        find_run_summary(date='20221120', base_dir=BASE_DIR)


@pytest.mark.private_data
def test_find_time_calibration_file():
    from lstchain.onsite import find_time_calibration_file

    path = find_time_calibration_file(pro=PRO, run=2008, base_dir=BASE_DIR)
    assert path.name == 'time_calibration.Run01625.0000.h5'
    assert PRO in str(path)
    assert path.exists()

    path = find_time_calibration_file(pro=PRO, run=1625, base_dir=BASE_DIR)
    assert path.name == 'time_calibration.Run01625.0000.h5'
    assert PRO in str(path)
    assert path.exists()


@pytest.mark.private_data
def test_find_systematics_correction_file():
    from lstchain.onsite import find_systematics_correction_file

    # no sys date
    path = find_systematics_correction_file(pro=PRO, date='20200218', base_dir=BASE_DIR)
    assert path.name == 'ffactor_systematics_20200725.h5'

    path = find_systematics_correction_file(pro=PRO, date='20200218', sys_date='20200725', base_dir=BASE_DIR)
    assert path.name == 'ffactor_systematics_20200725.h5'

    with pytest.raises(IOError):
        # nonexistent sys date
        path = find_systematics_correction_file(pro=PRO, date='20200218', sys_date='20190101', base_dir=BASE_DIR)
