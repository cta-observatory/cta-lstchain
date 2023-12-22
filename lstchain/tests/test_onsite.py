import pytest
import os
from pathlib import Path
import json
import shutil


test_data = Path(os.getenv('LSTCHAIN_TEST_DATA', 'test_data'))
test_r0_path = test_data / 'real/R0/'
test_subrun1 = test_r0_path / '20200218/LST-1.1.Run02008.0000_first50.fits.fz'
test_subrun2 = test_r0_path / '20210215/LST-1.1.Run03669.0000_first50.fits.fz'

PRO = 'ctapipe-v0.17'
BASE_DIR = test_data / 'real'


def test_default_config():
    from lstchain.onsite import DEFAULT_CONFIG

    assert DEFAULT_CONFIG.is_file()

    # test it's valid json
    with DEFAULT_CONFIG.open('rb') as f:
        json.load(f)


def test_create_symlink_overwrite(tmp_path):
    from lstchain.onsite import create_symlink_overwrite
    target1 = tmp_path / 'target1'
    target1.open('w').close()

    target2 = tmp_path / 'target2'
    target2.open('w').close()

    # link not yet existing case
    link = tmp_path / 'link'
    create_symlink_overwrite(link, target1)
    assert link.resolve() == target1.resolve()

    # link points to the wrong target, recreate
    create_symlink_overwrite(link, target2)
    assert link.resolve() == target2.resolve()


    # link exists, points already to the target, this should be a no-op
    # but I didn't find a good way to verify that it really is one
    assert link.resolve() == target2.resolve()


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

    # test that prolink is relative, not absolute
    assert os.readlink(pro) == 'v1'

    # test pro exists and points to older version
    create_pro_symlink(v2)
    assert pro.exists()
    assert pro.resolve() == v2


@pytest.mark.private_data
def test_find_r0_subrun(tmp_path):
    from lstchain.onsite import find_r0_subrun

    tmp_r0 = tmp_path / 'R0'
    correct = tmp_r0 / test_subrun1.parent.name
    correct.mkdir(parents=True)
    shutil.copy2(test_subrun1, correct / test_subrun1.name)

    # copy another run so we can make sure we really find the right one
    other = tmp_r0 / test_subrun2.parent.name
    other.mkdir(parents=True)
    shutil.copy2(test_subrun2, other / test_subrun2.name)

    path = find_r0_subrun(2008, 0, tmp_r0)
    assert path.resolve().parent == correct


@pytest.mark.private_data
def test_find_r0_subrun_trash(tmp_path):
    from lstchain.onsite import find_r0_subrun

    # test we ignore everything not looking like a date
    tmp_r0 = tmp_path / 'R0'
    trash = tmp_r0 / 'Trash'
    correct = tmp_r0 / '20200218'
    trash.mkdir(parents=True)
    correct.mkdir(parents=True)

    shutil.copy2(test_subrun1, trash / test_subrun1.name)
    shutil.copy2(test_subrun1, correct / test_subrun1.name)

    path = find_r0_subrun(2008, 0, tmp_r0)
    assert path.resolve().parent == correct



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


def test_rglob_symlinks(tmp_path):
    from lstchain.onsite import rglob_symlinks

    # create a test structure similar to the real data
    r0 = tmp_path / 'R0'
    r0g = tmp_path / 'R0G'

    paths = [
        r0 / '20220101/run1.dat',
        r0 / '20220101/run2.dat',
        r0 / '20220102/run3.dat',
        r0 / '20220102/run4.dat',
        r0g / '20220103/run5.dat',
        r0g / '20220103/run6.dat',
    ]
    for path in paths:
        path.parent.mkdir(exist_ok=True, parents=True)
        path.open("w").close()
        if "R0G" in path.parts:
            # symlink R0G files to R0
            target = Path(str(path.parent).replace('R0G', 'R0'))
            print(target, target.exists())
            if not target.exists():
                target.symlink_to(path.parent)


    # check "normal" file
    matches = rglob_symlinks(r0, "run1.dat")
    # check we get an iterator and not a list
    assert iter(matches) is iter(matches)
    assert list(matches) == [r0 / "20220101/run1.dat"]

    # check file in symlinked dir
    matches = rglob_symlinks(r0, "run5.dat")
    # check we get an iterator and not a list
    assert list(matches) == [r0 / "20220103/run5.dat"]

    # check multiple files
    matches = rglob_symlinks(r0, "run*.dat")
    # check we get an iterator and not a list
    assert len(list(matches)) == 6

@pytest.mark.private_data
def test_find_calibration_file():
    from lstchain.onsite import find_calibration_file

    # find by run_id
    path = find_calibration_file(pro=PRO, calibration_run=9506, base_dir=BASE_DIR)
    assert path.name == 'calibration_filters_52.Run09506.0000.h5'

    # find by night
    path = find_calibration_file(pro=PRO, date='20200218', base_dir=BASE_DIR)
    assert path.name == 'calibration_filters_52.Run02006.0000.h5'

    # if both are given, run takes precedence
    path = find_calibration_file(pro=PRO, calibration_run=2006, date='20191124', base_dir=BASE_DIR)
    assert path.name == 'calibration_filters_52.Run02006.0000.h5'

    with pytest.raises(IOError):
        # if many calibration runs in one date
        find_calibration_file(pro=PRO, date='20221001', base_dir=BASE_DIR)
   
    with pytest.raises(IOError):
        # wrong run
        find_calibration_file(pro=PRO, calibration_run=2010, base_dir=BASE_DIR)
