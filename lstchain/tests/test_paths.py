import pytest
from pathlib import Path


def test_generic():
    from lstchain.paths import run_info_from_filename, Run

    assert run_info_from_filename('LST-1.2.Run01920.0001') == Run(1, 1920, 1, 2)
    assert run_info_from_filename('/foo/bar/LST-1.2.Run01920.0001') == Run(1, 1920, 1, 2)
    assert run_info_from_filename('./foo/LST-1.2.Run01920.0001') == Run(1, 1920, 1, 2)
    assert run_info_from_filename('foo/LST-1.2.Run01920.0001') == Run(1, 1920, 1, 2)
    assert run_info_from_filename('fooLST-1.2.Run01920.0001.bar') == Run(1, 1920, 1, 2)
    assert run_info_from_filename('fooLST-1.Run01920.0001.fits.gz') == Run(1, 1920, 1)
    with pytest.raises(ValueError):
        run_info_from_filename('MST-1.Run01920.0001.fits.gz')


def test_parse_dl1():
    from lstchain.paths import parse_dl1_filename

    run = parse_dl1_filename('dl1_LST-1.1.Run01920.0000.fits.h5')
    assert run.tel_id == 1
    assert run.run == 1920
    assert run.subrun == 0
    assert run.stream == 1

    run = parse_dl1_filename(Path('dl1_LST-1.1.Run01920.0000.fits.h5'))
    assert run.tel_id == 1
    assert run.run == 1920
    assert run.subrun == 0
    assert run.stream == 1

    run = parse_dl1_filename('dl1_LST-1.Run01920.0000.fits.h5')
    assert run.tel_id == 1
    assert run.run == 1920
    assert run.subrun == 0
    assert run.stream is None

    run = parse_dl1_filename('dl1_LST-1.Run01920.0000.h5')
    assert run.tel_id == 1
    assert run.run == 1920
    assert run.subrun == 0
    assert run.stream is None

    run = parse_dl1_filename('dl1_LST-1.Run01920.0000.fits.hdf5')
    assert run.tel_id == 1
    assert run.run == 1920
    assert run.subrun == 0
    assert run.stream is None

    run = parse_dl1_filename('dl1_LST-1.Run01920.0000_some_custom_junk.hdf5')
    assert run.tel_id == 1
    assert run.run == 1920
    assert run.subrun == 0
    assert run.stream is None

    with pytest.raises(ValueError):
        run = parse_dl1_filename('foo.fits.fz')


def test_dl1_to_filename():
    from lstchain.paths import run_to_dl1_filename, Run

    assert run_to_dl1_filename(
        tel_id=1, run=1920, subrun=2
    ) == 'dl1_LST-1.Run01920.0002.h5'
    assert run_to_dl1_filename(
        tel_id=1, run=1920, subrun=3, stream=2
    ) == 'dl1_LST-1.2.Run01920.0003.h5'

    run = Run(tel_id=2, run=5, subrun=1)
    assert run_to_dl1_filename(*run) == 'dl1_LST-2.Run00005.0001.h5'


def test_dl2_to_filename():
    from lstchain.paths import run_to_dl2_filename, Run

    assert run_to_dl2_filename(
        tel_id=1, run=1920, subrun=2
    ) == 'dl2_LST-1.Run01920.0002.h5'
    assert run_to_dl2_filename(
        tel_id=1, run=1920, subrun=3, stream=2
    ) == 'dl2_LST-1.2.Run01920.0003.h5'

    run = Run(tel_id=2, run=5, subrun=1)
    assert run_to_dl2_filename(*run) == 'dl2_LST-2.Run00005.0001.h5'


def test_parse_r0():
    from lstchain.paths import parse_r0_filename

    run = parse_r0_filename('LST-1.1.Run01920.0000.fits.fz')
    assert run.tel_id == 1
    assert run.stream == 1
    assert run.run == 1920
    assert run.subrun == 0

    run = parse_r0_filename(Path('LST-1.1.Run01920.0000.fits.fz'))
    assert run.tel_id == 1
    assert run.stream == 1
    assert run.run == 1920
    assert run.subrun == 0

    with pytest.raises(ValueError):
        run = parse_r0_filename('foo.fits.fz')


def test_r0_to_filename():
    from lstchain.paths import run_to_r0_filename

    assert run_to_r0_filename(
        tel_id=1, run=1920, subrun=3, stream=2
    ) == 'LST-1.2.Run01920.0003.fits.fz'


def test_muon_filename():
    from lstchain.paths import run_to_muon_filename

    assert run_to_muon_filename(
        tel_id=1, run=2, subrun=3
    ) == 'muons_LST-1.Run00002.0003.fits.gz'

    assert run_to_muon_filename(
        tel_id=1, run=2, subrun=3, gzip=False
    ) == 'muons_LST-1.Run00002.0003.fits'

    assert run_to_muon_filename(
        tel_id=1, run=2, subrun=3, stream=4
    ) == 'muons_LST-1.4.Run00002.0003.fits.gz'


def test_r0_to_dl1_filename():
    from lstchain.paths import r0_to_dl1_filename

    assert str(r0_to_dl1_filename('/foo/test.simtel.gz')) == '/foo/dl1_test.h5'
    assert str(r0_to_dl1_filename('test.simtel')) == 'dl1_test.h5'
    assert str(r0_to_dl1_filename('test.test.simtel.gz')) == 'dl1_test.test.h5'
    assert str(r0_to_dl1_filename('LST1_custom.fits.fz')) == 'dl1_LST1_custom.h5'
    assert str(r0_to_dl1_filename('LST1_custom.test.fits.fz')) == 'dl1_LST1_custom.test.h5'
