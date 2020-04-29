import pytest


def test_parse_dl1():
    from lstchain.paths import parse_dl1_filename

    run = parse_dl1_filename('dl1_LST-1.1.Run01920.0000.fits.h5')
    assert run.tel_id == 1
    assert run.stream == 1
    assert run.run == 1920
    assert run.subrun == 0

    run = parse_dl1_filename('dl1_LST-1.Run01920.0000.fits.h5')
    assert run.tel_id == 1
    assert run.stream is None
    assert run.run == 1920
    assert run.subrun == 0

    run = parse_dl1_filename('dl1_LST-1.Run01920.0000.h5')
    assert run.tel_id == 1
    assert run.stream is None
    assert run.run == 1920
    assert run.subrun == 0

    run = parse_dl1_filename('dl1_LST-1.Run01920.0000.fits.hdf5')
    assert run.tel_id == 1
    assert run.stream is None
    assert run.run == 1920
    assert run.subrun == 0

    with pytest.raises(ValueError):
        run = parse_dl1_filename('foo.fits.fz')


def test_dl1_to_filename():
    from lstchain.paths import Run, dl1_run_to_filename

    assert dl1_run_to_filename(Run(1, None, 1920, 2)) == 'dl1_LST-1.Run01920.0002.h5'
    assert dl1_run_to_filename(Run(1, 2, 1920, 3)) == 'dl1_LST-1.2.Run01920.0003.h5'


def test_parse_r0():
    from lstchain.paths import parse_r0_filename

    run = parse_r0_filename('LST-1.1.Run01920.0000.fits.fz')
    assert run.tel_id == 1
    assert run.stream == 1
    assert run.run == 1920
    assert run.subrun == 0

    with pytest.raises(ValueError):
        run = parse_r0_filename('foo.fits.fz')


def test_r0_to_filename():
    from lstchain.paths import Run, r0_run_to_filename

    assert r0_run_to_filename(Run(1, 2, 1920, 3)) == 'LST-1.2.Run01920.0003.fits.fz'
