import pytest
import os
from pathlib import Path


test_data = Path(os.getenv('LSTCHAIN_TEST_DATA', 'test_data'))
test_r0_path = test_data / 'real/R0/'
test_subrun = test_r0_path / '20200218/LST-1.1.Run02008.0000_first50.fits.fz'


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
