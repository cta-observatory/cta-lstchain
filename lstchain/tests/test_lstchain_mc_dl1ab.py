import pytest
from lstchain.tests.test_lstchain import dl1_file, test_dir
import os

@pytest.mark.run(after='test_dl0_to_dl1')
def test_mc_dl1ab():
    output_file = os.path.join(test_dir, 'dl1ab.h5')
    cmd = 'lstchain_mc_dl1ab.py {} {}'.format(dl1_file, output_file)
    os.system(cmd)
    assert os.path.exists(output_file)