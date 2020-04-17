import os
import pytest
from lstchain.tests.test_lstchain import test_dir, mc_gamma_testfile
import pandas as pd
from lstchain.io.io import dl1_params_lstcam_key
import numpy as np

output_dir = os.path.join(test_dir, 'scripts')
dl1_file = os.path.join(output_dir, 'dl1_gamma_test_large.simtel.h5')
dl2_file = os.path.join(output_dir, 'dl2_dl1_gamma_test_large.simtel.h5')
file_model_energy = os.path.join(output_dir, 'reg_energy.sav')
file_model_disp = os.path.join(output_dir, 'reg_disp_vector.sav')
file_model_gh_sep = os.path.join(output_dir, 'cls_gh.sav')

def test_lstchain_mc_r0_to_dl1():
    input_file = mc_gamma_testfile
    cmd = f'lstchain_mc_r0_to_dl1 -f {input_file} -o {output_dir}'
    os.system(cmd)
    assert os.path.exists(dl1_file)


@pytest.mark.run(after='test_lstchain_mc_r0_to_dl1')
def test_lstchain_trainpipe():
    gamma_file = dl1_file
    proton_file = dl1_file
    cmd = f'lstchain_mc_trainpipe -fg {gamma_file} -fp {proton_file} -o {output_dir}'
    os.system(cmd)
    assert os.path.exists(file_model_gh_sep)
    assert os.path.exists(file_model_disp)
    assert os.path.exists(file_model_energy)


@pytest.mark.run(after='test_lstchain_trainpipe')
def test_lstchain_dl1_to_dl2():
    cmd = f'lstchain_dl1_to_dl2 -f {dl1_file} -p {output_dir} -o {output_dir}'
    os.system(cmd)
    assert os.path.exists(dl2_file)


@pytest.mark.run(after='test_lstchain_mc_r0_to_dl1')
def test_mc_dl1ab():
    output_file = os.path.join(output_dir, 'dl1ab.h5')
    cmd = 'lstchain_mc_dl1ab {} {}'.format(dl1_file, output_file)
    os.system(cmd)
    assert os.path.exists(output_file)

@pytest.mark.run(after='test_mc_dl1ab')
def test_mc_dl1ab_validity():
    dl1 = pd.read_hdf(dl1_file, key=dl1_params_lstcam_key)
    dl1ab = pd.read_hdf(os.path.join(output_dir, 'dl1ab.h5'), key=dl1_params_lstcam_key)
    np.testing.assert_allclose(dl1.to_numpy(), dl1ab.to_numpy())


@pytest.mark.run(after='test_lstchain_dl1_to_dl2')
def test_mc_r0_to_dl2():
    cmd = f'lstchain_mc_r0_to_dl2 -f {mc_gamma_testfile} -p {output_dir} -s1 False -o {output_dir}'
    os.remove(dl1_file)
    os.remove(dl2_file)
    os.system(cmd)
    # output_file = os.path.join(output_dir, 'dl2_' + os.path.basename(mc_gamma_testfile))
    assert os.path.exists(dl2_file)
