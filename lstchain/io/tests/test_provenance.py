import pytest
import h5py
from lstchain.io.provenance import write_provenance, read_provenance
from ctapipe.core import Provenance

@pytest.fixture
def hdf5_file(tmp_path):
    test_file = tmp_path / 'test_provenance.h5'
    yield test_file

def test_write_provenance(hdf5_file):
    stage_name = 'test_stage'
    p = Provenance()
    p.add_input_file('input file', role='test input file')
    p.add_output_file('output file', role='test output file')
    write_provenance(hdf5_file, stage_name)
    with h5py.File(hdf5_file, 'r') as h5file:
        assert 'provenance' in h5file
        assert stage_name in h5file['provenance']


def test_read_provenance(hdf5_file):
    stage_name = 'test_stage'
    write_provenance(hdf5_file, stage_name)
    result = read_provenance(hdf5_file, stage_name)    
    assert 'activity_name' in result
    assert 'activity_uuid' in result

