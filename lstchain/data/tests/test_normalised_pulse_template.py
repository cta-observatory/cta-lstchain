import numpy as np
from lstchain.data.normalised_pulse_template import NormalizedPulseTemplate


def test_load_from_file_and_miscellaneous(tmp_path):
    path = tmp_path / "tmp_pulse.txt"
    with open(path, 'w') as f:
        f.write('0 1\n1 2\n2 3\n3 0\n4 0')
    template = NormalizedPulseTemplate.load_from_file(path)
    with open(path, 'w') as f:
        f.write('0 1 5\n1 2 6\n2 3 7\n3 0 0\n4 0 0')
    template = NormalizedPulseTemplate.load_from_file(path)
    with open(path, 'w') as f:
        f.write('0 0.1 0 0.002 0.01\n1 0.7 0.8 0.005 0.02\n'
                '2 0.2 0.2 0.01 0.04\n3 0 0 0 0\n4 0 0 0 0')
    template = NormalizedPulseTemplate.load_from_file(path)
    assert np.isclose(template(np.array([1.5]), 'HG'), 0.45, rtol=0.1)
    assert np.isclose(template.get_error(1.5, 'LG'), 0.03, rtol=0.2)
    assert np.isclose(template.compute_time_of_max(), 1.0, rtol=0.1)
    save_path = tmp_path / "tmp_pulse2.txt"
    template.save(save_path)
    template = NormalizedPulseTemplate.load_from_file(path, resample=True)
    template(np.array([0.2, 0.8, 2, 3.5]), 'HG')
    template = NormalizedPulseTemplate.load_from_file(path, resample=True, dt=0.5)
