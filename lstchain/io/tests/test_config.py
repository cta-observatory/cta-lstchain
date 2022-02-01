from lstchain.io import config
from lstchain.io.config import get_cleaning_parameters

def test_get_standard_config():
    config.get_standard_config()

def test_get_srcdep_config():
    srcdep_config = config.get_srcdep_config()
    assert srcdep_config['source_dependent'] == True
    assert srcdep_config['mc_nominal_source_x_deg'] == 0.4
    assert srcdep_config['observation_mode'] == 'wobble'
    assert srcdep_config['n_off_wobble'] == 3

def test_replace_config():
    a = dict(toto=1, tata=2)
    b = dict(tata=3, tutu=4)
    c = config.replace_config(a, b)
    assert c["toto"] == 1
    assert c["tata"] == 3
    assert c["tutu"] == 4

def test_get_cleaning_parameters():
    std_config = config.get_standard_config()
    cleaning_params = get_cleaning_parameters(std_config, 'tailcut')
    picture_th, boundary_th, isolated_pixels, min_n_neighbors = cleaning_params
    assert std_config['tailcut']['picture_thresh'] == picture_th
    assert std_config['tailcut']['boundary_thresh'] == boundary_th
    assert std_config['tailcut']['keep_isolated_pixels'] == isolated_pixels
    assert std_config['tailcut']['min_number_picture_neighbors'] == min_n_neighbors
