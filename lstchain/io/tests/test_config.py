import tempfile
from lstchain.io import config
from lstchain.io.config import get_cleaning_parameters, includes_image_modification


def test_get_standard_config():
    std_cfg = config.get_standard_config()
    assert 'source_config' in std_cfg
    assert 'tailcut' in std_cfg


def test_get_srcdep_config():
    srcdep_config = config.get_srcdep_config()
    assert 'tailcut' in srcdep_config
    assert srcdep_config['source_dependent']
    assert srcdep_config['mc_nominal_source_x_deg'] == 0.4
    assert srcdep_config['mc_nominal_source_y_deg'] == 0
    assert srcdep_config['observation_mode'] == 'wobble'
    assert srcdep_config['n_off_wobble'] == 1
    assert srcdep_config['train_gamma_src_r_deg'] == [0, 1]


def test_get_mc_config():
    mc_cfg = config.get_mc_config()
    assert 'tailcut' in mc_cfg
    assert mc_cfg['LocalPeakWindowSum']['apply_integration_correction']
    assert mc_cfg['GlobalPeakWindowSum']['apply_integration_correction']


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


def test_dump_config():
    cfg = {'myconf': 1}
    with tempfile.NamedTemporaryFile() as file:
        config.dump_config(cfg, file.name, overwrite=True)
        read_cfg = config.read_configuration_file(file.name)
        assert read_cfg['myconf'] == 1


def test_includes_image_modification_no_modif():
    cfg = {}
    assert not includes_image_modification(cfg)
    cfg = {"image_modifier": {}}
    assert not includes_image_modification(cfg)


def test_includes_image_modification_with_modif():
    cfg = {"image_modifier": {"increase_psf": True, "increase_nsb": False}}
    assert includes_image_modification(cfg)
    cfg = {"image_modifier": {"increase_nsb": True, "increase_psf": False}}
    assert includes_image_modification(cfg)
    cfg = {"image_modifier": {"increase_psf": True, "increase_nsb": True}}
    assert includes_image_modification(cfg)
    cfg = {"image_modifier": {"increase_psf": False, "increase_nsb": False}}
    assert not includes_image_modification(cfg)
