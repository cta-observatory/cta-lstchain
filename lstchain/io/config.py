import json
import os
from copy import copy

__all__ = [
    'get_cleaning_parameters',
    'get_standard_config',
    'get_srcdep_config',
    'read_configuration_file',
    'replace_config',
]


def read_configuration_file(config_filename):
    """
    Read a configuration filename for the lstchain.

    Parameters
    ----------
    config_filename: str

    Returns
    -------
    Dictionnary
    """
    assert os.path.exists(config_filename)

    with open(config_filename) as json_file:
        data = json.load(json_file)

    return data


def get_standard_config():
    """
    Load the standard config from the file 'data/lstchain_standard_config.json'

    Returns
    -------
    dict
    """
    standard_config_file = os.path.join(os.path.dirname(__file__), "../data/lstchain_standard_config.json")
    return read_configuration_file(standard_config_file)

def get_srcdep_config():
    """
    Load the config for source-dependent analysis from the file 'data/lstchain_src_dep_config.json'

    Returns
    -------
    dict
    """
    srcdep_config_file = os.path.join(os.path.dirname(__file__), "../data/lstchain_src_dep_config.json")
    return read_configuration_file(srcdep_config_file)

def replace_config(base_config, new_config):
    """
    Return a copy of the base_config with new configuration from new_config

    Parameters
    ----------
    base_config: dict
    new_config: dict

    Returns
    -------
    dict
    """
    config = copy(base_config)

    for k in new_config.keys():
        config[k] = new_config[k]

    return config


def get_cleaning_parameters(config, clean_method_name):
    """
    Return cleaning parameters from configuration dict.

    Parameters
    ----------
    config: configuration dict
    clean_method_name: name of cleaning method

    Returns
    -------
    tuple (picture threshold, boundary threshold, keep isolated pixels, min number picture neighbors)
    """
    picture_th = config[clean_method_name]['picture_thresh']
    boundary_th = config[clean_method_name]['boundary_thresh']
    isolated_pixels = config[clean_method_name]['keep_isolated_pixels']
    min_n_picture_neighbors = config[clean_method_name]['min_number_picture_neighbors']
    return picture_th, boundary_th, isolated_pixels, min_n_picture_neighbors
