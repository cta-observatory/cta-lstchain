import json
import os


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
