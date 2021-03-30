#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst
# import sys
from setuptools import setup, find_packages
import os
import sys

# Add lstchain folder to path (contains version.py)
# this is needed as lstchain/__init__.py imports dependencies
# that might not be installed before setup runs, so we cannot import
# lstchain.version
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lstchain'))
from version import get_version, update_release_version  # noqa


update_release_version()
version = get_version()


def find_scripts(script_dir, prefix):
    script_list = [
        os.path.splitext(f)[0]
        for f in os.listdir(script_dir) if f.startswith(prefix)
    ]
    script_dir = script_dir.replace('/', '.')
    point_list = []

    for f in script_list:
        point_list.append(f"{f} = {script_dir}.{f}:main")

    return point_list


lstchain_list = find_scripts('lstchain/scripts', 'lstchain_')
onsite_list = find_scripts('lstchain/scripts/onsite', 'onsite_')
tools_list = find_scripts('lstchain/tools', 'lstchain_')

entry_points = {}
entry_points['console_scripts'] = lstchain_list + onsite_list + tools_list

setup(
    version=version,
    packages=find_packages(),
    install_requires=[
        'astropy~=4.2',
        'ctapipe~=0.10.5',
        'ctapipe_io_lst~=0.9.1',
        'ctaplot~=0.5.5',
        'eventio>=1.5.1,<2.0.0a0',  # at least 1.1.1, but not 2
        'gammapy>=0.18',
        'h5py',
        'joblib',
        'matplotlib',
        'numba',
        'numpy',
        'pandas',
        'pyirf~=0.4.0',
        'scipy',
        'seaborn',
        'scikit-learn',
        'tables',
        'toml',
        'traitlets',
        'iminuit~=1.5',
    ],
    package_data={
        'lstchain': [
            'data/lstchain_standard_config.json',
            'data/onsite_camera_calibration_param.json',
            'resources/LST_pixid_to_cluster.txt',
        ],
    },
    tests_require=[
        'pytest',
        'pytest-ordering',
    ],
    entry_points=entry_points
)
