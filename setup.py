#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst
# import sys
from setuptools import setup, find_packages
import os
import sys

# pep 517 builds do not have pwd in PATH
sys.path.insert(0, os.path.dirname(__file__))
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
    py_modules='version',
    install_requires=[
        'astropy',
        'ctapipe',
        'gammapy>=0.17',
        'h5py',
        'numba',
        'numpy',
        'pandas',
        'scipy',
        'seaborn',
        'tables',
    ],
    package_data={
      'lstchain': ['data/lstchain_standard_config.json']
    },
    tests_require=[
      'pytest',
      'pytest-ordering',
    ],
    entry_points=entry_points
)
