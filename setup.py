#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst
# import sys
from setuptools import setup, find_packages
import os


def find_scripts(script_dir, prefix):
    script_list = [
        os.path.splitext(f)[0] for f in os.listdir(script_dir) if f.startswith(prefix)
    ]
    script_dir = script_dir.replace("/", ".")
    point_list = []

    for f in script_list:
        point_list.append(f"{f} = {script_dir}.{f}:main")

    return point_list


lstchain_list = find_scripts("lstchain/scripts", "lstchain_")
onsite_list = find_scripts("lstchain/scripts/onsite", "onsite_")
tools_list = find_scripts("lstchain/tools", "lstchain_")

entry_points = {}
entry_points["console_scripts"] = lstchain_list + onsite_list + tools_list

tests_require = ["pytest"]
docs_require = [
    "sphinx~=4.2",
    "sphinx-automodapi",
    "sphinx_argparse",
    "sphinx_rtd_theme",
    "numpydoc",
    "nbsphinx"
]

setup(
    use_scm_version={"write_to": os.path.join("lstchain", "_version.py")},
    packages=find_packages(),
    install_requires=[
        'astropy~=4.2',
        'bokeh~=1.0',
        'ctapipe~=0.11.0',
        'ctapipe_io_lst~=0.13.2',
        'ctaplot~=0.5.5',
        'eventio>=1.5.1,<2.0.0a0',  # at least 1.1.1, but not 2
        'gammapy~=0.18.2',
        'h5py',
        'joblib',
        'matplotlib~=3.4.3',  # 3.5 is incompatible with gammapy 0.18.2
        'pyparsing~=2.4',     # fixes issue with matplotlib math in mpl < 3.5
        'numba',
        'numpy<1.22.0a0',
        'pandas',
        'pyirf~=0.5.0',
        'scipy',
        'seaborn',
        'scikit-learn',
        'tables',
        'toml',
        'traitlets~=5.0.5',
        'iminuit~=1.5',
        'pymongo',
        'pyparsing',
        'setuptools_scm',
    ],
    extras_require={
        "all": tests_require + docs_require,
        "tests": tests_require,
        "docs": docs_require,
    },
    package_data={
        'lstchain': [
            'data/*',
            'resources/*',
        ],
    },
    entry_points=entry_points,
)
