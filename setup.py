#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst
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
    "nbsphinx",
    "sphinxcontrib-mermaid"
]

setup(
    use_scm_version={"write_to": os.path.join("lstchain", "_version.py")},
    packages=find_packages(exclude="lstchain._dev_version"),
    install_requires=[
        'astropy~=4.2',
        'bokeh~=1.0',
        'ctapipe~=0.12.0',
        'ctapipe_io_lst~=0.18.2',
        'ctaplot~=0.6.2',
        'eventio>=1.9.1,<2.0.0a0',  # at least 1.1.1, but not 2
        'gammapy~=0.19.0',
        'h5py',
        'iminuit>=2',
        'joblib',
        'matplotlib~=3.5',
        'numba',
        'numpy<1.22.0a0',
        'pandas',
        'protobuf~=3.20.0',
        'pyirf~=0.6.0',
        'scipy',
        'seaborn',
        'scikit-learn~=1.0',
        'tables',
        'toml',
        'pymongo',
        'pyparsing',
        'setuptools_scm',
        'jinja2~=3.0.2',  # pinned for bokeh 1.0 compatibility
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
