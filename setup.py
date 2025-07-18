# Licensed under a 3-clause BSD style license - see LICENSE.rst
from setuptools import setup, find_packages
import os


def find_scripts(script_dir, prefix, suffix='.py'):
    script_list = [
        os.path.splitext(f)[0] for f in os.listdir(script_dir) 
        if (f.startswith(prefix) and f.endswith(suffix))
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
    "sphinx",
    "sphinx-automodapi",
    "sphinx_argparse",
    "sphinx_rtd_theme",
    "numpydoc",
    "nbsphinx",
    "sphinxcontrib-mermaid",
    "sphinx-togglebutton"
]
dev_extra = [
    "setuptools_scm"
]

setup(
    use_scm_version={"write_to": os.path.join("lstchain", "_version.py")},
    packages=find_packages(exclude="lstchain._dev_version"),
    install_requires=[
        'astropy>=6.1,<8',
        'bokeh~=3.0',
        'ctapipe~=0.25.1',
        'ctapipe_io_lst~=0.27.2',
        'ctaplot~=0.6.4',
        'eventio>=1.9.1,<2.0.0a0',  # at least 1.1.1, but not 2
        'gammapy~=1.3.0',
        'h5py',
        'iminuit>=2',
        'joblib~=1.4.0',
        'matplotlib~=3.10.0',
        'numba',
        'numpy',
        'pandas',
        'pyirf~=0.12.0',
        'scipy>=1.8',
        'seaborn',
        'scikit-learn>=1.2',
        'tables',
        'toml',
        'protozfits>=2.7.3,<3',
        'pymongo',
        'pyparsing',
    ],
    extras_require={
        "dev": dev_extra,
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
