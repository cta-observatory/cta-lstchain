#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst
# import sys
import setuptools
import lstchain
import os


def find_scripts(script_dir, prefix):
    script_list = [f'{os.path.splitext(f)[0]}' for f in os.listdir(script_dir) if f.startswith(prefix)]
    script_dir = script_dir.replace('/', '.')
    point_list = []
    for f in script_list:
        point_list.append(f"{f} = {script_dir}.{f}:main")
    print(point_list)
    return point_list

lstchain_list = find_scripts('lstchain/scripts','lstchain_')
onsite_list = find_scripts('lstchain/scripts/onsite', 'onsite_')
tools_list = find_scripts('lstchain/tools', 'lstchain_')

entry_points = {}
entry_points['console_scripts'] = lstchain_list + onsite_list + tools_list

setuptools.setup(name='lstchain',
                 version=lstchain.__version__,
                 description="DESCRIPTION",  # these should be minimum list of what is needed to run
                 packages=setuptools.find_packages(),
                 install_requires=['h5py',
                                   'seaborn'
                                   ],
                 package_data={'lstchain': ['data/lstchain_standard_config.json']},
                 tests_require=['pytest', 'pytest-ordering'],
                 author='LST collaboration',
                 author_email='',
                 license='',
                 url='https://github.com/cta-observatory/cta-lstchain',
                 long_description='',
                 classifiers=[],
                 entry_points=entry_points
                 )
