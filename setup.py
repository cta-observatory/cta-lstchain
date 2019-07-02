#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst
# import sys
import setuptools


setuptools.setup(name='lstchain',
                 version=0.1,
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
                 )
