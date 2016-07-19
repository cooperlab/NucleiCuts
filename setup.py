#! /usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import os
from pkg_resources import parse_requirements, RequirementParseError

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('LICENSE') as f:
    license_str = f.read()

try:
    with open('requirements.txt') as f:
        ireqs = parse_requirements(f.read())
except RequirementParseError:
    raise
requirements = [str(req) for req in ireqs]

# if not on ReadTheDocs then add requirements depending on C libraries
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

if not on_rtd:

    requirements_c_files = ['requirements_c_conda.txt',
                            'requirements_c.txt']

    for reqfile in requirements_c_files:
        try:
            with open(reqfile) as f:
                ireqs_c = parse_requirements(f.read())
        except RequirementParseError:
            raise
        cur_requirements = [str(req) for req in ireqs_c]
        requirements += cur_requirements


setup(name='nucleicuts',
      version='0.1.0',
      description='Graph-cut algorithm for nuclei segmentation',
      author='Kitware, Inc.',
      author_email='lee.cooper@emory.edu',
      url='https://github.com/cooperlab/NucleiCuts',
      packages=['nucleicuts'],
      package_dir={'nucleicuts': 'nucleicuts'},
      include_package_data=True,
      install_requires=requirements,
      license=license_str,
      zip_safe=False,
      keywords='nucleicuts',
      classifiers=[
          'Development Status :: 2 - Pre-Alpha',
          'Environment :: Console',
          'License :: OSI Approved :: Apache Software License',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 2',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Software Development :: Libraries :: Python Modules',
      ],
)

