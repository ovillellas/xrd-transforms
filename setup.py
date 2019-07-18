# ============================================================
# Copyright (c) 2012, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by Joel Bernier <bernier2@llnl.gov> and others.
# LLNL-CODE-529294.
# All rights reserved.
#
# This file is part of HEXRD. For details on dowloading the source,
# see the file COPYING.
#
# Please also see the file LICENSE.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the terms and conditions of the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this program (see file LICENSE); if not, write to
# the Free Software Foundation, Inc., 59 Temple Place, Suite 330,
# Boston, MA 02111-1307 USA or visit <http://www.gnu.org/licenses/>.
# ============================================================
from __future__ import print_function

import glob
import os
import sys

import numpy

from setuptools import Command, Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

import versioneer

cmdclass = versioneer.get_cmdclass()

versioneer.VCS = 'git'
versioneer.style = 'default'
versioneer.versionfile_source = 'xrd_transforms/_version.py'
versioneer.versionfile_build = 'xrd_transforms/_version.py'
versioneer.tag_prefix = 'xrd-transforms-v'
versioneer.parentdir_prefix = 'xrd-transforms-v'



################################################################################
# Dependencies
################################################################################
base_dependencies = [
    'numpy',
    ]

# this are not required to run the packages, but are included to document its
# potential usage
optional_dependencies = [
    'numba', # some functions may be implemented in numba if available.
    'pytest', # in order to run tests, use pytest.
    ]

class test(Command):

    """Run the test suite."""

    description = "Run the test suite"

    user_options = [('verbosity', 'v', 'set test report verbosity')]

    def initialize_options(self):
        self.verbosity = 0

    def finalize_options(self):
        try:
            self.verbosity = int(self.verbosity)
        except ValueError:
            raise ValueError('Verbosity must be an integer.')

    def run(self):
        import unittest
        suite = unittest.TestLoader().discover('test')
        unittest.TextTestRunner(verbosity=self.verbosity+1).run(suite)


cmdclass['test'] = test

transforms_capi_extension = Extension(
    'xrd_transforms._transforms_CAPI',
    sources=glob.glob('src/*.c'),
    include_dirs=[numpy.get_include()],
    # extra_compile_args=['-std=99'],
    )

_version = versioneer.get_version()

setup(
    name = 'xrd_transforms',
    version = versioneer.get_version(True),
    author = 'The HEXRD Development Team',
    author_email = 'praxes@googlegroups.com',
    description = 'xrd transform utilities',
    long_description = open('README.md').read(),
    license = 'LGPLv2',
    url = 'http://xrd_transforms.readthedocs.org',

    install_requires = base_dependencies,

    ext_modules = [transforms_capi_extension],
    packages = find_packages(),

    classifiers = [
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Lesser General Public License v2 (LGPLv2)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        ],
    cmdclass = cmdclass,
    )
