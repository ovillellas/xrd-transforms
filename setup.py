#! /usr/bin/env python

# -*- encoding: utf-8 -*-

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

from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import glob

import numpy

from setuptools import Command, Extension, find_packages, setup


################################################################################
# versioneer is used to base the generate the version using the tag name in git.
# Just use a tag named xrd-transforms-v<major>.<minor>.<release>[.dev<devrel>]
################################################################################
import versioneer

cmdclass = versioneer.get_cmdclass()

versioneer.VCS = 'git'
versioneer.style = 'default'
versioneer.versionfile_source = 'src/xrd_transforms/_version.py'
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


transforms_capi_extension = Extension(
    'xrd_transforms._transforms_CAPI',
    sources=glob.glob('src/c-module/_transforms_CAPI/*.c'),
    include_dirs=[numpy.get_include()],
    # extra_compile_args=['-std=99'],
    )

_version = versioneer.get_version()

setup(
    name = 'xrd_transforms',
    version = versioneer.get_version(True),
    license = 'LGPLv2',

    description = 'xrd transform utilities',
    long_description = open('README.md').read(),
    long_description_content_type = 'text/markdown',

    author = 'The HEXRD Development Team',
    author_email = 'praxes@googlegroups.com',
    url = 'http://xrd_transforms.readthedocs.org',

    ext_modules = [transforms_capi_extension],
    packages = find_packages(where='src/', ),
    package_dir = { '': 'src'},

    include_package_data = True,
    zip_safe = False,
    
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

    keywords=[
    ],

    install_requires = base_dependencies,

    extras_requires = {
    },

    entry_points = {
    },
    
    cmdclass = cmdclass,
)
