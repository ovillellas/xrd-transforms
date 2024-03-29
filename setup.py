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

new_transforms_capi_extension = Extension(
    'xrd_transforms._new_transforms_capi',
    sources=['src/c-module/new_transforms_capi/module.c'],
    include_dirs=[numpy.get_include()],
    )

setup(
    ext_modules = [transforms_capi_extension,
                   new_transforms_capi_extension],

    include_package_data = True,
    zip_safe = False,    
)
