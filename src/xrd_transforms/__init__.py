"""Transforms module.

Contains different implementations based on Python+Numpy, numba and
a supporting C module. All three should adhere to the same interface,
but performance will vary.

Use the functions under this module scope to use the preferred versions,
import the specific submodule if you want to use a specific version.
"""
from __future__ import absolute_import

from collections import OrderedDict

from .transforms_definitions import API
from . import xf_numpy as numpy
from . import xf_capi as capi
from . import xf_new_capi as new_capi
try:
    from . import xf_numba as numba
except ImportError:
    numba = None
    pass

implementations=OrderedDict()
implementations["numpy"] = numpy
implementations["capi"] = capi
implementations["new_capi"] = new_capi

if numba is not None:
    implementations["numba"] = numba

# assign default implementations of functions
_default_implementations = { function: getattr(numpy, function)
                             for function in API }

# overrides can go here.
_default_implementations['rotate_vecs_about_axis'] = new_capi.rotate_vecs_about_axis
globals().update(_default_implementations)
del _default_implementations

