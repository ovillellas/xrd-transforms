#
# This will allow to easily adapt tests in case the module name changes.
#
# The tests are meant to be kept independent so that they don't need to be
# installed.
#

from __future__ import absolute_import


import xrd_transforms as xf
from xrd_transforms import xf_numpy
from xrd_transforms import xf_capi
from xrd_transforms import xf_numba
