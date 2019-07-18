=====================================
 The design of the transforms module
=====================================

The transforms module provides the mathematical support for the xray
crystallography. Due to its evolution, and to meet its performance
goals there've been several implementations with different
characteristics:

- Original NumPy version: An original version implemented using Python
  NumPy that serves as a reference.

- 'CAPI' version: A Python module written in C provides an alternative
  implementation that is faster.

- 'numba' version: A version that uses numba to get close-to C speed
  and that can potentially target CUDA.

All this appeared in a rather organic way. Originally much of the code
existed and evolved in a way that made it diverge. So a "clean-up" was
performed and it was decided to keep all versions, but provide means
to enforce the non-divergence.


NumPy, CAPI and numba functions
===============================

Every function has potentially three different versions (maybe more
in the future). Not all versions are required though.

The NumPy version acts as reference implementation, while the CAPI and
numba versions, in case they exists, should be interchangeable with
the CAPI versions.

Physically, the different versions reside in different python
submodules that can be imported on their own. These are:

- transforms.xf_numpy: for the numpy reference implementations.
- transforms.xf_capi: for the capi implementations.
- transforms.xf_numba: for the numba implementations.

To enforce that all functions follow the same interface, the interface
for the transforms API is established in a definitions file called
"transforms_definitions.py".

A decorator, xf_api, is provided to mark implementations of an
API. The decorator performs some operations and sanity checks on the
decorated function:

1. Checks that the function actually exists in the API (that is, there
   is a definition class for the function).

2. Checks that the definition is correct.

3. Checks that the implementation function adheres to the same
   interfaced defined in the definition class (it has the expected
   argument signature)

4. Patches the docstring of the definition class as the function
   docstring. The definition class docstring acts as the documentation
   for the API. This avoids the same documentation to be in all
   implementation files, removing any risk of divergence in their
   documentation.

5. Optionally, it is possible to add run-time checks for some PRE and
   POST conditions on the API functions. This is also handled by the
   decorator. The run-time checks are disabled by default but can be
   enabled by setting the XRD_TRANSFORMS_CHECK environment variable.

   
Implementation selection and default implementations
====================================================
   
When importing the transforms module, it exports symbols for each API
function in that function that are the "recommended" version to use
by default. The functions are selected in the __init__.py file by
mapping to the desired implementation function.

It is possible (and supported) to force a given implementation by
importing its symbol from the implementation file.

For example:

transforms.gvec_to_xy will use the default "gvec_to_xy".
transforms.xf_numpy.gvec_to_xy will use the Numpy version.
transforms.xf_capi.gvec_to_xy will use the C module version.
transforms.xf_numba.gvec_to_xy will use the numba version.


The API definition
==================

Each function in the transforms API has an entry in the definitions
file ("transforms_definitions.py"). This comes in the form of a class
named DEF_<api_function>, that subclases DEF_Func.

The class has the following characteristics:

1. It is named DEF_<api_function>. Any implementation file has to be
   named <api_function> that is the actual name of the function in the
   API. Currently the xf_api decorator enforces this.
   
2. The class docstring will become the docstring for every
   implementation of the API function. The xf_api decorator will patch
   the exposed function docstring to use that docstring. It is
   recommended that the actual implementation doesn't provide a
   docstring on its own.

3. A method called _signature that has the same signature (arguments
   and keyword arguments) as the API function. This is just a
   convenient way to specify the signature. The method is never called
   and its implementation should be a mere "pass".

4. Optionally, a _PRECOND and _POSTCOND pair of methods. These
   implement the pre and post conditions for the API. If run-time
   checks are enabled, they will be called before and after the API
   function, respectively. _PRECOND will take the function's
   positional and keyword arguments, while _POSTCOND will take the
   function's result, positional and keyword arguments. The code
   should check any restriction imposed in the API and raise an error
   in case it is not met.


API testing
===========

Every function in the API has its set of unittests on its own file.
Each test file is named after the API function it tests. Tests are
implemented using pytest. All tests are run on all the supported
implementations. The default implementation is tested as if it was
a different implementation altogether.

In order to check all implementations pytest.mark.parametrize is used
and the tests must be written supporting this. Typically an
"all_impls" decorator is defined at the beginning of the test file
that will define the parametrize options to use.


A note about the CAPI implementation
====================================

As of now, the C module implementing the API is pretty bare-bones and
has little to no error checking/handling. It also assumes that arrays
come as NumPy arrays arranged in C order.

Due to this, the CAPI calls are in fact calls to a set of wrapper
functions that curates the arguments to avoid bad behavior. Do not
call the C module functions directly.

