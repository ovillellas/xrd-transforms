
#if !defined(XRD_SINGLE_COMPILE_UNIT) || !XRD_SINGLE_COMPILE_UNIT
#  include "transforms_utils.h"
#  include "transforms_prototypes.h"
#endif


XRD_CFUNCTION int
unit_row_vector(size_t n, double * cIn, double * cOut)
{
    size_t j;
    double nrm;

    nrm = 0.0;
    for (j=0; j<n; j++) {
        nrm += cIn[j]*cIn[j];
    }
    nrm = sqrt(nrm);
    if ( nrm > epsf ) {
        for (j=0; j<n; j++) {
            cOut[j] = cIn[j]/nrm;
        }
        return 0;
    } else {
        for (j=0; j<n; j++) {
            cOut[j] = cIn[j];
        }
        return 1;
    }
}

XRD_CFUNCTION void
unit_row_vectors(size_t m, size_t n, double *cIn, double *cOut)
{
    size_t i, j;
    double nrm;

    for (i=0; i<m; i++) {
        nrm = 0.0;
        for (j=0; j<n; j++) {
            nrm += cIn[n*i+j]*cIn[n*i+j];
        }
        nrm = sqrt(nrm);
        if ( nrm > epsf ) {
            for (j=0; j<n; j++) {
                cOut[n*i+j] = cIn[n*i+j]/nrm;
            }
        } else {
            for (j=0; j<n; j++) {
                cOut[n*i+j] = cIn[n*i+j];
            }
        }
    }
}


#if defined(XRD_INCLUDE_PYTHON_WRAPPERS) && XRD_INCLUDE_PYTHON_WRAPPERS

#  if !defined(XRD_SINGLE_COMPILE_UNIT) || !XRD_SINGLE_COMPILE_UNIT
#    define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#    include <Python.h>
#    include <numpy/arrayobject.h>
#  endif /* XRD_SINGLE_COMPILE_UNIT */

XRD_PYTHON_WRAPPER const char *docstring_unitRowVector =
    "c module implementation of unit_row_vector (one row).\n"
    "Please use the Python wrapper.\n";

XRD_PYTHON_WRAPPER const char *docstring_unitRowVectors =
    "c module implementation of unit_row_vector (multiple rows).\n"
    "Please use the Python wrapper.\n";

XRD_PYTHON_WRAPPER PyObject *
python_unitRowVector(PyObject * self, PyObject * args)
{
    PyArrayObject *vecIn, *vecOut;
    double *cIn, *cOut;
    int d;
    npy_intp n;

    if ( !PyArg_ParseTuple(args,"O", &vecIn)) return(NULL);
    if ( vecIn  == NULL ) return(NULL);

    assert( PyArray_ISCONTIGUOUS(vecIn) );
    assert( PyArray_ISALIGNED(vecIn) );

    d = PyArray_NDIM(vecIn);

    assert(d == 1);

    n = PyArray_DIMS(vecIn)[0];

    vecOut = (PyArrayObject*)PyArray_EMPTY(d,PyArray_DIMS(vecIn),NPY_DOUBLE,0);

    cIn  = (double*)PyArray_DATA(vecIn);
    cOut = (double*)PyArray_DATA(vecOut);

    unit_row_vector(n,cIn,cOut);

    return((PyObject*)vecOut);
}


XRD_PYTHON_WRAPPER PyObject *
python_unitRowVectors(PyObject *self, PyObject *args)
{
    PyArrayObject *vecIn, *vecOut;
    double *cIn, *cOut;
    int d;
    npy_intp m,n;

    if ( !PyArg_ParseTuple(args,"O", &vecIn)) return(NULL);
    if ( vecIn  == NULL ) return(NULL);

    assert( PyArray_ISCONTIGUOUS(vecIn) );
    assert( PyArray_ISALIGNED(vecIn) );

    d = PyArray_NDIM(vecIn);

    assert(d == 2);

    m = PyArray_DIMS(vecIn)[0];
    n = PyArray_DIMS(vecIn)[1];

    vecOut = (PyArrayObject*)PyArray_EMPTY(d,PyArray_DIMS(vecIn),NPY_DOUBLE,0);

    cIn  = (double*)PyArray_DATA(vecIn);
    cOut = (double*)PyArray_DATA(vecOut);

    unit_row_vectors(m,n,cIn,cOut);

    return((PyObject*)vecOut);
}


#endif /* XRD_INCLUDE_PYTHON_WRAPPERS */
