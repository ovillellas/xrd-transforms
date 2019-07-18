
#if !defined(XRD_SINGLE_COMPILE_UNIT) || !XRD_SINGLE_COMPILE_UNIT
#  include "transforms_utils.h"
#  include "transforms_prototypes.h"
#endif


XRD_CFUNCTION int 
make_beam_rmat(double * bPtr, double * ePtr, double * rPtr)
{
    /*
     * This function generates a COB matrix that takes components in the
     * BEAM frame to LAB frame
     *
     * NOTE: the beam and eta vectors MUST NOT BE COLINEAR!!!!
     * 
     * Added a return value acting as an error code. Possible values are:
     * 0 - No error
     * 1 - beam vector is zero within an epsilon
     * 2 - beam and eta vectors are collinear.
     */
    int i, err;
    double yPtr[3], bHat[3], yHat[3], xHat[3];

    err = unit_row_vector(3, bPtr, bHat);
    if (0 != err) {
        /* can't normalize beam vector due to being zero */
        return TF_MAKE_BEAM_RMAT_ERR_BEAM_ZERO;
    }

    /* find Y as e ^ b */
    yPtr[0] = ePtr[1]*bPtr[2] - bPtr[1]*ePtr[2];
    yPtr[1] = ePtr[2]*bPtr[0] - bPtr[2]*ePtr[0];
    yPtr[2] = ePtr[0]*bPtr[1] - bPtr[0]*ePtr[1];

    /* Normalize e ^ b */
    err = unit_row_vector(3, yPtr, yHat);
    if (0 != err) {
        /* e ^ b is close to zero... eta and beam are collinear... */
        return TF_MAKE_BEAM_RMAT_ERR_COLLINEAR;
    }

    /* Find X as b ^ Y */
    xHat[0] = bHat[1]*yHat[2] - yHat[1]*bHat[2];
    xHat[1] = bHat[2]*yHat[0] - yHat[2]*bHat[0];
    xHat[2] = bHat[0]*yHat[1] - yHat[0]*bHat[1];

    /* Assign columns */
    for (i=0; i<3; i++) {
        rPtr[3*i] = xHat[i];
        rPtr[3*i+1] = yHat[i];
        rPtr[3*i+2] = -bHat[i];
    }

    return 0; /* no error */
}


#if defined(XRD_INCLUDE_PYTHON_WRAPPERS) && XRD_INCLUDE_PYTHON_WRAPPERS

#  if !defined(XRD_SINGLE_COMPILE_UNIT) || !XRD_SINGLE_COMPILE_UNIT
#    define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#    include <Python.h>
#    include <numpy/arrayobject.h>
#  endif /* XRD_SINGLE_COMPILE_UNIT */

XRD_PYTHON_WRAPPER const char *docstring_makeEtaFrameRotMat =
    "c module implementation of make_beam_mat.\n"
    "Please use the Python wrapper.\n";

XRD_PYTHON_WRAPPER PyObject *
python_makeEtaFrameRotMat(PyObject * self, PyObject * args)
{
    PyArrayObject *bHat, *eHat, *rMat=NULL;
    int db, de;
    npy_intp nb, ne, dims[2];
    double *bPtr, *ePtr, *rPtr;
    int errcode;

    /* Parse arguments */
    if ( !PyArg_ParseTuple(args,"OO", &bHat,&eHat)) return(NULL);
    if ( bHat  == NULL || eHat == NULL ) return(NULL);

    /* Verify shape of input arrays */
    db = PyArray_NDIM(bHat);
    de = PyArray_NDIM(eHat);
    assert( db == 1 && de == 1);

    /* Verify dimensions of input arrays */
    nb = PyArray_DIMS(bHat)[0];
    ne = PyArray_DIMS(eHat)[0];
    assert( nb == 3 && ne == 3 );

    /* Allocate the result matrix with appropriate dimensions and type */
    dims[0] = 3; dims[1] = 3;
    rMat = (PyArrayObject*)PyArray_EMPTY(2,dims,NPY_DOUBLE,0);

    if (rMat == NULL)
        goto fail;

    /* Grab pointers to the various data arrays */
    bPtr = (double*)PyArray_DATA(bHat);
    ePtr = (double*)PyArray_DATA(eHat);
    rPtr = (double*)PyArray_DATA(rMat);

    /* Call the actual function */
    errcode = make_beam_rmat(bPtr, ePtr, rPtr);

    if (errcode == 0) {
        /* No error, return the matrix */
        return((PyObject*)rMat);
    } /* else ... fall back to fail code, but generating the appropriate error before */

    switch (errcode) {
    case TF_MAKE_BEAM_RMAT_ERR_BEAM_ZERO: /* beam vec is zero (within an epsilon) */
        PyErr_SetString(PyExc_RuntimeError, "bvec_l MUST NOT be ZERO!");
        break;
    case TF_MAKE_BEAM_RMAT_ERR_COLLINEAR: /* beam vec and eta vec are collinear */
        PyErr_SetString(PyExc_RuntimeError, "bvec_l and evec_l MUST NOT be collinear!");
        break;
    }

 fail:
    /* Free the array if allocated, since it won't be returned */
    Py_XDECREF(rMat);
    return NULL;
}


#endif /* XRD_INCLUDE_PYTHON_WRAPPERS */
