
#if !defined(XRD_SINGLE_COMPILE_UNIT) || !XRD_SINGLE_COMPILE_UNIT
#  include "transforms_utils.h"
#  include "transforms_prototypes.h"
#endif

XRD_CFUNCTION void
make_sample_rmat(double chi, double ome, double * restrict rPtr)
{
  double c[2],s[2];

  c[0] = cos(chi);
  s[0] = sin(chi);
  c[1] = cos(ome);
  s[1] = sin(ome);

  rPtr[0] =  c[1];
  rPtr[1] =  0.0;
  rPtr[2] =  s[1];
  rPtr[3] =  s[0]*s[1];
  rPtr[4] =  c[0];
  rPtr[5] = -s[0]*c[1];
  rPtr[6] = -c[0]*s[1];
  rPtr[7] =  s[0];
  rPtr[8] =  c[0]*c[1];
}


XRD_CFUNCTION void
make_sample_rmat_polar(double sin_chi, double cos_chi,
                       double sin_ome, double cos_ome,
                       double * restrict rPtr)
{
    rPtr[0] =  cos_ome;
    rPtr[1] =  0.0;
    rPtr[2] =  sin_ome;
    rPtr[3] =  sin_chi * sin_ome;
    rPtr[4] =  cos_chi;
    rPtr[5] = -sin_chi * cos_ome;
    rPtr[6] = -cos_chi * sin_ome;
    rPtr[7] =  sin_chi;
    rPtr[8] =  cos_chi * cos_ome;
}

XRD_CFUNCTION void
make_sample_rmat_polar_f(float sin_chi, float cos_chi,
                         float sin_ome, float cos_ome,
                         float * restrict rPtr)
{
    rPtr[0] =  cos_ome;
    rPtr[1] =  0.0;
    rPtr[2] =  sin_ome;
    rPtr[3] =  sin_chi * sin_ome;
    rPtr[4] =  cos_chi;
    rPtr[5] = -sin_chi * cos_ome;
    rPtr[6] = -cos_chi * sin_ome;
    rPtr[7] =  sin_chi;
    rPtr[8] =  cos_chi * cos_ome;
}

XRD_CFUNCTION void
make_sample_rmat_t_polar_f(float sin_chi, float cos_chi,
                         float sin_ome, float cos_ome,
                         float * restrict rPtr)
{
    rPtr[0] =  cos_ome;
    rPtr[1] =  sin_chi * sin_ome;
    rPtr[2] = -cos_chi * sin_ome;
    rPtr[3] =  0.0;
    rPtr[4] =  cos_chi;
    rPtr[5] =  sin_chi;
    rPtr[6] =  sin_ome;
    rPtr[7] = -sin_chi * cos_ome;
    rPtr[8] =  cos_chi * cos_ome;
}


#if defined(XRD_INCLUDE_PYTHON_WRAPPERS) && XRD_INCLUDE_PYTHON_WRAPPERS

#  if !defined(XRD_SINGLE_COMPILE_UNIT) || !XRD_SINGLE_COMPILE_UNIT
#    define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#    include <Python.h>
#    include <numpy/arrayobject.h>
#  endif /* XRD_SINGLE_COMPILE_UNIT */

XRD_PYTHON_WRAPPER const char *docstring_makeOscillRotMat =
    "c module implementation of make_sample_rotmat (single).\n"
    "Please use the Python wrapper.\n";

XRD_PYTHON_WRAPPER const char *docstring_makeOscillRotMatArray =
    "c module implementation of make_sample_rotmat (multiple).\n"
    "Please use the Python wrapper.\n";

XRD_PYTHON_WRAPPER PyObject *
python_makeOscillRotMat(PyObject * self, PyObject * args)
{
    PyArrayObject *oscillAngles, *rMat;
    int doa;
    npy_intp no, dims[2];
    double *oPtr, *rPtr;

    /* Parse arguments */
    if ( !PyArg_ParseTuple(args,"O", &oscillAngles)) return(NULL);
    if ( oscillAngles == NULL ) return(NULL);

    /* Verify shape of input arrays */
    doa = PyArray_NDIM(oscillAngles);
    assert( doa == 1 );

    /* Verify dimensions of input arrays */
    no = PyArray_DIMS(oscillAngles)[0];
    assert( no == 2 );

    /* Allocate the result matrix with appropriate dimensions and type */
    dims[0] = 3; dims[1] = 3;
    rMat = (PyArrayObject*)PyArray_EMPTY(2,dims,NPY_DOUBLE,0);

    /* Grab pointers to the various data arrays */
    oPtr = (double*)PyArray_DATA(oscillAngles);
    rPtr = (double*)PyArray_DATA(rMat);

    /* Call the actual function */
    make_sample_rmat(oPtr[0], oPtr[1], rPtr);

    return((PyObject*)rMat);
}

XRD_PYTHON_WRAPPER PyObject *
python_makeOscillRotMatArray(PyObject * self, PyObject * args)
{
    PyObject *chiObj;
    double chi;
    PyArrayObject *omeArray, *rMat;
    int doa;
    npy_intp i, no, dims[3];
    double *oPtr, *rPtr;

    /* Parse arguments */
    if ( !PyArg_ParseTuple(args,"OO", &chiObj, &omeArray)) return(NULL);
    if ( chiObj == NULL ) return(NULL);
    if ( omeArray == NULL ) return(NULL);

    /* Get chi */
    chi = PyFloat_AsDouble(chiObj);
    if (chi == -1 && PyErr_Occurred()) return(NULL);

    /* Verify shape of input arrays */
    doa = PyArray_NDIM(omeArray);
    assert( doa == 1 );

    /* Verify dimensions of input arrays */
    no = PyArray_DIMS(omeArray)[0];

    /* Allocate the result matrix with appropriate dimensions and type */
    dims[0] = no; dims[1] = 3; dims[2] = 3;
    rMat = (PyArrayObject*)PyArray_EMPTY(3,dims,NPY_DOUBLE,0);

    /* Grab pointers to the various data arrays */
    oPtr = (double*)PyArray_DATA(omeArray);
    rPtr = (double*)PyArray_DATA(rMat);

    /* Call the actual function repeatedly */
    for (i = 0; i < no; ++i) {
        make_sample_rmat(chi, oPtr[i], rPtr + i*9);
    }

    return((PyObject*)rMat);
}


#endif /* XRD_INCLUDE_PYTHON_WRAPPERS */
