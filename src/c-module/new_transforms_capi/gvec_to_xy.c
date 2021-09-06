
#if !defined(XRD_SINGLE_COMPILE_UNIT) || !XRD_SINGLE_COMPILE_UNIT
#  include "transforms_utils.h"
#  include "transforms_prototypes.h"
#endif


static void
gvec_to_xy_single(const double *gVec_c, const double *rMat_d, const double *rMat_sc,
                  const double *tVec_d, const double *nbHat_l, const double *nVec_l,
                  const double num, const double *P0_l,
                  double * restrict result)
{
    double bDot, ztol, denom, u;
    double gHat_c[3], gVec_l[3], dVec_l[3], P2_l[3];
    double P2_l_minus_tVec_d[3];
    double brMat[9];

    ztol = epsf;

    /* Compute unit reciprocal lattice vector in crystal frame w/o translation */
    v3_normalize(gVec_c, gHat_c);

    /*
	 * Compute unit reciprocal lattice vector in lab frame
	 * and dot with beam vector
	 */
    m33_v3s_multiply(rMat_sc, gHat_c, 1, gVec_l);
    bDot = v3_v3s_dot(nbHat_l, gVec_l, 1);

    if ( bDot < ztol && bDot > 1.0-ztol )
        goto no_diffraction;

    /* diffraction */
    v3_make_binary_rmat(gVec_l, brMat);

    m33_v3s_multiply(brMat, nbHat_l, 1, dVec_l);
    denom = v3_v3s_dot(nVec_l, dVec_l, 1);

    if ( denom > -ztol )
        goto no_diffraction;

    u = num/denom;

    v3_v3s_muladd(P0_l, dVec_l, 1, u, P2_l);
    v3_v3s_sub(P2_l, tVec_d, 1, P2_l_minus_tVec_d);

    /* P2_l-tVec_d is a point in the detector plane. As we are changing to th
       detector frame, the z coordinate will always be 0, so avoid computing
       it. Note that the result is an xy point (that is, 2d), so using a
       m33_v3_multiply would result in a potential rogue memory write */
    result[0] = v3_v3s_dot(rMat_d, P2_l_minus_tVec_d, 1);
    result[1] = v3_v3s_dot(rMat_d + 3, P2_l_minus_tVec_d, 1);

    return;

 no_diffraction:
    result[0] = NAN;
    result[1] = NAN;
    return;
}

XRD_CFUNCTION void
gvec_to_xy(size_t npts, const double *gVec_c, const double *rMat_d,
           const double *rMat_s, const double *rMat_c, const double *tVec_d,
           const double *tVec_s, const double *tVec_c, const double *beamVec,
           double * restrict result)
{
    size_t i;
    double num;
    double nVec_l[3], bHat_l[3], P0_l[3], tVec_d_s[3], tmp[3];
    double rMat_sc[9];

    /* Normalize the beam vector */
    v3_negate(beamVec, bHat_l);
    v3_inplace_normalize(bHat_l);

    /* compute detector normal in LAB (nVec_l) The normal will just be the Z
       column vector of rMat_d */
    v3s_copy(rMat_d + 2, 3, nVec_l);

    /* tVec_d_s is the translation vector taking from sample to detector */
    v3_v3s_sub(tVec_d, tVec_s, 1, tVec_d_s);

    /* P0_l <= tVec_s + rMat_s x tVec_c */
    m33_v3s_multiply(rMat_s, tVec_c, 1, P0_l);
    v3_v3s_sub(tVec_d_s, P0_l, 1, tmp);
    num = v3_v3s_dot(nVec_l, tmp, 1);

    /* accumulate rMat_s and rMat_c. rMat_sc is a COB Matrix from LAB to
       CRYSTAL */
    m33_m33_multiply(rMat_s, rMat_c, rMat_sc);

    for (i=0L; i<npts; i++) {
        gvec_to_xy_single(&gVec_c[3*i], rMat_d, rMat_sc, tVec_d,
                          bHat_l, nVec_l, num,
                          P0_l, &result[2*i]);
    }
}

/*
 * The only difference between this and the non-Array version
 * is that rMat_s is an array of matrices of length npts instead
 * of a single matrix.
 */
XRD_CFUNCTION void
gvec_to_xy_array(size_t npts, const double *gVec_c, const double *rMat_d,
                 const double *rMat_s, const double *rMat_c, const double *tVec_d,
                 const double *tVec_s, const double *tVec_c, const double *beamVec,
                 double * restrict result)
{
    size_t i;
    double num;
    double nVec_l[3], bHat_l[3], P0_l[3], tVec_d_s[3], tVec_d_c[3];
    double rMat_sc[9];

    /* Normalize the beam vector */
    v3_negate(beamVec, bHat_l);
    v3_inplace_normalize(bHat_l);

    /* compute detector normal in LAB (nVec_l)
       The normal will just be the Z column vector of rMat_d
     */
    v3s_copy(rMat_d + 2, 3, nVec_l);

    /* tVec_d_s is the translation from P2 to P1 (sample to detector) */
    v3_v3s_sub(tVec_d, tVec_s, 1, tVec_d_s);

    for (i=0L; i<npts; i++) {
        /* P0_l <= tVec_s + rMat_s x tVec_c */
        m33_v3s_multiply(rMat_s + 9*i, tVec_c, 1, P0_l);
        v3_v3s_sub(tVec_d_s, P0_l, 1, tVec_d_c);
        num = v3_v3s_dot(nVec_l, tVec_d_c, 1);

        /* Compute the matrix product of rMat_s and rMat_c */
        m33_m33_multiply(rMat_s + 9*i, rMat_c, rMat_sc);

        gvec_to_xy_single(&gVec_c[3*i], rMat_d, rMat_sc, tVec_d,
                          bHat_l, nVec_l, num,
                          P0_l, &result[2*i]);
    }
}

#if defined(XRD_INCLUDE_PYTHON_WRAPPERS) && XRD_INCLUDE_PYTHON_WRAPPERS

#  if !defined(XRD_SINGLE_COMPILE_UNIT) || !XRD_SINGLE_COMPILE_UNIT
#    define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#    include <Python.h>
#    include <numpy/arrayobject.h>
#  endif /* XRD_SINGLE_COMPILE_UNIT */

XRD_PYTHON_WRAPPER const char *docstring_gvecToDetectorXY =
    "c module implementation of gvec_to_xy (single sample).\n"
    "Please use the Python wrapper.\n";

XRD_PYTHON_WRAPPER const char *docstring_gvecToDetectorXYArray =
    "c module implementation of gvec_to_xy (multi sample).\n"
    "Please use the Python wrapper.\n";

/*
  Takes a list of unit reciprocal lattice vectors in crystal frame to the
  specified detector-relative frame, subject to the conditions:

  1) the reciprocal lattice vector must be able to satisfy a bragg condition
  2) the associated diffracted beam must intersect the detector plane

  Required Arguments:
  gVec_c -- (n, 3) ndarray of n reciprocal lattice vectors in the CRYSTAL FRAME
  rMat_d -- (3, 3) ndarray, the COB taking DETECTOR FRAME components to LAB FRAME
  rMat_s -- (3, 3) ndarray, the COB taking SAMPLE FRAME components to LAB FRAME
  rMat_c -- (3, 3) ndarray, the COB taking CRYSTAL FRAME components to SAMPLE FRAME
  tVec_d -- (3, 1) ndarray, the translation vector connecting LAB to DETECTOR
  tVec_s -- (3, 1) ndarray, the translation vector connecting LAB to SAMPLE
  tVec_c -- (3, 1) ndarray, the translation vector connecting SAMPLE to CRYSTAL

  Outputs:
  (m, 2) ndarray containing the intersections of m <= n diffracted beams
  associated with gVecs
*/
XRD_PYTHON_WRAPPER PyObject *
python_gvecToDetectorXY(PyObject * self, PyObject * args)
{
    PyArrayObject *gVec_c,
		*rMat_d, *rMat_s, *rMat_c,
		*tVec_d, *tVec_s, *tVec_c,
		*beamVec;
    PyArrayObject *result;

    int dgc, drd, drs, drc, dtd, dts, dtc, dbv;
    npy_intp npts, dims[2];

    double *gVec_c_Ptr,
        *rMat_d_Ptr, *rMat_s_Ptr, *rMat_c_Ptr,
        *tVec_d_Ptr, *tVec_s_Ptr, *tVec_c_Ptr,
        *beamVec_Ptr;
    double *result_Ptr;

    /* Parse arguments */
    if ( !PyArg_ParseTuple(args,"OOOOOOOO",
                           &gVec_c,
                           &rMat_d, &rMat_s, &rMat_c,
                           &tVec_d, &tVec_s, &tVec_c,
                           &beamVec)) return(NULL);
    if ( gVec_c  == NULL ||
         rMat_d  == NULL || rMat_s == NULL || rMat_c == NULL ||
         tVec_d  == NULL || tVec_s == NULL || tVec_c == NULL ||
         beamVec == NULL ) return(NULL);

    /* Verify shape of input arrays */
    dgc = PyArray_NDIM(gVec_c);
    drd = PyArray_NDIM(rMat_d);
    drs = PyArray_NDIM(rMat_s);
    drc = PyArray_NDIM(rMat_c);
    dtd = PyArray_NDIM(tVec_d);
    dts = PyArray_NDIM(tVec_s);
    dtc = PyArray_NDIM(tVec_c);
    dbv = PyArray_NDIM(beamVec);
    assert( dgc == 2 );
    assert( drd == 2 && drs == 2 && drc == 2 );
    assert( dtd == 1 && dts == 1 && dtc == 1 );
    assert( dbv == 1 );

    /* Verify dimensions of input arrays */
    npts = PyArray_DIMS(gVec_c)[0];

    assert( PyArray_DIMS(gVec_c)[1]  == 3 );
    assert( PyArray_DIMS(rMat_d)[0]  == 3 && PyArray_DIMS(rMat_d)[1] == 3 );
    assert( PyArray_DIMS(rMat_s)[0]  == 3 && PyArray_DIMS(rMat_s)[1] == 3 );
    assert( PyArray_DIMS(rMat_c)[0]  == 3 && PyArray_DIMS(rMat_c)[1] == 3 );
    assert( PyArray_DIMS(tVec_d)[0]  == 3 );
    assert( PyArray_DIMS(tVec_s)[0]  == 3 );
    assert( PyArray_DIMS(tVec_c)[0]  == 3 );
    assert( PyArray_DIMS(beamVec)[0] == 3 );

    /* Allocate C-style array for return data */
    dims[0] = npts; dims[1] = 2;
    result = (PyArrayObject*)PyArray_EMPTY(2,dims,NPY_DOUBLE,0);

    /* Grab data pointers into various arrays */
    gVec_c_Ptr  = (double*)PyArray_DATA(gVec_c);

    rMat_d_Ptr  = (double*)PyArray_DATA(rMat_d);
    rMat_s_Ptr  = (double*)PyArray_DATA(rMat_s);
    rMat_c_Ptr  = (double*)PyArray_DATA(rMat_c);

    tVec_d_Ptr  = (double*)PyArray_DATA(tVec_d);
    tVec_s_Ptr  = (double*)PyArray_DATA(tVec_s);
    tVec_c_Ptr  = (double*)PyArray_DATA(tVec_c);

    beamVec_Ptr = (double*)PyArray_DATA(beamVec);

    result_Ptr     = (double*)PyArray_DATA(result);

    /* Call the computational routine */
    gvec_to_xy(npts, gVec_c_Ptr,
               rMat_d_Ptr, rMat_s_Ptr, rMat_c_Ptr,
               tVec_d_Ptr, tVec_s_Ptr, tVec_c_Ptr,
               beamVec_Ptr,
               result_Ptr);

    /* Build and return the nested data structure */
    return((PyObject*)result);
}

/*
  Takes a list of unit reciprocal lattice vectors in crystal frame to the
  specified detector-relative frame, subject to the conditions:

  1) the reciprocal lattice vector must be able to satisfy a bragg condition
  2) the associated diffracted beam must intersect the detector plane

  Required Arguments:
  gVec_c -- (n, 3) ndarray of n reciprocal lattice vectors in the CRYSTAL FRAME
  rMat_d -- (3, 3) ndarray, the COB taking DETECTOR FRAME components to LAB FRAME
  rMat_s -- (n, 3, 3) ndarray, the COB taking SAMPLE FRAME components to LAB FRAME
  rMat_c -- (3, 3) ndarray, the COB taking CRYSTAL FRAME components to SAMPLE FRAME
  tVec_d -- (3, 1) ndarray, the translation vector connecting LAB to DETECTOR
  tVec_s -- (3, 1) ndarray, the translation vector connecting LAB to SAMPLE
  tVec_c -- (3, 1) ndarray, the translation vector connecting SAMPLE to CRYSTAL

  Outputs:
  (m, 2) ndarray containing the intersections of m <= n diffracted beams
  associated with gVecs
*/
XRD_PYTHON_WRAPPER PyObject *
python_gvecToDetectorXYArray(PyObject * self, PyObject * args)
{
    PyArrayObject *gVec_c,
		*rMat_d, *rMat_s, *rMat_c,
		*tVec_d, *tVec_s, *tVec_c,
		*beamVec;
    PyArrayObject *result;

    int dgc, drd, drs, drc, dtd, dts, dtc, dbv;
    npy_intp npts, dims[2];

    double *gVec_c_Ptr,
        *rMat_d_Ptr, *rMat_s_Ptr, *rMat_c_Ptr,
        *tVec_d_Ptr, *tVec_s_Ptr, *tVec_c_Ptr,
        *beamVec_Ptr;
    double *result_Ptr;

    /* Parse arguments */
    if ( !PyArg_ParseTuple(args,"OOOOOOOO",
                           &gVec_c,
                           &rMat_d, &rMat_s, &rMat_c,
                           &tVec_d, &tVec_s, &tVec_c,
                           &beamVec)) return(NULL);
    if ( gVec_c  == NULL ||
         rMat_d  == NULL || rMat_s == NULL || rMat_c == NULL ||
         tVec_d  == NULL || tVec_s == NULL || tVec_c == NULL ||
         beamVec == NULL ) return(NULL);

    /* Verify shape of input arrays */
    dgc = PyArray_NDIM(gVec_c);
    drd = PyArray_NDIM(rMat_d);
    drs = PyArray_NDIM(rMat_s);
    drc = PyArray_NDIM(rMat_c);
    dtd = PyArray_NDIM(tVec_d);
    dts = PyArray_NDIM(tVec_s);
    dtc = PyArray_NDIM(tVec_c);
    dbv = PyArray_NDIM(beamVec);
    assert( dgc == 2 );
    assert( drd == 2 && drs == 3 && drc == 2 );
    assert( dtd == 1 && dts == 1 && dtc == 1 );
    assert( dbv == 1 );

    /* Verify dimensions of input arrays */
    npts = PyArray_DIMS(gVec_c)[0];

    if (npts != PyArray_DIM(rMat_s, 0)) {
        PyErr_Format(PyExc_ValueError, "gVec_c and rMat_s length mismatch %d vs %d",
                     (int)PyArray_DIM(gVec_c, 0), (int)PyArray_DIM(rMat_s, 0));
        return NULL;
    }
    assert( PyArray_DIMS(gVec_c)[1]  == 3 );
    assert( PyArray_DIMS(rMat_d)[0]  == 3 && PyArray_DIMS(rMat_d)[1] == 3 );
    assert( PyArray_DIMS(rMat_s)[1]  == 3 && PyArray_DIMS(rMat_s)[2] == 3 );
    assert( PyArray_DIMS(rMat_c)[0]  == 3 && PyArray_DIMS(rMat_c)[1] == 3 );
    assert( PyArray_DIMS(tVec_d)[0]  == 3 );
    assert( PyArray_DIMS(tVec_s)[0]  == 3 );
    assert( PyArray_DIMS(tVec_c)[0]  == 3 );
    assert( PyArray_DIMS(beamVec)[0] == 3 );

    /* Allocate C-style array for return data */
    dims[0] = npts; dims[1] = 2;
    result = (PyArrayObject*)PyArray_EMPTY(2,dims,NPY_DOUBLE,0);

    /* Grab data pointers into various arrays */
    gVec_c_Ptr  = (double*)PyArray_DATA(gVec_c);

    rMat_d_Ptr  = (double*)PyArray_DATA(rMat_d);
    rMat_s_Ptr  = (double*)PyArray_DATA(rMat_s);
    rMat_c_Ptr  = (double*)PyArray_DATA(rMat_c);

    tVec_d_Ptr  = (double*)PyArray_DATA(tVec_d);
    tVec_s_Ptr  = (double*)PyArray_DATA(tVec_s);
    tVec_c_Ptr  = (double*)PyArray_DATA(tVec_c);

    beamVec_Ptr = (double*)PyArray_DATA(beamVec);

    result_Ptr     = (double*)PyArray_DATA(result);

    /* Call the computational routine */
    gvec_to_xy_array(npts, gVec_c_Ptr,
                     rMat_d_Ptr, rMat_s_Ptr, rMat_c_Ptr,
                     tVec_d_Ptr, tVec_s_Ptr, tVec_c_Ptr,
                     beamVec_Ptr,
                     result_Ptr);

    /* Build and return the nested data structure */
    return((PyObject*)result);
}


#endif /* XRD_INCLUDE_PYTHON_WRAPPERS */
