
#if !defined(XRD_SINGLE_COMPILE_UNIT) || !XRD_SINGLE_COMPILE_UNIT
#  include "transforms_utils.h"
#  include "transforms_prototypes.h"
#endif


/*
  returns N, where N solves N*vect+origin laying in the plane defined by plane.

  origin -> point where the ray starts.
  vect -> direction vector for the ray.
  plane -> vector[4] for A B C D in the plane formula Ax + By + Cz + D = 0. Note
           this means that (A, B, C) is the plane normal.

  Note that an N value will always be returned. In case of the line being
  parallel to the plane, an infinity will be generated. In case of the plane
  being "behind" the plane, a negative value will be generated.
 */
static inline
double
RayPlaneIntersect(const double *origin, const double *vect,
                  const double *plane)
{
    return (plane[3] - v3_v3s_dot(plane, origin, 1)) / v3_v3s_dot(plane, vect, 1);
}

static void
gvec_to_xy_single(const double *gVec_c, const double *rMat_d, const double *rMat_sc,
                  const double *tVec_d, const double *nVec_l,
                  const double num, const double *tVec_dc,
                  double * restrict result)
{
    double bDot, ztol, denom, u;
    double gVec_l[3], dVec_l[3];
    double P2_d[3];

    ztol = epsf;

    /* Compute unit reciprocal lattice vector in crystal frame w/o translation */

    /*
	 * Compute unit reciprocal lattice vector in lab frame
	 * and dot with beam vector
	 */
    m33_v3s_multiply(rMat_sc, gVec_c, 1, gVec_l);
    bDot = gVec_l[2];

    if ( bDot < ztol && bDot > 1.0-ztol )
        goto no_diffraction;

    /* diffraction directly on a vector. Assumes beam vector is { 0, 0, 1},
       so that only the last column of the binary_rmat is needed
    */
    dVec_l[0] = 2*gVec_l[0]*gVec_l[2];
    dVec_l[1] = 2*gVec_l[1]*gVec_l[2];
    dVec_l[2] = 2*gVec_l[2]*gVec_l[2] - 1.0;
    denom = v3_v3s_dot(nVec_l, dVec_l, 1);

    if ( denom > -ztol )
        goto no_diffraction;

    u = num/denom;

    v3s_s_v3_muladd(dVec_l, 1, -u, tVec_dc,  P2_d);
    /* P2_d is a point in the detector plane. As we are changing to th
       detector frame, the z coordinate will always be 0, so avoid computing
       it. Note that the result is an xy point (that is, 2d), so using a
       m33_v3_multiply would result in a potential rogue memory write */
    result[0] = v3_v3s_dot(rMat_d, P2_d, 1);
    result[1] = v3_v3s_dot(rMat_d + 3, P2_d, 1);

    return;

 no_diffraction:
    result[0] = NAN;
    result[1] = NAN;
    return;
}


XRD_CFUNCTION void
gvec_to_xy(size_t npts, const double *gVec_c,
           const double *rMat_d, const double *rMat_s, const double *rMat_c,
           const double *tVec_d, const double *tVec_s, const double *tVec_c,
           double * restrict result)
{
    size_t i;
    double num;
    double nVec_l[3], tVec_sc[3], tVec_ds[3], tVec_dc[3];
    double rMat_sc[9];

    /*
       compute detector normal in LAB (nVec_l)
       the normal will just be the Z column vector of rMat_d
    */
    v3s_copy(rMat_d + 2, 3, nVec_l);

    /* tVec_ds is the translation from DETECTOR to SAMPLE */
    v3_v3s_sub(tVec_s, tVec_d, 1, tVec_ds);

    /*
       tVec_dc <= tVec_s + rMat_s x tVec_c
       tVec_sc is transform from SAMPLE to CRYSTAL
       tVec_dc is transform from DETECTOR to CRYSTAL
     */
    m33_v3s_multiply(rMat_s, tVec_c, 1, tVec_sc);
    v3_v3s_add(tVec_ds, tVec_sc, 1, tVec_dc);

    num = v3_v3s_dot(nVec_l, tVec_dc, 1);

    /* accumulate rMat_s and rMat_c. rMat_sc is a COB Matrix from LAB to
       CRYSTAL */
    m33_m33_multiply(rMat_s, rMat_c, rMat_sc);

    for (i=0L; i<npts; i++) {
        gvec_to_xy_single(gVec_c + 3*i, rMat_d, rMat_sc, tVec_d,
                          nVec_l, num, tVec_dc, result + 2*i);
    }
}

/*
 * The only difference between this and the non-Array version
 * is that rMat_s is an array of matrices of length npts instead
 * of a single matrix.
 */
XRD_CFUNCTION void
gvec_to_xy_array(size_t npts, const double *gVec_c,
                 const double *rMat_d, const double *rMat_ss, const double *rMat_c,
                 const double *tVec_d, const double *tVec_s, const double *tVec_c,
                 double * restrict result)
{
    size_t i;
    double num;
    double nVec_l[3], tVec_sc[3], tVec_ds[3], tVec_dc[3];
    double rMat_sc[9];
    /*
      compute detector normal in LAB (nVec_l)
       The normal will just be the Z column vector of rMat_d
     */
    v3s_copy(rMat_d + 2, 3, nVec_l);

    /* tVec_fd is the translation from DETECTOR to SAMPLE */
    v3_v3s_sub(tVec_s, tVec_d, 1, tVec_ds);

    for (i=0L; i<npts; i++) {
        const double *rMat_s = rMat_ss + 9*i;
        /*
           tVec_dc <= tVec_s + rMat_s x tVec_c
           tVec_sc is transform from SAMPLE to CRYSTAL
           tVec_dc is transform from DETECTOR to CRYSTAL
        */
        m33_v3s_multiply(rMat_s, tVec_c, 1, tVec_sc);
        v3_v3s_add(tVec_ds, tVec_sc, 1, tVec_dc);

        num = v3_v3s_dot(nVec_l, tVec_dc, 1);

        /* Compute the matrix product of rMat_s and rMat_c */
        m33_m33_multiply(rMat_s, rMat_c, rMat_sc);

        gvec_to_xy_single(gVec_c + 3*i, rMat_d, rMat_sc, tVec_d,
                          nVec_l, num, tVec_dc, result + 2*i);
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

    result_Ptr  = (double*)PyArray_DATA(result);

    /* Call the computational routine */
    gvec_to_xy(npts, gVec_c_Ptr,
               rMat_d_Ptr, rMat_s_Ptr, rMat_c_Ptr,
               tVec_d_Ptr, tVec_s_Ptr, tVec_c_Ptr,
               /*               beamVec_Ptr,*/
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

    result_Ptr  = (double*)PyArray_DATA(result);

    /* Call the computational routine */
    gvec_to_xy_array(npts, gVec_c_Ptr,
                     rMat_d_Ptr, rMat_s_Ptr, rMat_c_Ptr,
                     tVec_d_Ptr, tVec_s_Ptr, tVec_c_Ptr,
                     /*                     beamVec_Ptr,*/
                     result_Ptr);

    /* Build and return the nested data structure */
    return((PyObject*)result);
}


#endif /* XRD_INCLUDE_PYTHON_WRAPPERS */
