
#if !defined(XRD_SINGLE_COMPILE_UNIT) || !XRD_SINGLE_COMPILE_UNIT
#  include "transforms_utils.h"
#  include "transforms_prototypes.h"
#endif

/*
  computes the diffraction vector for the G vector ```gvec```
  Assumes the inciding beam vector is the Z unit vector {0,0,1}
*/
static inline
void
diffract_z(const double *gvec, double * restrict diffracted)
{
    diffracted[0] = 2*gvec[0]*gvec[2];
    diffracted[1] = 2*gvec[1]*gvec[2];
    diffracted[2] = 2*gvec[2]*gvec[2] - 1.0;
}

static inline
void
diffract(const double *beam, double *vec, double * restrict diffracted)
{
    /* take advantage of the simmetry so no need to build the full matrix */
    double bm00 = beam[0]*beam[0]*2.0 - 1.0;
    double bm11 = beam[1]*beam[1]*2.0 - 1.0;
    double bm22 = beam[2]*beam[2]*2.0 - 1.0;
    double bm01 = beam[0]*beam[1]*2.0;
    double bm02 = beam[0]*beam[2]*2.0;
    double bm12 = beam[1]*beam[2]*2.0;
    double v0 = vec[0];
    double v1 = vec[1];
    double v2 = vec[2];

    diffracted[0] = bm00*v0 + bm01*v1 + bm02*v2;
    diffracted[1] = bm01*v0 + bm11*v1 + bm12*v2;
    diffracted[2] = bm02*v0 + bm12*v1 + bm22*v2;
}


/*
  returns N, where N solves N*vect+origin laying in the plane defined by plane.

  origin -> point where the ray starts.
  vect -> direction vector for the ray.
  plane -> vector[4] for A B C D in the plane formula Ax + By + Cz + D = 0. Note
           this means that (A, B, C) is the plane normal.

  returns 1 on intersection, 0 if there is no intersection. In case of
  intersection, the intersecting point will be filled in
  collision_point. Otherwise collision_point will be left untouched.

  Note this code does not handle division by 0. on IEEE-754 arithmetic t may
  become an infinity in that case. The result would be infinity in the resulting
  collision point. As the result is going to be clipped at some point, this
  should not be a problem.
 */
static inline
int
ray_plane_intersect(const double *origin, const double *vect,
                    const double *plane, double * restrict collision_point)
{
    double t;
    t = (plane[3] - v3_v3s_dot(plane, origin, 1)) / v3_v3s_dot(plane, vect, 1);
    if (t < 0.0)
        return 0;
    v3s_s_v3_muladd(vect, 1, t, origin, collision_point);
    return 1;
}


XRD_CFUNCTION void
gvec_to_xy(size_t npts, const double *gVec_cs,
           const double *rMat_d, const double *rMat_ss, const double *rMat_c,
           const double *tVec_d, const double *tVec_s, const double *tVec_c,
           const double *beamVec_Ptr,
           double * restrict result, unsigned int flags)
{
    size_t i;
    double plane[4], tVec_sc[3], tVec_ds[3];
    double ray_origin[3];
    double rMat_sc[9];
    int use_single_rMat_s = flags & GV2XY_SINGLE_RMAT_S;

    /*
      compute detector normal in LAB (nVec_l)
       The normal will just be the Z column vector of rMat_d
     */
    v3s_copy(rMat_d + 2, 3, plane);
    plane[3] = 0.0;

    /* tVec_fd is the translation from DETECTOR to SAMPLE */
    v3_v3s_sub(tVec_s, tVec_d, 1, tVec_ds);

    if (use_single_rMat_s)
    {
        const double *rMat_s = rMat_ss; /* only one */
        /* All gvecs use the same rMat_s */
        /* ray origin can be already computed */
        m33_v3s_multiply(rMat_s, tVec_c, 1, tVec_sc);
        v3_v3s_add(tVec_ds, tVec_sc, 1, ray_origin);

        /* rMat_sc can be precomputed, so transforming the transformer gvec in
           the loop is faster.
        */
        m33_m33_multiply(rMat_s, rMat_c, rMat_sc);
    }

    for (i=0L; i<npts; i++) {
        /* This loop generates the ray, checks collision with detector plane,
           obtains the collision point and projects it into DETECTOR frame.
           Collision detection is made in LAB coordinates, but centered around
           the detector position.

           Ray ORIGIN will be the input "tVec_c" in LAB frame centered on the
           detector.

           Ray VECTOR will be the diffracted direction based on the associated
           gVec_c and the beam vector (hardcoded to z unit vector {0,0,1}) for
           speed.
        */
        double gVec_l[3], ray_vector[3], point[3];

        const double *gVec_c = gVec_cs + 3*i;

        if (use_single_rMat_s)
        {
            m33_v3s_multiply(rMat_sc, gVec_c, 1, gVec_l);
        }
        else
        {
            double gVec_s[3];
            const double *rMat_s = rMat_ss + 9*i;
            /*
              tVec_dc <= tVec_s + rMat_s x tVec_c
              tVec_sc is transform from SAMPLE to CRYSTAL
              tVec_dc is transform from DETECTOR to CRYSTAL
            */

            m33_v3s_multiply(rMat_s, tVec_c, 1, tVec_sc);
            v3_v3s_add(tVec_ds, tVec_sc, 1, ray_origin);

            /* Compute gVec in lab frame */
            m33_v3s_multiply(rMat_c, gVec_c, 1, gVec_s); /* gVec in SAMPLE frame */
            m33_v3s_multiply(rMat_s, gVec_s, 1, gVec_l); /* gVec in LAB frame */
        }

        if (!beamVec_Ptr) /* this will be properly predicted in most cases */
        {
            /* if no beamVec is provided, assume the {0,0,1} and use a fast-path*/
            diffract_z(gVec_l, ray_vector);
        }
        else
        {
            diffract(beamVec_Ptr, gVec_l, ray_vector);
        }

        /* Compute collision point of ray-plane */
        if (0 == ray_plane_intersect(ray_origin, ray_vector, plane, point))
        {
            /* no intersection */
            result[2*i] = NAN;
            result[2*i + 1] = NAN;
            continue;
        }

        /*
           project into DETECTOR coordinates. Only ```x``` and ```y``` as
           ```z``` will always be 0.0
        */
        result[2*i] = v3_v3s_dot(rMat_d, point, 1);
        result[2*i+1] = v3_v3s_dot(rMat_d + 3, point, 1);
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
    double beam[3], beam_sqrnorm;
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

    /* normalize beam vector and detect if it is z */
    beam_sqrnorm = v3_v3s_dot(beamVec_Ptr, beamVec_Ptr, 1);
    if (beam_sqrnorm > (epsf*epsf))
    {
        double beam_z;
        double beam_rnorm = -1.0/sqrt(beam_sqrnorm);
        beam_z = beamVec_Ptr[2] * beam_rnorm;
        if (fabs(beam_z - 1.0) < epsf)
        {
            /* consider that beam is {0, 0, 1}, activate fast path by using a
               NULL beam */
            beamVec_Ptr = NULL;
        }
        else
        {
            /* normalize the beam vector in full... */
            beam[0] = beamVec_Ptr[0] * beam_rnorm;
            beam[1] = beamVec_Ptr[1] * beam_rnorm;
            beam[2] = beam_z;
            beamVec_Ptr = beam;
        }
    }

    /* Call the computational routine */
    gvec_to_xy(npts, gVec_c_Ptr,
               rMat_d_Ptr, rMat_s_Ptr, rMat_c_Ptr,
               tVec_d_Ptr, tVec_s_Ptr, tVec_c_Ptr,
               beamVec_Ptr,
               result_Ptr,
               GV2XY_SINGLE_RMAT_S);

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
    double beam[3], beam_sqrnorm;
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

    /* normalize beam vector and detect if it is z */
    beam_sqrnorm = v3_v3s_dot(beamVec_Ptr, beamVec_Ptr, 1);
    if (beam_sqrnorm > (epsf*epsf))
    {
        double beam_z;
        double beam_rnorm = -1.0/sqrt(beam_sqrnorm);
        beam_z = beamVec_Ptr[2] * beam_rnorm;
        if (fabs(beam_z - 1.0) < epsf)
        {
            /* consider that beam is {0, 0, 1}, activate fast path by using a
               NULL beam */
            beamVec_Ptr = NULL;
        }
        else
        {
            /* normalize the beam vector in full... */
            beam[0] = beamVec_Ptr[0] * beam_rnorm;
            beam[1] = beamVec_Ptr[1] * beam_rnorm;
            beam[2] = beam_z;
            beamVec_Ptr = beam;
        }
    }

    /* Call the computational routine */
    gvec_to_xy(npts, gVec_c_Ptr,
               rMat_d_Ptr, rMat_s_Ptr, rMat_c_Ptr,
               tVec_d_Ptr, tVec_s_Ptr, tVec_c_Ptr,
               beamVec_Ptr,
               result_Ptr,
               0);

    /* Build and return the nested data structure */
    return((PyObject*)result);
}


#endif /* XRD_INCLUDE_PYTHON_WRAPPERS */
