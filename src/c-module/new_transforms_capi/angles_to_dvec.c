
#if !defined(XRD_SINGLE_COMPILE_UNIT) || !XRD_SINGLE_COMPILE_UNIT
#  include "transforms_utils.h"
#  include "transforms_prototypes.h"
#  if defined(XRD_TASK_SYSTEM) && (XRD_TASK_SYSTEM)
#     include "transforms_task_system.h"
#  endif # XRD_TASK_SYSTEM
#endif


#define DEFAULT_CHUNK_SIZE ((size_t)128)

typedef struct {
    /* out */
    double* dVec_c;

    /* in */
    double* angs;
    double* rMat_c;
    double  rMat_e[9];
    double chi;
    size_t chunk_size;
    size_t total_count;
} angles_to_dvec_params;


static inline void
angles_to_dvec_chunked(angles_to_dvec_params* params, size_t chunk)
{
    size_t loop_start, loop_end;
    size_t i, j, k, l;
    double rMat_s[9], rMat_ctst[9];
    double gVec_e[3], gVec_l[3], gVec_c_tmp[3];
    const double *angs = params->angs;
    const double *rMat_c = params->rMat_c;
    const double *rMat_e = &params->rMat_e[0];
    double *dVec_c = params->dVec_c;
    double chi = params->chi;
    size_t chunk_size = params->chunk_size;
    size_t total_count = params->total_count;
    
    loop_start = chunk * chunk_size;
    loop_end = loop_start + chunk_size;
    loop_end = (loop_end <= total_count)? loop_end : total_count;

    /* make vector array */
    for (i=loop_start; i<loop_end; i++) {
        /* components in BEAM frame */
        double theta = angs[3*i];
        double eta = angs[3*i+1];
        double omega = angs[3*i+2];

        double sin_theta = sin(theta);
        double cos_theta = cos(theta);
        double sin_eta = sin(eta);
        double cos_eta = cos(eta);
        
        gVec_e[0] = sin_theta * cos_eta;
        gVec_e[1] = sin_theta * sin_eta;
        gVec_e[2] = -cos_theta;

        /* take from BEAM frame to LAB frame */
        for (j=0; j<3; j++) {
            gVec_l[j] = 0.0;
            for (k=0; k<3; k++) {
                gVec_l[j] += rMat_e[3*j+k]*gVec_e[k];
            }
        }

        /* need pointwise rMat_s according to omega */
        make_sample_rmat(chi, omega, rMat_s);

        /* compute dot(rMat_c.T, rMat_s.T) and hit against gVec_l */
        for (j=0; j<3; j++) {
            for (k=0; k<3; k++) {
                rMat_ctst[3*j+k] = 0.0;
                for (l=0; l<3; l++) {
                    rMat_ctst[3*j+k] += rMat_c[3*l+j]*rMat_s[3*k+l];
                }
            }
            gVec_c_tmp[j] = 0.0;
            for (k=0; k<3; k++) {
                gVec_c_tmp[j] += rMat_ctst[3*j+k]*gVec_l[k];
            }
            dVec_c[3*i+j] = gVec_c_tmp[j];
        }
    }

}

XRD_CFUNCTION void
angles_to_dvec(size_t nvecs,
               double * angs,
               double * bHat_l, double * eHat_l,
               double chi, double * rMat_c,
               double * dVec_c)
{
    angles_to_dvec_params params;
    
    params.dVec_c = dVec_c;
    params.angs = angs;
    params.rMat_c = rMat_c;
    /* Need eta frame cob matrix (could omit for standard setting) */
    make_beam_rmat(bHat_l, eHat_l, &params.rMat_e[0]);
    params.chi = chi;
    params.chunk_size = nvecs;
    params.total_count = nvecs;

    /* for the moment being, leave this wrapper single chunked and serial */
    angles_to_dvec_chunked(&params, 0);
}


#if defined(XRD_INCLUDE_PYTHON_WRAPPERS) && XRD_INCLUDE_PYTHON_WRAPPERS

#  if !defined(XRD_SINGLE_COMPILE_UNIT) || !XRD_SINGLE_COMPILE_UNIT
#    define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#    include <Python.h>
#    include <numpy/arrayobject.h>
#  endif /* XRD_SINGLE_COMPILE_UNIT */

XRD_PYTHON_WRAPPER const char *docstring_anglesToDVec =
    "c module implementation of angles_to_dvec.\n"
    "Please use the Python wrapper.\n";

XRD_PYTHON_WRAPPER PyObject *
python_anglesToDVec(PyObject * self, PyObject * args)
{
    PyArrayObject *angs, *bHat_l, *eHat_l, *rMat_c;
    PyArrayObject *dVec_c;
    double chi;
    npy_intp nvecs, rdims[2];

    int nangs, nbhat, nehat, nrmat;
    int da1, db1, de1, dr1, dr2;
    
    /* Parse arguments */
    if (!PyArg_ParseTuple(args,"OOOdO",
                          &angs,
                          &bHat_l, &eHat_l,
                          &chi, &rMat_c)) return(NULL);
    if ( angs == NULL ) return(NULL);

    /* Verify shape of input arrays */
    nangs = PyArray_NDIM(angs);
    nbhat = PyArray_NDIM(bHat_l);
    nehat = PyArray_NDIM(eHat_l);
    nrmat = PyArray_NDIM(rMat_c);

    assert( nangs==2 && nbhat==1 && nehat==1 && nrmat==2 );

    /* Verify dimensions of input arrays */
    nvecs = PyArray_DIMS(angs)[0]; /* rows */
    da1   = PyArray_DIMS(angs)[1]; /* cols */

    db1   = PyArray_DIMS(bHat_l)[0];
    de1   = PyArray_DIMS(eHat_l)[0];
    dr1   = PyArray_DIMS(rMat_c)[0];
    dr2   = PyArray_DIMS(rMat_c)[1];

    assert( da1 == 3 );
    assert( db1 == 3 && de1 == 3);
    assert( dr1 == 3 && dr2 == 3);

    
    /* Allocate C-style array for return data */
    rdims[0] = nvecs; rdims[1] = 3;
    dVec_c = (PyArrayObject*)PyArray_EMPTY(2,rdims,NPY_DOUBLE,0);

    if (nvecs > 0)
    {
        angles_to_dvec_params fn_params;
        double *bHat_l_ptr, *eHat_l_ptr;
        size_t chunk_count;
        fn_params.dVec_c = (double*)PyArray_DATA(dVec_c);
        fn_params.angs = (double*)PyArray_DATA(angs);
        fn_params.rMat_c = (double*)PyArray_DATA(rMat_c);
        /* Need eta frame cob matrix (could omit for standard setting) */
        bHat_l_ptr = (double*)PyArray_DATA(bHat_l);
        eHat_l_ptr = (double*)PyArray_DATA(eHat_l);
        make_beam_rmat(bHat_l_ptr, eHat_l_ptr, &fn_params.rMat_e[0]);
        fn_params.chi = chi;
        fn_params.chunk_size = DEFAULT_CHUNK_SIZE;
        fn_params.total_count = nvecs;

        chunk_count = 1 + ((nvecs - 1) / DEFAULT_CHUNK_SIZE);

        if (chunk_count > 1) {
            xrd_transforms_task_apply(chunk_count,
                                      &fn_params, angles_to_dvec_chunked);
        } else {
            angles_to_dvec_chunked(&fn_params, 0);
        }
    }

    return ((PyObject*)dVec_c);
}

#endif /* XRD_INCLUDE_PYTHON_WRAPPERS */
