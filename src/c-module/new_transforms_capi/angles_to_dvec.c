#define USE_TRANSFORMS_UTILS 1
#define DEFAULT_CHUNK_SIZE ((size_t)500)
#define USE_VECTOR_SINCOS 0
#define USE_FLOAT_KERNEL 0
#define USE_MEMORY_ONLY_KERNEL 0
#define USE_METAL 1
#if !defined(XRD_SINGLE_COMPILE_UNIT) || !XRD_SINGLE_COMPILE_UNIT
#  include "transforms_utils.h"
#  include "transforms_prototypes.h"
#  if defined(XRD_TASK_SYSTEM) && (XRD_TASK_SYSTEM)
#     include "transforms_task_system.h"
#  endif # XRD_TASK_SYSTEM
#endif

#if USE_VECTOR_SINCOS
#  include <Accelerate/Accelerate.h>
#endif

typedef struct angles_to_dvec_params_tag angles_to_dvec_params;

#if defined(USE_METAL) && USE_METAL
int
metal_support_init(void);

void
metal_support_release(void);

int
metal_support_angles_to_dvec(const angles_to_dvec_params *params);

#endif /* USE_METAL */

struct angles_to_dvec_params_tag {
    /* invariant */
    double rMat_c[9];
    double rMat_e[9];
    double chi;
    size_t chunk_size;
    size_t total_count;

    /* in stream */
    const double *angs;

    /* out stream*/
    double *dVec_c;
};

#if defined(USE_MEMORY_ONLY_KERNEL) && USE_MEMORY_ONLY_KERNEL
static inline void
angles_to_dvec_memonly_chunked(angles_to_dvec_params *params, size_t chunk)
{
    size_t loop_start, loop_end, i;
    const size_t chunk_size = params->chunk_size;
    const size_t total_count = params->total_count;
    const uint64_t *angs = params->angs;
    double * restrict dVec_c = params->dVec_c;

    loop_start = chunk * chunk_size;
    loop_end = loop_start + chunk_size;
    loop_end = (loop_end <= total_count)? loop_end : total_count;

    /* a hard limit on speed... memcpy speed */
    /* for (i = loop_start; i < loop_end; i++) { */
    /*     dVec_c[i*3+0] = angs[i*3+0]; */
    /*     dVec_c[i*3+1] = angs[i*3+1]; */
    /*     dVec_c[i*3+2] = angs[i*3+2];  */
    /* } */
    memcpy(dVec_c + 3*loop_start, angs + 3*loop_start, 3*(loop_end - loop_start)*sizeof(double));
}
#endif /* USE_MEMORY_ONLY_KERNEL */

static size_t transpose_table[] = { 0, 3, 6, 1, 4, 7, 2, 5, 8 };

static inline void
angles_to_dvec_sp_chunked(angles_to_dvec_params *params, size_t chunk)
{
    size_t loop_start, loop_end;
    size_t i;
    float rMat_s[9];
    float gVec_e[3], gVec_l[3];
    float vec3_tmp[3], vec3_result[3];
    const size_t chunk_size = params->chunk_size;
    const size_t total_count = params->total_count;
    const double *angs = params->angs;
    float rMat_c[9];
    float rMat_e[9];
    double * restrict dVec_c = params->dVec_c;
    float chi = params->chi;
    float sin_chi = sinf(chi);
    float cos_chi = cosf(chi);
#if defined(USE_VECTOR_SINCOS) &&  USE_VECTOR_SINCOS
    float sin_teo[3*chunk_size];
    float cos_teo[3*chunk_size];
#endif

    for (i=0; i<9; i++) {
        rMat_e[i] = (float)params->rMat_e[transpose_table[i]];
    }
    for (i=0; i<9; i++) {
        rMat_c[i] = (float)params->rMat_c[transpose_table[i]];
    }


    loop_start = chunk * chunk_size;
    loop_end = loop_start + chunk_size;
    loop_end = (loop_end <= total_count)? loop_end : total_count;

#if defined(USE_VECTOR_SINCOS) &&  USE_VECTOR_SINCOS
    {
        float angles[3*chunk_size];
        int this_chunk_size = loop_end - loop_start;
        int angs_size = 3*this_chunk_size;

        vDSP_vdpsp(angs, 1, angles, 1, angs_size);
        vvsincosf(sin_teo, cos_teo, angles, &angs_size);
    }
#endif

    
    for (i=loop_start; i<loop_end; i++) {
        /* components in BEAM frame */
#if defined(USE_VECTOR_SINCOS) &&  USE_VECTOR_SINCOS
        size_t teo_idx = 3*(i-loop_start);
        float sin_theta = sin_teo[teo_idx];
        float cos_theta = cos_teo[teo_idx];
        float sin_eta   = sin_teo[teo_idx+1];
        float cos_eta   = cos_teo[teo_idx+1];
        float sin_omega = sin_teo[teo_idx+2];
        float cos_omega = cos_teo[teo_idx+2];
#else
        float theta = angs[3*i];
        float eta = angs[3*i+1];
        float omega = angs[3*i+2];
        float sin_theta = sinf(theta);
        float cos_theta = cosf(theta);
        float sin_eta = sinf(eta);
        float cos_eta = cosf(eta);
        float sin_omega = sinf(omega);
        float cos_omega = cosf(omega);
#endif

        gVec_e[0] = sin_theta * cos_eta;
        gVec_e[1] = sin_theta * sin_eta;
        gVec_e[2] = -cos_theta;

        /* need pointwise rMat_s according to omega */
        make_sample_rmat_t_polar_f(sin_chi, cos_chi, sin_omega, cos_omega, rMat_s);

        /* take from BEAM frame to LAB frame */
        vec3_mat33t_product_f(gVec_l, gVec_e, rMat_e);
        vec3_mat33t_product_f(vec3_tmp, gVec_l, rMat_s);
        vec3_mat33t_product_f(vec3_result, vec3_tmp, rMat_c);
        dVec_c[3*i+0] = vec3_result[0];
        dVec_c[3*i+1] = vec3_result[1];
        dVec_c[3*i+2] = vec3_result[2];
    }
}

static inline void
angles_to_dvec_chunked(angles_to_dvec_params* params, size_t chunk)
{
    size_t loop_start, loop_end;
    size_t i;
    double rMat_s[9];
    double gVec_e[3], gVec_l[3];
#if defined(USE_TRANSFORMS_UTILS) && USE_TRANSFORMS_UTILS
    double vec3_tmp[3];
#else
    size_t j, k, l;
    double rMat_ctst[9], gVec_c_tmp[3];
#endif
    const size_t chunk_size = params->chunk_size;
    const size_t total_count = params->total_count;
    const double *angs = params->angs;
    const double *rMat_c = params->rMat_c;
    const double *rMat_e = &params->rMat_e[0];
    double * restrict dVec_c = params->dVec_c;
    double chi = params->chi;
    double sin_chi = sin(chi);
    double cos_chi = cos(chi);
#if defined(USE_VECTOR_SINCOS) &&  USE_VECTOR_SINCOS
    double sin_teo[3*chunk_size];
    double cos_teo[3*chunk_size];
#endif

    loop_start = chunk * chunk_size;
    loop_end = loop_start + chunk_size;
    loop_end = (loop_end <= total_count)? loop_end : total_count;

#if defined(USE_VECTOR_SINCOS) &&  USE_VECTOR_SINCOS
    {
        int this_chunk_size = loop_end - loop_start;
        int angs_size = 3*this_chunk_size;
        vvsincos(sin_teo, cos_teo, angs+3*loop_start, &angs_size);
    }
#endif

    for (i=loop_start; i<loop_end; i++) {
        /* components in BEAM frame */

#if defined(USE_VECTOR_SINCOS) &&  USE_VECTOR_SINCOS
        size_t teo_idx = 3*(i-loop_start);
        double sin_theta = sin_teo[teo_idx];
        double cos_theta = cos_teo[teo_idx];
        double sin_eta   = sin_teo[teo_idx+1];
        double cos_eta   = cos_teo[teo_idx+1];
        double sin_omega = sin_teo[teo_idx+2];
        double cos_omega = cos_teo[teo_idx+2];
#else
        double theta = angs[3*i];
        double eta = angs[3*i+1];
        double omega = angs[3*i+2];
        double sin_theta = sin(theta);
        double cos_theta = cos(theta);
        double sin_eta = sin(eta);
        double cos_eta = cos(eta);
        double sin_omega = sin(omega);
        double cos_omega = cos(omega);
#endif

        gVec_e[0] = sin_theta * cos_eta;
        gVec_e[1] = sin_theta * sin_eta;
        gVec_e[2] = -cos_theta;

        /* need pointwise rMat_s according to omega */
        make_sample_rmat_polar(sin_chi, cos_chi, sin_omega, cos_omega, rMat_s);

#if defined(USE_TRANSFORMS_UTILS) && USE_TRANSFORMS_UTILS
        /* take from BEAM frame to LAB frame */
        vec3_mat33_product(gVec_l, gVec_e, rMat_e);
        vec3_mat33_product(vec3_tmp, gVec_l, rMat_s);
        vec3_mat33_product(dVec_c+3*i, vec3_tmp, rMat_c);
#else
        for (j=0; j<3; j++) {
            gVec_l[j] = 0.0;
            for (k=0; k<3; k++) {
                gVec_l[j] += rMat_e[3*j+k]*gVec_e[k];
            }
        }
        /* take from LAB to CRYSTAL */
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
#endif
    }
}

XRD_CFUNCTION void
angles_to_dvec(size_t nvecs,
               double * angs,
               double * bHat_l, double * eHat_l,
               double chi, double * rMat_c,
               double * dVec_c)
{
    size_t i;
    angles_to_dvec_params params;


    for (i=0; i<9;i++)
        params.rMat_c[i] = rMat_c[i];
    /* Need eta frame cob matrix (could omit for standard setting) */
    make_beam_rmat(bHat_l, eHat_l, &params.rMat_e[0]);
    params.chi = chi;
    params.chunk_size = nvecs;
    params.total_count = nvecs;
    params.angs = angs;
    params.dVec_c = dVec_c;

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

static int
extract_vec3_d_from_pyarray(double * restrict dst, PyArrayObject* arr)
{
    /* checks that arr is a 3 vector and copies its contents to dst */
    
    int ndims = PyArray_NDIM(arr);
    int type_number = PyArray_TYPE(arr);
    npy_intp *shape = PyArray_SHAPE(arr);
    npy_intp *strides = PyArray_STRIDES(arr);
    char *data = PyArray_BYTES(arr);
    
    /* 3x3 array */
    if ((ndims != 1) || (shape[0] != 3))
        goto error;

    /* note: all this kind of assumes everything is in NBO */
    /* support either float or double arrays */
    if (NPY_DOUBLE == type_number)
    {
        int i;

        for (i=0; i<3; i++)
        {
            dst[i] = *(double*)(data + strides[0]*i);
        }
    }
    else if (NPY_FLOAT == type_number)
    {
        int i;

        for (i=0; i<3; i++)
        {
            dst[i] = (double)*(float*)(data + strides[0]*i);
        }
    }
    else
        goto error;

    return 0;

 error:
    return -1;
}

static int
extract_mat33_d_from_pyarray(double * restrict dst, PyArrayObject* arr)
{
    /* checks that the argument arr is a 3x3 array and copies its contents to
       dst as column-contiguous (FORTRAN order/transposed C order)
    */
    int ndims = PyArray_NDIM(arr);
    int type_number = PyArray_TYPE(arr);
    npy_intp *shape = PyArray_SHAPE(arr);
    npy_intp *strides = PyArray_STRIDES(arr);
    char *data = PyArray_BYTES(arr);
    
    /* 3x3 array */
    if ((ndims != 2) || (shape[0] != 3) || (shape[1]!=3))
        goto error;

    /* note: all this kind of assumes everything is in NBO */
    /* support either float or double arrays */
    if (NPY_DOUBLE == type_number)
    {
        int i, j;

        for (j=0; j<3; j++)
            for (i=0; i<3; i++)
            {
                dst[j*3+i] = *(double*)(data + strides[1]*i + strides[0]*j);
            }
    }
    else if (NPY_FLOAT == type_number)
    {
        int i, j;

        for (j=0; j<3; j++)
            for (i=0; i<3; i++)
            {
                dst[j*3+i] = (double)*(float*)(data + strides[1]*i + strides[0]*j);
            }
    }
    else
        goto error;

    return 0;

 error:
    return -1;
}


XRD_PYTHON_WRAPPER PyObject *
python_anglesToDVec(PyObject * self, PyObject * args)
{
    PyArrayObject *angs, *bHat_l, *eHat_l, *rMat_c;
    PyArrayObject *dVec_c = NULL;
    double chi;
    npy_intp nvecs;
    angles_to_dvec_params fn_params;
    double bHat[3], eHat[3];
    size_t chunk_count;
    /* Parse arguments */
    if (!PyArg_ParseTuple(args,"OOOdO",
                          &angs,
                          &bHat_l, &eHat_l,
                          &chi, &rMat_c)) return(NULL);
    if ( angs == NULL ) return(NULL);


    /* Verify dimensions of input arrays */

    /* Argument error checking...  */
    if (!PyArray_Check(angs) || !PyArray_Check(bHat_l) ||
        !PyArray_Check(eHat_l) || !PyArray_Check(rMat_c))
        goto error;
    
    /* angs must be a contiguous array of [theta, eta, omega] angles. */
    if ((2 != PyArray_NDIM(angs)) || (3 != PyArray_DIMS(angs)[1]))
        goto error;
        
    /* check and extract the C rotation matrix -converts floats- */
    if (0 != extract_mat33_d_from_pyarray(&fn_params.rMat_c[0], rMat_c))
        goto error;

    if (0 != extract_vec3_d_from_pyarray(bHat, bHat_l))
        goto error;

    if (0 != extract_vec3_d_from_pyarray(eHat, eHat_l))
        goto error;

    nvecs = PyArray_DIMS(angs)[0];
    /* Allocate C-style array for return data */
    {
        npy_intp rdims[] = { nvecs, 3 };
        dVec_c = (PyArrayObject*)PyArray_EMPTY(2,rdims, NPY_DOUBLE, 0);
    }

    if (nvecs < 1)
        goto end_ok;
    
    Py_BEGIN_ALLOW_THREADS

    make_beam_rmat(bHat, eHat, &fn_params.rMat_e[0]);
    fn_params.chi = chi;
    fn_params.chunk_size = DEFAULT_CHUNK_SIZE;
    fn_params.total_count = nvecs;

    fn_params.angs = (const double *)PyArray_DATA(angs);
    fn_params.dVec_c = (double *)PyArray_DATA(dVec_c);

    chunk_count = 1 + ((nvecs - 1) / DEFAULT_CHUNK_SIZE);


#if defined(USE_METAL) && USE_METAL
    if (0 == metal_support_init()) {
        int res;
        res = metal_support_angles_to_dvec(&fn_params);
        metal_support_release();
        if (0 == res) {
            Py_BLOCK_THREADS
            goto end_ok;
        } else {
            Py_BLOCK_THREADS
            goto metal_error;
        }
    } else {
        Py_BLOCK_THREADS
        goto metal_error;
    }
    
#endif /* USE_METAL */

#if defined(USE_MEMORY_ONLY_KERNEL) && USE_MEMORY_ONLY_KERNEL
#  define ANGLES_TO_DVEC_KERNEL angles_to_dvec_memonly_chunked
#elif defined(USE_FLOAT_KERNEL) && USE_FLOAT_KERNEL
#  define ANGLES_TO_DVEC_KERNEL angles_to_dvec_sp_chunked
#else
#  define ANGLES_TO_DVEC_KERNEL angles_to_dvec_chunked
#endif

    if (chunk_count > 1) {
        xrd_transforms_task_apply(chunk_count,
                                  &fn_params, ANGLES_TO_DVEC_KERNEL);
    } else {
        ANGLES_TO_DVEC_KERNEL(&fn_params, 0);
    }


#undef ANGLES_TO_DVEC_KERNEL

    Py_END_ALLOW_THREADS
 end_ok:
    return ((PyObject*)dVec_c);
#if defined(USE_METAL) && USE_METAL
 metal_error:
    PyErr_SetString(PyExc_RuntimeError, "Failed execution of metal compute kernel");
    return NULL;
#endif /* USE_METAL */
 error:
    PyErr_SetString(PyExc_TypeError, "Bad types in call to 'angles_to_dvec'");
    return NULL;
}

#endif /* XRD_INCLUDE_PYTHON_WRAPPERS */
