/*
 * %BEGIN LICENSE HEADER
 * %END LICENSE HEADER
 */


/* =============================================================================
 * THIS_MODULE_NAME is used in several places in macros to generate the module
 * entry point for Python 2 and Python 3, as well as for the internal module
 * name exported to Python.
 *
 * Having this as a define makes it easy to change as needed (only a single
 * place to modify
 *
 * Note the supporting CONCAT and STR macros...
 * =============================================================================
 */
#define THIS_MODULE_NAME _new_transforms_capi

#define _CONCAT(a,b) a ## b
#define CONCAT(a,b) _CONCAT(a,b)
#define _STR(a) # a
#define STR(a) _STR(a)

/* ************************************************************************** */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

/* =============================================================================
 * Note: As we really want to only the entry point of the module to be visible
 * in the exported library, but there is a clear convenience on having different
 * functions in their own source file, the approach will be to include use this
 * file as the compile unit and import the source code of all the functions.
 *
 * This file will also contain all the module scaffolding needed (like the init
 * function as well as the exported method list declaration.
 * =============================================================================
 */



/* =============================================================================
 * This macros configure the way the module is built.
 * All implementation files included in a single compile unit.
 * Python wrappers are to be included.
 * Visibility for all functions are wrappers is removed by turning them into
 * statics.
 * =============================================================================
 */
#define XRD_SINGLE_COMPILE_UNIT 1
#define XRD_INCLUDE_PYTHON_WRAPPERS 1
#define XRD_TASK_SYSTEM 1
#define XRD_CFUNCTION static inline
#define XRD_PYTHON_WRAPPER static

#include "transforms_types.h"
#include "transforms_utils.h"
#include "transforms_prototypes.h"
#include "checks.h"
#include "transforms_task_system.h"


#include "angles_to_gvec.c"
#include "angles_to_dvec.c"
#include "gvec_to_xy.c"
#include "xy_to_gvec.c"
#include "oscill_angles_of_HKLs.c"
#include "unit_row_vector.c"
/* #include "make_detector_rmat.c" */
#include "make_sample_rmat.c"
#include "make_rmat_of_expmap.c"
#include "make_binary_rmat.c"
#include "make_beam_rmat.c"
#include "validate_angle_ranges.c"
#include "rotate_vecs_about_axis.c"
#include "quat_distance.c"


/* TEST TEST TEST TEST */
#include "transforms_task_libdispatch.c"


/* =============================================================================
 * Module initialization
 * =============================================================================
 */

/*#define EXPORT_METHOD(name)                                           \
    { STR(name), CONCAT(python_, name), METH_VARARGS, CONCAT(docstring_, name) }
*/
#define EXPORT_METHOD_VA(name) \
    { STR(name), CONCAT(python_, name), METH_VARARGS, "" }

static PyMethodDef _module_methods[] = {
    EXPORT_METHOD_VA(anglesToGVec), /* angles_to_gvec */
    EXPORT_METHOD_VA(anglesToDVec), /* angles_to_dvec */
    EXPORT_METHOD_VA(gvecToDetectorXY),  /* gvec_to_xy */
    EXPORT_METHOD_VA(gvecToDetectorXYArray), /* gvec_to_xy */
    EXPORT_METHOD_VA(detectorXYToGvec), /* xy_to_gvec */
    EXPORT_METHOD_VA(oscillAnglesOfHKLs), /* solve_omega */
    EXPORT_METHOD_VA(unitRowVector), /* unit_vector */
    EXPORT_METHOD_VA(unitRowVectors), /* unit_vector */
    EXPORT_METHOD_VA(makeOscillRotMat), /* make_sample_rmat */
    EXPORT_METHOD_VA(makeOscillRotMatArray), /* make_sample_rmat */
    EXPORT_METHOD_VA(makeRotMatOfExpMap), /* make_rmat_of_expmap */
    EXPORT_METHOD_VA(makeBinaryRotMat), /* make_binary_rmat */
    EXPORT_METHOD_VA(makeEtaFrameRotMat), /* make_beam_rmat */
    EXPORT_METHOD_VA(validateAngleRanges), /* validate_angle_ranges */
    EXPORT_METHOD_VA(rotate_vecs_about_axis), /* rotate_vecs_about_axis */
    EXPORT_METHOD_VA(quat_distance), /* quat_distance */

    {NULL,NULL,0,NULL} /* sentinel */
};

/*
 * In Python 3 the entry point for a C module changes slightly, but both can
 * be supported with little effort with some conditionals and macro magic
 */

#if PY_VERSION_HEX >= 0x03000000
#  define IS_PY3K
#endif

#if defined(IS_PY3K)
/* a module definition structure is required in Python 3 */
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        STR(THIS_MODULE_NAME),
        NULL,
        -1,
        _module_methods,
        NULL,
        NULL,
        NULL,
        NULL
};
#endif

#if defined(IS_PY3K)
#  define MODULE_INIT_FUNC_NAME CONCAT(PyInit_, THIS_MODULE_NAME)
#  define MODULE_RETURN(module) return module
#else
#  define MODULE_INIT_FUNC_NAME CONCAT(init, THIS_MODULE_NAME)
#  define MODULE_RETURN(module) return
#endif
PyMODINIT_FUNC
MODULE_INIT_FUNC_NAME(void)
{
    PyObject *module;

#if defined(IS_PY3K)
    module = PyModule_Create(&moduledef);
#else
    module = Py_InitModule(STR(THIS_MODULE_NAME),_module_methods);
#endif
    if (NULL == module)
        MODULE_RETURN(module);

    import_array();
    
    MODULE_RETURN(module);
}

