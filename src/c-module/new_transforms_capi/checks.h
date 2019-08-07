


/* returns non-zero if the object is a numpy array that with shape (elements,),
   dtype is double and is in C order.
*/
static inline int
is_valid_vector(PyObject* op, npy_intp elements)
{
    PyArrayObject *aop = (PyArrayObject *)op;
    return PyArray_Check(op) &&
        1 == PyArray_NDIM(aop) &&
        elements == PyArray_SHAPE(aop)[0] &&
        NPY_FLOAT64 == PyArray_TYPE(aop) &&
        PyArray_ISCARRAY(aop);      
}

/* returns non-zero if the object is a numpy array with shape (X, elements),
   dtype is double and is in C order. X can be any positive integer.
*/
static inline int
is_valid_vector_array(PyObject *op, npy_intp elements)
{
    PyArrayObject *aop = (PyArrayObject *)op;    
    return PyArray_Check(op) &&
        2 == PyArray_NDIM(aop) &&
        0 < PyArray_SHAPE(aop)[0] &&
        elements == PyArray_SHAPE(aop)[1] &&
        NPY_FLOAT64 == PyArray_TYPE(aop) &&
        PyArray_ISCARRAY(aop);      
}


/* returns non-zero if the object is a numpy array with shape (outerd, innerd),
   dtype is double and is in C order.
*/
static inline int
is_valid_matrix(PyObject *op, npy_intp outerd, npy_intp innerd)
{
    PyArrayObject *aop = (PyArrayObject *)op;    
    return PyArray_Check(op) &&
        2 == PyArray_NDIM(aop) &&
        outerd == PyArray_SHAPE(aop)[0] &&
        innerd == PyArray_SHAPE(aop)[1] &&
        NPY_FLOAT64 == PyArray_TYPE(aop) &&
        PyArray_ISCARRAY(aop);      
}

/* returns non-zero if the object is a numpy array with shape (X, outerd,
   innerd), dtype is double and is in C order. X can be any positive integer.
*/
static inline int
is_valid_matrix_array(PyObject *op, npy_intp outerd, npy_intp innerd)
{
    PyArrayObject *aop = (PyArrayObject *)op;    
    return PyArray_Check(op) &&
        3 == PyArray_NDIM(aop) &&
        0 < PyArray_SHAPE(aop)[0] &&
        outerd == PyArray_SHAPE(aop)[1] &&
        innerd == PyArray_SHAPE(aop)[2] &&
        NPY_FLOAT64 == PyArray_TYPE(aop) &&
        PyArray_ISCARRAY(aop);      
}

static inline void
raise_value_error(const char *why)
{
    PyErr_SetString(PyExc_ValueError, why);
}

static inline void
raise_runtime_error(const char *why)
{
    PyErr_SetString(PyExc_RuntimeError, why);
}


/*
 * These are used so that actual checks can be done by the ParseTuple itself.
 *
 * Using some structs for the result allow piggybacking some extra information
 * to the converter and back (like the argument name to improve error reporting).
 */

typedef struct {
    const char *name;
    double *data;
} named_vector3;

static int
vector3_converter(PyObject *op, void *result)
{
    named_vector3 *res = (named_vector3 *) result;

    if (is_valid_vector(op, 3))
    {
        res->data = PyArray_DATA(op);
        return 1;
    }
    else
    {
        PyErr_Format(PyExc_ValueError,
                     "%s expected to be a (3,) float64 array", res->name);
        return 0;
    }
}
