#ifndef TRANSFORMS_UTILS_H
#define TRANSFORMS_UTILS_H

#if !defined(XRD_SINGLE_COMPILE_UNIT) || !XRD_SINGLE_COMPILE_UNIT
#  include "transforms_types.h"
#endif


static inline double
vector3_dot(const vector3* lhs, const vector3* rhs)
{
    return lhs->e[0]*rhs->e[0] + lhs->e[1]*rhs->e[1] + lhs->e[2]*rhs->e[2];
}

static inline double
dot3ll(const double *lhs, const double *rhs)
{
    return lhs[0]*rhs[0] + lhs[1]*rhs[1] + lhs[2]*rhs[2];
}

static inline double
dot3ls(const double *lhs, const double *rhs, size_t stride)
{
    return lhs[0]*rhs[0] + lhs[1]*rhs[stride] + lhs[2]*rhs[2*stride];    
}

static inline double
dot3sl(const double *lhs, size_t lhs_stride, const double *rhs)
{
    return lhs[0]*rhs[0] + lhs[lhs_stride]*rhs[1] + lhs[2*lhs_stride]*rhs[2];    
}

static inline double
dot3ss(const double *lhs, size_t lhs_stride, const double *rhs, size_t rhs_stride)
{
    return lhs[0]*rhs[0] +
        lhs[lhs_stride]*rhs[rhs_stride] +
        lhs[2*lhs_stride]*rhs[2*rhs_stride];    
}

static inline void
vec3_mat33_product(double * restrict result, const double *lhs, const double *rhs)
{
    result[0] = dot3ls(lhs, rhs+0, 3);
    result[1] = dot3ls(lhs, rhs+1, 3);
    result[2] = dot3ls(lhs, rhs+2, 3);
}

static inline void
vec3_mat33t_product(double * restrict result, const double *lhs, const double *rhs)
{
    result[0] = dot3ll(lhs, rhs+0);
    result[1] = dot3ll(lhs, rhs+3);
    result[2] = dot3ll(lhs, rhs+6);
}

static inline void
mat33_mat33_product(double * restrict result, const double *lhs, const double *rhs)
{
    result[0] = dot3ls(lhs+0, rhs+0, 3);
    result[1] = dot3ls(lhs+0, rhs+1, 3);
    result[2] = dot3ls(lhs+0, rhs+2, 3);
    result[3] = dot3ls(lhs+3, rhs+0, 3);
    result[4] = dot3ls(lhs+3, rhs+1, 3);
    result[5] = dot3ls(lhs+3, rhs+2, 3);
    result[6] = dot3ls(lhs+6, rhs+0, 3);
    result[7] = dot3ls(lhs+6, rhs+1, 3);
    result[8] = dot3ls(lhs+6, rhs+2, 3);
}


static inline void
vector3_normalized(const vector3* in, vector3* restrict out)
{
    int i;
    double sqr_norm = vector3_dot(in, in);

    if (sqr_norm > epsf) {
        double recip_norm = 1.0/sqrt(sqr_norm);
        for (i=0; i<3; ++i)
            out->e[i] = in->e[i] * recip_norm;
    } else {
        *out = *in;
    }
}

static inline
void matrix33_set_identity(matrix33* restrict out)
{
    int i;
    for (i = 0; i < 9; ++i)
        out->e[i] = (i%4 == 0)? 1.0: 0.0;
}

/* single precision versions (as needed) */
static inline float
dot3ll_f(const float *lhs, const float *rhs)
{
    return lhs[0]*rhs[0] + lhs[1]*rhs[1] + lhs[2]*rhs[2];
}

static inline float
dot3ls_f(const float *lhs, const float *rhs, size_t stride)
{
    return lhs[0]*rhs[0] + lhs[1]*rhs[stride] + lhs[2]*rhs[2*stride];    
}

static inline void
vec3_mat33_product_f(float * restrict result, const float *lhs, const float *rhs)
{
    result[0] = dot3ls_f(lhs, rhs+0, 3);
    result[1] = dot3ls_f(lhs, rhs+1, 3);
    result[2] = dot3ls_f(lhs, rhs+2, 3);
}

static inline void
vec3_mat33t_product_f(float * restrict result, const float *lhs, const float *rhs)
{
    result[0] = dot3ll_f(lhs, rhs+0);
    result[1] = dot3ll_f(lhs, rhs+3);
    result[2] = dot3ll_f(lhs, rhs+6);
}

#endif /* TRANSFORMS_UTILS_H */

