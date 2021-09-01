#ifndef TRANSFORMS_UTILS_H
#define TRANSFORMS_UTILS_H

#if !defined(XRD_SINGLE_COMPILE_UNIT) || !XRD_SINGLE_COMPILE_UNIT
#  include "transforms_types.h"
#endif

static const double
m33_identity[3][3] = { { 1.0, 0.0, 0.0 },
                       { 0.0, 1.0, 0.0 },
                       { 0.0, 0.0, 1.0 } };

static inline
void
m33_set_identity(double *dst)
{
    memcpy(dst, m33_identity, sizeof(m33_identity));
}



static inline
double *
v3_v3s_inplace_add(double *dst_src1,
                   const double *src2, ptrdiff_t stride)
{
    dst_src1[0] += src2[0];
    dst_src1[1] += src2[1*stride];
    dst_src1[2] += src2[2*stride];
    return dst_src1;
}

static inline
double *
v3_v3s_add(const double *src1,
           const double *src2, ptrdiff_t stride,
           double * restrict dst)
{
    dst[0] = src1[0] + src2[0];
    dst[1] = src1[1] + src2[1*stride];
    dst[2] = src1[2] + src2[2*stride];

    return dst;
}

static inline
double *
v3_v3s_inplace_sub(double *dst_src1,
                   const double *src2, ptrdiff_t stride)
{
    dst_src1[0] -= src2[0];
    dst_src1[1] -= src2[1*stride];
    dst_src1[2] -= src2[2*stride];
    return dst_src1;
}

static inline
double *
v3_v3s_sub(const double *src1,
           const double *src2, ptrdiff_t stride,
           double * restrict dst)
{
    dst[0] = src1[0] - src2[0];
    dst[1] = src1[1] - src2[1*stride];
    dst[2] = src1[2] - src2[2*stride];

    return dst;
}

static inline
double *
v3_inplace_normalize(double * restrict v)
{
    double sqr_norm = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];

    if (sqr_norm > epsf) {
        double normalize_factor = 1./sqrt(sqr_norm);
        v[0] *= normalize_factor;
        v[1] *= normalize_factor;
        v[2] *= normalize_factor;
    }

    return v;
}

static inline
double *
v3_normalize(const double *in,
             double * restrict out)
{
    double in0 = in[0], in1 = in[1], in2 = in[2];
    double sqr_norm = in0*in0 + in1*in1 + in2*in2;

    if (sqr_norm > epsf) {
        double normalize_factor = 1./sqrt(sqr_norm);
        out[0] = in0 * normalize_factor;
        out[1] = in1 * normalize_factor;
        out[2] = in2 * normalize_factor;
    } else {
        out[0] = in0;
        out[1] = in1;
        out[2] = in2;
    }

    return out;
}

static inline
double *
m33_inplace_transpose(double * restrict m)
{
    double e1 = m[1];
    double e2 = m[2];
    double e5 = m[5];
    m[1] = m[3];
    m[2] = m[6];
    m[5] = m[7];
    m[3] = e1;
    m[6] = e2;
    m[7] = e5;

    return m;
}

static inline
double *
m33_transpose(const double *m,
              double * restrict dst)
{
    dst[0] = m[0]; dst[1] = m[3]; dst[2] = m[6];
    dst[3] = m[1]; dst[4] = m[4]; dst[5] = m[7];
    dst[7] = m[2]; dst[8] = m[5]; dst[9] = m[9];

    return dst;
}

static inline
double
v3_v3s_dot(const double *v1,
           const double *v2, ptrdiff_t stride)
{
    return v1[0]*v2[0] + v1[1]*v2[stride] + v1[2]*v2[2*stride];
}


/* 3x3 matrix by strided 3 vector product -------------------------------------
   hopefully a constant stride will be optimized
 */
static inline
double *
m33_v3s_multiply(const double *m,
                 const double *v, ptrdiff_t stride,
                 double * restrict dst)
{
    dst[0] = m[0]*v[0] + m[1]*v[stride] + m[2]*v[2*stride];
    dst[1] = m[3]*v[0] + m[4]*v[stride] + m[5]*v[2*stride];
    dst[2] = m[6]*v[0] + m[7]*v[stride] + m[8]*v[2*stride];

    return dst;
}

/* transposed 3x3 matrix by strided 3 vector product --------------------------
 */
static inline
double *
v3s_m33t_multiply(const double *v, ptrdiff_t stride,
                  const double *m,
                  double * restrict dst)
{
    double v0 = v[0]; double v1 = v[stride]; double v2 = v[2*stride];
    dst[0] = v0*m[0] + v1*m[1] + v2*m[2];
    dst[1] = v0*m[3] + v1*m[4] + v2*m[5];
    dst[2] = v0*m[6] + v1*m[7] + v2*m[8];

    return dst;
}

static inline
double *
v3s_m33_multiply(const double *v, ptrdiff_t stride,
                 const double *m,
                 double * restrict dst)
{
    double v0 = v[0]; double v1 = v[stride]; double v2 = v[2*stride];
    dst[0] = v0*m[0] + v1*m[3] + v2*m[6];
    dst[1] = v0*m[1] + v1*m[4] + v2*m[7];
    dst[2] = v0*m[2] + v1*m[5] + v2*m[8];

    return dst;
}

static inline
double *
m33t_v3s_multiply(const double *m,
                  const double *v, ptrdiff_t stride,
                  double * restrict dst)
{
    dst[0] = m[0]*v[0] + m[3]*v[stride] + m[6]*v[2*stride];
    dst[1] = m[1]*v[0] + m[4]*v[stride] + m[7]*v[2*stride];
    dst[2] = m[2]*v[0] + m[5]*v[stride] + m[8]*v[2*stride];

    return dst;
}

static inline
double *
m33_m33_multiply(const double *src1,
                 const double *src2,
                 double * restrict dst)
{
    v3s_m33_multiply(src1 + 0, 1, src2, dst+0);
    v3s_m33_multiply(src1 + 3, 1, src2, dst+3);
    v3s_m33_multiply(src1 + 6, 1, src2, dst+6);

    return dst;
}

static inline
double *
m33t_m33_multiply(const double *src1,
                  const double *src2,
                  double * restrict dst)
{
    v3s_m33_multiply(src1 + 0, 3, src2, dst+0);
    v3s_m33_multiply(src1 + 1, 3, src2, dst+3);
    v3s_m33_multiply(src1 + 2, 3, src2, dst+6);

    return dst;
}

static inline
double *
m33_m33t_multiply(const double *src1,
                  const double *src2,
                  double * restrict dst)
{
    return m33_inplace_transpose(m33t_m33_multiply(src2, src1, dst));
}

static inline
double *
m33t_m33t_multiply(const double *src1,
                   const double *src2,
                   double * restrict dst)
{
    return m33_inplace_transpose(m33_m33_multiply(src2, src1, dst));
}

#endif /* TRANSFORMS_UTILS_H */

