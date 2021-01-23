#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

static id<MTLDevice> transforms_metal_device = nil;

/* HACK HACK HACK */
typedef struct {
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
} angles_to_dvec_params;
/* END HACK HACK HACK */


extern BOOL NSDebugEnabled;
int
metal_support_init(void)
{
    NSDebugEnabled = YES;
    static id<MTLDevice> device;
    device = MTLCreateSystemDefaultDevice();
    if (nil == device) {
        NSLog(@"Cannot find Metal Default Device.");
        goto error;
    }

    /* copy to globals */
    transforms_metal_device = device;

    return 0;
 error:
    return -1;
}

void
metal_support_release(void)
{
    transforms_metal_device = nil;
}


static NSString* angles_to_dvec_kernel_src = @""
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"struct angles_to_dvec_uniforms {\n"
"    float3x3 rMat_c;\n"
"    float3x3 rMat_e;\n"
"    float sin_chi;\n"
"    float cos_chi;\n"
"};\n"
"\n"
"kernel void\n"
"angle_to_dvec(device float* dVec_c_out,\n"
"              device const float* angs_in,\n"
"              constant angles_to_dvec_uniforms& uniforms,\n"
"              uint tid [[thread_position_in_grid]])\n"
"{\n"
"    // angles come in theta, eta, omega order\n"
"    float3 angs = float3(angs_in[tid*3], angs_in[tid*3+1], angs_in[tid*3+2]);\n"
"    float3 sin_angs = sin(angs);\n"
"    float3 cos_angs = cos(angs);\n"
"\n"
"    float3 gVec_e = float3(sin_angs[0] * cos_angs[1],\n"
"                           sin_angs[0] * sin_angs[1],\n"
"                          -cos_angs[0]);\n"
"\n"
"    /* make_sample_rmat_polar. Note metal builds in matrices in column order */\n"
"    float3x3 rMat_s = float3x3(float3(cos_angs[2],\n"
"                                      uniforms.sin_chi * sin_angs[2],\n"
"                                     -uniforms.cos_chi * sin_angs[2]),\n"
"\n"
"                               float3(0.0,\n"
"                                      uniforms.cos_chi,\n"
"                                      uniforms.sin_chi),\n"
"\n"
"                               float3(sin_angs[2],\n"
"                                     -uniforms.sin_chi * cos_angs[2],\n"
"                                      uniforms.cos_chi * cos_angs[2]));\n"
"\n"
"    /* actual transforms are made easy by metal matrix operations */\n"
"    float3 gVec_l = gVec_e*uniforms.rMat_e;\n"
"    float3 vec3_tmp = gVec_l*rMat_s;\n"
"    float3 vec3_result = vec3_tmp*uniforms.rMat_c;\n"
"\n"
"    /* copy results to out stream */ \n"
"    dVec_c_out[tid*3] = vec3_result[0];\n"
"    dVec_c_out[tid*3+1] = vec3_result[1];\n"
"    dVec_c_out[tid*3+2] = vec3_result[2];\n"
"}\n";

typedef struct angles_to_dvec_uniforms_tag
{
    float rMat_c[9];
    float rMat_e[9];
    float sin_chi;
    float cos_chi;
} angles_to_dvec_uniforms;

int
metal_support_angles_to_dvec(const angles_to_dvec_params *params)
{
    id<MTLDevice> device = transforms_metal_device;
    MTLCompileOptions *compile_options = [MTLCompileOptions new];
    NSError *compile_error = nil;

    id<MTLLibrary> lib = [device newLibraryWithSource: angles_to_dvec_kernel_src
                                              options: compile_options
                                                error: &compile_error];
    if (nil == lib) {
        NSLog(@"Failed to create angles_to_dvec_kernel for metal:\n%@",
                compile_error);
        goto error;
    }
    if (nil != compile_error) {
        NSLog(@"Compile error: %@", compile_error);
    }

    id<MTLFunction> angles_to_dvec_kernel;
    angles_to_dvec_kernel = [lib newFunctionWithName: @"angle_to_dvec"];
    if (nil == angles_to_dvec_kernel) {
        NSLog(@"angle_to_dvec not found in lib. Lib has: %@", lib.functionNames);
        NSLog(@"Failed to find the angle_to_dvec kernel.");
        goto error;
    }

    id<MTLComputePipelineState> angles_to_dvec_PSO;
    NSError *create_pipeline_error = nil;
    angles_to_dvec_PSO = [device newComputePipelineStateWithFunction: angles_to_dvec_kernel
                                                               error: &create_pipeline_error];
    if (nil == angles_to_dvec_PSO) {
        NSLog(@"Failed to create pipeline state object with error:\n%@",
              create_pipeline_error);
        goto error;
    }

    id<MTLBuffer> angs_in_buffer, dvecs_c_out_buffer, uniforms_buffer;
    size_t angs_in_size = sizeof(float)*3*params->total_count;
    size_t dvecs_c_out_size = sizeof(float)*3*params->total_count;
    size_t uniforms_buffer_size = sizeof(angles_to_dvec_uniforms);
    angs_in_buffer = [device newBufferWithLength: angs_in_size
                                         options: MTLResourceStorageModeShared];
    dvecs_c_out_buffer = [device newBufferWithLength: dvecs_c_out_size
                                             options: MTLResourceStorageModeShared];
    uniforms_buffer = [device newBufferWithLength: uniforms_buffer_size
                                          options: MTLResourceStorageModeShared];

    /* copy data into input buffer */
    {
        double *src = params->angs;
        float *dst = angs_in_buffer.contents;
        size_t count = 3*params->total_count;
        size_t i;
        for (i=0; i<count; i++) {
            dst[i] = (float)src[i];
        }
    }

    /* copy data into uniforms buffer */
    {
        angles_to_dvec_uniforms *uniforms = uniforms_buffer.contents;
        size_t i;

        for (i=0; i<9; i++) {
            uniforms->rMat_c[i] = (float)params->rMat_c[i];
        }

        for (i=0; i<9; i++) {
            uniforms->rMat_e[i] = (float)params->rMat_e[i];
        }

        uniforms->sin_chi = (float)sin(params->chi);
        uniforms->cos_chi = (float)cos(params->chi);
    }
    
    id<MTLCommandQueue> cq;
    cq = [device newCommandQueue];
    if (nil == cq) {
        NSLog(@"Failed to create the command queue.");
        goto error;
    }

    id<MTLCommandBuffer> cb = [cq commandBuffer];
    if (nil == cb) {
        NSLog(@"Failed to create a command buffer.");
        goto error;
    }

    id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
    if (nil == encoder) {
        NSLog(@"Failed to create a command encoder.");
        goto error;
    }

    [encoder setComputePipelineState: angles_to_dvec_PSO];
    [encoder setBuffer: dvecs_c_out_buffer offset: 0 atIndex: 0];
    [encoder setBuffer: angs_in_buffer offset: 0 atIndex: 1];
    [encoder setBuffer: uniforms_buffer offset: 0 atIndex: 2];

    MTLSize grid_size = MTLSizeMake(params->total_count, 1, 1);

    NSUInteger threadgroup_size = angles_to_dvec_PSO.maxTotalThreadsPerThreadgroup;
    if (threadgroup_size > params->total_count) {
        threadgroup_size = params->total_count;
    }
    MTLSize tg_size = MTLSizeMake(threadgroup_size, 1, 1);
    
    [encoder dispatchThreads: grid_size
       threadsPerThreadgroup: tg_size];

    [encoder endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    /* copy results out from the output buffer to memory */
    {
        float *src = dvecs_c_out_buffer.contents;
        double *dst = params->dVec_c;
        size_t count = 3*params->total_count;
        size_t i;

        for (i=0; i<count; i++) {
            dst[i] = (double)src[i];
        }
    }

    return 0;
 error:
    return -1;
}
