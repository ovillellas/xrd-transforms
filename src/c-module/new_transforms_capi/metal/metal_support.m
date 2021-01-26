#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <Accelerate/Accelerate.h>
#import <os/signpost.h>

static const size_t metal_command_buffers_inflight_count = 3;
static const size_t metal_buffer_max_size = 512*1024;
static const size_t metal_buffer_uniforms_size = 2*1024;
static NSString* angles_to_dvec_kernel_src = @""
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"struct angles_to_dvec_uniforms {\n"
"    packed_float3 rMat_c[3];\n"
"    packed_float3 rMat_e[3];\n"
"    float sin_chi;\n"
"    float cos_chi;\n"
"};\n"
"\n"
"kernel void\n"
"angle_to_dvec(device packed_float3* dVec_c_out,\n"
"              device const packed_float3* angs_in,\n"
"              constant angles_to_dvec_uniforms& uniforms,\n"
"              uint tid [[thread_position_in_grid]])\n"
"{\n"
"    // angles come in theta, eta, omega order\n"
"    float3 sin_angs = sin(angs_in[tid]);\n"
"    float3 cos_angs = cos(angs_in[tid]);\n"
"\n"
"    float3 gVec_e = float3(sin_angs[0] * cos_angs[1],\n"
"                           sin_angs[0] * sin_angs[1],\n"
"                          -cos_angs[0]);\n"
"\n"
"    /* make_sample_rmat_polar. Note metal builds in matrices in column order */\n"
"    float sin_chi = uniforms.sin_chi;\n"
"    float cos_chi = uniforms.cos_chi;\n"
"    float3x3 rMat_s = float3x3(float3(cos_angs[2],\n"
"                                      sin_chi * sin_angs[2],\n"
"                                     -cos_chi * sin_angs[2]),\n"
"\n"
"                               float3(0.0,\n"
"                                      cos_chi,\n"
"                                      sin_chi),\n"
"\n"
"                               float3(sin_angs[2],\n"
"                                     -sin_chi * cos_angs[2],\n"
"                                      cos_chi * cos_angs[2]));\n"
"\n"
"    float3x3 rMat_e = float3x3(uniforms.rMat_e[0], uniforms.rMat_e[1], uniforms.rMat_e[2]);\n"
"    float3x3 rMat_c = float3x3(uniforms.rMat_c[0], uniforms.rMat_c[1], uniforms.rMat_c[2]);\n"
"    /* actual transforms are made easy by metal matrix operations */\n"
"    float3 gVec_l = gVec_e*rMat_e;\n"
"    float3 vec3_tmp = gVec_l*rMat_s;\n"
"    float3 vec3_result = vec3_tmp*rMat_c;\n"
"\n"
"    /* copy results to out stream */ \n"
"    dVec_c_out[tid] = vec3_result;\n"
"}\n";


/* state for Metal */
static id<MTLDevice> metal_transforms_device = nil;
static id<MTLComputePipelineState> metal_transforms_pso_angles_to_dvec = nil;
static id<MTLBuffer> metal_transforms_kernel_input_buffer[metal_command_buffers_inflight_count] = {nil};
static id<MTLBuffer> metal_transforms_kernel_output_buffer[metal_command_buffers_inflight_count] = {nil};
static id<MTLBuffer> metal_transforms_kernel_uniforms_buffer = nil;
static id<MTLCommandQueue> metal_transforms_command_queue = nil;

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

static void
dump_angles_to_dvec_params(const angles_to_dvec_params *params)
{
    NSLog(@"angles_to_dvec_params (%p):\n"
          "rMat_c:\n"
          "%6.4f %6.4f %6.4f\n"
          "%6.4f %6.4f %6.4f\n"
          "%6.4f %6.4f %6.4f\n"
          "rMat_e:\n"
          "%6.4f %6.4f %6.4f\n"
          "%6.4f %6.4f %6.4f\n"
          "%6.4f %6.4f %6.4f\n"
          "chi: %6.4f\n"
          "chunk_size: %zu total_count: %zu\n"
          "angs: [%p - %p]\n"
          "results: [%p - %p]\n",
          params,
          params->rMat_c[0], params->rMat_c[1], params->rMat_c[2],
          params->rMat_c[3], params->rMat_c[4], params->rMat_c[5],
          params->rMat_c[6], params->rMat_c[7], params->rMat_c[8],
          params->rMat_e[0], params->rMat_e[1], params->rMat_e[2],
          params->rMat_e[3], params->rMat_e[4], params->rMat_e[5],
          params->rMat_e[6], params->rMat_e[7], params->rMat_e[8],
          params->chi,
          params->chunk_size, params->total_count,
          params->angs, params->angs + 3*params->total_count,
          params->dVec_c, params->dVec_c + 3*params->total_count);
}

static void
dump_floats(const char *name, const float *floats, size_t count)
{
    switch(count) {
    case 0:
        NSLog(@"%10s: <no floats>", name);
        break;
    case 1:
        NSLog(@"%10s: %6.4f", name, floats[0]);
        break;
    case 2:
        NSLog(@"%10s: %6.4f %6.4f",
              name, floats[0], floats[1]);
        break;
    case 3:
        NSLog(@"%10s: %6.4f %6.4f %6.4f",
              name, floats[0], floats[1], floats[2]);
        break;
    case 4:
        NSLog(@"%10s: %6.4f %6.4f %6.4f %6.4f",
              name, floats[0], floats[1], floats[2], floats[3]);
        break;
    default:
        NSLog(@"%10s: %6.4f %6.4f %6.4f ... %6.4f <%zu total>",
              name, floats[0], floats[1], floats[2], floats[count-1], count);
    }
}

static void
dump_doubles(const char *name, const double *floats, size_t count)
{
    switch(count) {
    case 0:
        NSLog(@"%10s: <no floats>", name);
        break;
    case 1:
        NSLog(@"%10s: %6.4f", name, floats[0]);
        break;
    case 2:
        NSLog(@"%10s: %6.4f %6.4f",
              name, floats[0], floats[1]);
        break;
    case 3:
        NSLog(@"%10s: %6.4f %6.4f %6.4f",
              name, floats[0], floats[1], floats[2]);
        break;
    case 4:
        NSLog(@"%10s: %6.4f %6.4f %6.4f %6.4f",
              name, floats[0], floats[1], floats[2], floats[3]);
        break;
    default:
        NSLog(@"%10s: %6.4f %6.4f %6.4f ... %6.4f <%zu total>",
              name, floats[0], floats[1], floats[2], floats[count-1], count);
    }
}

static void
dump_available_devices(void)
{
    NSArray<id<MTLDevice>> *deviceList = nil;

    deviceList = MTLCopyAllDevices();
    NSLog(@"Devices: %@", deviceList);
}

void
metal_support_release(void)
{
    for (size_t idx = 0; idx < metal_command_buffers_inflight_count; idx++)
        metal_transforms_kernel_input_buffer[idx] = nil;
    for (size_t idx = 0; idx < metal_command_buffers_inflight_count; idx++)
        metal_transforms_kernel_output_buffer[idx] = nil;
    metal_transforms_kernel_uniforms_buffer = nil;
    metal_transforms_pso_angles_to_dvec = nil;
    metal_transforms_device = nil;
}

int
metal_support_init(void)
{
    if (nil != metal_transforms_device)
        return 0; // already initialized

    /* dump_available_devices();*/
    id<MTLDevice> device;
    device = MTLCreateSystemDefaultDevice();
    if (nil == device) {
        NSLog(@"Cannot find Metal Default Device.");
        goto error;
    }

    MTLCompileOptions *compile_options = [MTLCompileOptions new];
    NSError *compile_error = nil;

    id<MTLLibrary> lib = [device newLibraryWithSource: angles_to_dvec_kernel_src
                                              options: compile_options
                                                error: &compile_error];

    if (nil == lib) {
        NSLog(@"Failed to create angles_to_dvec_kernel:\n%@", compile_error);
        goto error;
    }

    if (nil != compile_error) {
        /* Sometimes compile fails but newLibraryWithSource returns non-nil (!) */
        NSLog(@"Metal library compile error: %@", compile_error);
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

    static id<MTLBuffer> input_buffer[metal_command_buffers_inflight_count];
    static id<MTLBuffer> output_buffer[metal_command_buffers_inflight_count];
    static id<MTLBuffer> uniforms_buffer;
    int failed_buffers= 0;

    for (size_t idx = 0; idx < metal_command_buffers_inflight_count; idx++) {
        input_buffer[idx] = [device newBufferWithLength: metal_buffer_max_size
                                                options: MTLResourceStorageModeShared];
        if (nil != input_buffer[idx])
            input_buffer[idx].label = [NSString stringWithFormat: @"transforms input [%zu]", idx];
        else
            failed_buffers++;
    }

    for (size_t idx = 0; idx < metal_command_buffers_inflight_count; idx++) {
        output_buffer[idx] = [device newBufferWithLength: metal_buffer_max_size
                                                 options: MTLResourceStorageModeShared];
        if (nil != output_buffer[idx])
            output_buffer[idx].label = [NSString stringWithFormat: @"transforms output [%zu]", idx];
        else
            failed_buffers++;
    }

    uniforms_buffer = [device newBufferWithLength: metal_buffer_uniforms_size
                                          options: MTLResourceStorageModeShared];
    uniforms_buffer.label = @"transforms uniforms";

    if (failed_buffers || nil == uniforms_buffer) {
        NSLog(@"Failed to allocate Metal buffers.");
        goto error;
    }

    id<MTLCommandQueue> cq = [device newCommandQueue];
    cq.label = @"transforms shared command queue";
    if (nil == cq) {
        NSLog(@"Failed to create the command queue.");
        goto error;
    }

    /* copy to globals */
    metal_transforms_device = device;
    metal_transforms_pso_angles_to_dvec = angles_to_dvec_PSO;
    for (size_t idx = 0; idx < metal_command_buffers_inflight_count; idx++)
        metal_transforms_kernel_input_buffer[idx] = input_buffer[idx];
    for (size_t idx = 0; idx < metal_command_buffers_inflight_count; idx++)
        metal_transforms_kernel_output_buffer[idx] = output_buffer[idx];
    metal_transforms_kernel_uniforms_buffer = uniforms_buffer;
    metal_transforms_command_queue = cq;
    return 0;
 error:
    return -1;
}

typedef struct angles_to_dvec_uniforms_tag
{
    float rMat_c[9];
    float rMat_e[9];
    float sin_chi;
    float cos_chi;
} angles_to_dvec_uniforms;

static uint8_t transpose_idx[] = { 0, 3, 6, 1, 4, 7, 2, 5, 8 };

/* Specification of a chunking operation */
typedef struct chunking_spec_tag {
    size_t chunk_count;
    size_t chunk_size;
    size_t total_size;
} chunking_spec;

/* A description of a chunk */
typedef struct chunk_desc_tag {
    size_t start;
    size_t size;
} chunk_desc;

static inline void
init_chunking_spec(chunking_spec *cs, size_t max_chunk_size, size_t total_size)
{
    size_t nchunks, chunk_size;
    if (total_size > max_chunk_size) {
        /* several chunks needed. Use the minimum amount of chunks possible. Try
           to have their size balanced as much as possible (last chunk may be
           shorter) */
        nchunks = (total_size-1)/max_chunk_size + 1;
        chunk_size = (total_size-1)/nchunks + 1;
    } else {
        /* one chunk is enough */
        nchunks = 1;
        chunk_size = total_size;
    }

    cs->chunk_count = nchunks;
    cs->chunk_size = chunk_size;
    cs->total_size = total_size;
}

/* Obtain a chunk description form a chunk spec and an index. This basically
   builds the span of the chunk number idx based on the chunking_spec. Note that
   the result is undefined for indices going beyond the number of indices in
   the chunk_spec.
*/
static inline void
init_chunk_from_chunking_spec_and_index(chunk_desc *cd, const chunking_spec *cs,
                                        size_t idx)
{
    size_t start = cs->chunk_size * idx;
    size_t end;
    if (idx < cs->chunk_count-1)
        end = start + cs->chunk_size;
    else
        end = cs->total_size;

    cd->start = start;
    cd->size = end - start;
}

static inline id<MTLCommandBuffer>
launch_chunk(const chunk_desc *cd, const angles_to_dvec_params* params, size_t buff_index)
{
    // NSLog(@"Launching <%zu> [%zu - %zu[", buff_index, cd->start, cd->start+cd->size);

    id<MTLBuffer> input_buffer = metal_transforms_kernel_input_buffer[buff_index];
    id<MTLBuffer> output_buffer = metal_transforms_kernel_output_buffer[buff_index];
    /* copy convert chunk input data into inputs buffer */
    const double *angs_start = params->angs + 3*cd->start;
    size_t count = 3*cd->size;
    // NSLog(@"Copy-convert from %p into '%@' (%zu)",
    //       angs_start, input_buffer.label, count);
    vDSP_vdpsp(angs_start, 1, input_buffer.contents, 1, count);
    // dump_doubles("from", angs_start, count);
    // dump_floats("to", input_buffer.contents, count);
    id<MTLCommandBuffer> cb = [metal_transforms_command_queue commandBufferWithUnretainedReferences];
    id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];

    [encoder setComputePipelineState: metal_transforms_pso_angles_to_dvec];
    [encoder setBuffer: output_buffer offset: 0 atIndex: 0];
    [encoder setBuffer: input_buffer offset: 0 atIndex: 1];
    [encoder setBuffer: metal_transforms_kernel_uniforms_buffer offset: 0 atIndex: 2];

    NSUInteger threadgroup_size = metal_transforms_pso_angles_to_dvec.maxTotalThreadsPerThreadgroup;
    if (threadgroup_size > cd->size)
        threadgroup_size = cd->size;

    [encoder dispatchThreads: MTLSizeMake(cd->size, 1, 1)
       threadsPerThreadgroup: MTLSizeMake(threadgroup_size, 1, 1)];
    [encoder endEncoding];
    [cb commit];
    // NSLog(@"launched command buffer:\n %@\n with Buffers: %@\n %@\n %@\n",
    //       cb, input_buffer, output_buffer, metal_transforms_kernel_uniforms_buffer);
    return cb;
}

static inline void
sync_finalize_chunk(id<MTLCommandBuffer>cb, const angles_to_dvec_params *params,
                    const chunk_desc *cd, size_t buff_index)
{
    id<MTLBuffer> output_buffer = metal_transforms_kernel_output_buffer[buff_index];
    [cb waitUntilCompleted];
    /* copy convert results */
    double *dVec_c_start = params->dVec_c + 3*cd->start;
    size_t count = 3*cd->size;
    os_signpost_interval_begin(OS_LOG_DEFAULT, OS_SIGNPOST_ID_EXCLUSIVE, "vspdp");
    vDSP_vspdp(output_buffer.contents, 1, dVec_c_start, 1, count);
    os_signpost_interval_end(OS_LOG_DEFAULT, OS_SIGNPOST_ID_EXCLUSIVE, "vspdp");
}

static inline void
prepare_angles_to_dvec_uniforms(id<MTLBuffer> uniforms_buffer,
                                const angles_to_dvec_params *params)
{
    angles_to_dvec_uniforms *uniforms = uniforms_buffer.contents;
    size_t i;

    for (i=0; i<9; i++) {
        uniforms->rMat_c[i] = (float)params->rMat_c[transpose_idx[i]];
    }

    for (i=0; i<9; i++) {
        uniforms->rMat_e[i] = (float)params->rMat_e[transpose_idx[i]];
    }

    uniforms->sin_chi = (float)sin(params->chi);
    uniforms->cos_chi = (float)cos(params->chi);
}

int
metal_support_angles_to_dvec(const angles_to_dvec_params *params)
{
    // dump_angles_to_dvec_params(params);
    chunking_spec cs;
    size_t chunk_max = metal_buffer_max_size/(3*sizeof(float));
    init_chunking_spec(&cs, chunk_max, params->total_count);

    /* paranoid check */
    if (cs.chunk_size*cs.chunk_count < cs.total_size || cs.chunk_size > chunk_max) {
        NSLog(@"THIS IS BAD! - split in %zu chunks of %zu size (total: %zu chunk_max: %zu",
              cs.chunk_count, cs.chunk_size, cs.total_size, chunk_max);
        goto error;
    }
    /* copy data into uniforms buffer. This can be reused by different kernel
       calls */
    prepare_angles_to_dvec_uniforms(metal_transforms_kernel_uniforms_buffer, params);

    const size_t window_size = metal_command_buffers_inflight_count;
    chunk_desc cd[window_size];
    id<MTLCommandBuffer> cb[window_size];
    size_t iter_count = cs.chunk_count + window_size;
    for (size_t iter = 0; iter < iter_count; iter++) {
        size_t window_index = iter % window_size;

        // NSLog(@"iter: %zu window_index: %zu", iter, window_index);
        if (iter >= window_size)
        {
            sync_finalize_chunk(cb[window_index], params, &cd[window_index], window_index);
        }

        if (iter < cs.chunk_count) {
            init_chunk_from_chunking_spec_and_index(&cd[window_index], &cs, iter);
            cb[window_index] = launch_chunk(&cd[window_index], params, window_index);
        }
    }
    return 0;
 error:
    return -1;
}
