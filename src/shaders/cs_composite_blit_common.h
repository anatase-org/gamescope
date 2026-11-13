#include "descriptor_set.h"

layout(
  local_size_x = 8,
  local_size_y = 8,
  local_size_z = 1) in;

#include "blit_push_data.h"
#include "composite.h"

vec4 sampleLayer(uint layerIdx, vec2 uv) {
    if ((c_ycbcrMask & (1 << layerIdx)) != 0)
        return sampleLayer(s_ycbcr_samplers[layerIdx], layerIdx, uv, false);
    return sampleLayer(s_samplers[layerIdx], layerIdx, uv, true);
}

void main() {
    uvec2 coord = uvec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y);
    uvec2 outSize = imageSize(dst);

    if (coord.x >= outSize.x || coord.y >= outSize.y)
        return;

    vec2 uv = vec2(coord);
    vec4 outputValue = vec4(0.0f);

    if (checkDebugFlag(compositedebug_PlaneBorders))
        outputValue = vec4(1.0f, 0.0f, 0.0f, 0.0f);

    if (c_layerCount > 0) {
        outputValue = sampleLayer(0, uv) * u_opacity[0];
    }

    // Keep sampler indices constant on drivers without dynamic array indexing.
#define BLEND_LAYER(i) if (c_layerCount > i) outputValue = BlendLayer(i, outputValue, sampleLayer(i, uv), u_opacity[i]);
    BLEND_LAYER(1)
    BLEND_LAYER(2)
    BLEND_LAYER(3)
    BLEND_LAYER(4)
    BLEND_LAYER(5)
#undef BLEND_LAYER

    outputValue.rgb = encodeOutputColor(outputValue.rgb);
    imageStore(dst, ivec2(coord), outputValue);

    // Indicator to quickly tell if we're in the compositing path or not.
    if (checkDebugFlag(compositedebug_Markers))
        compositing_debug(coord);
}
