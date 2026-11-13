#include "descriptor_set.h"

layout(
  local_size_x = 8,
  local_size_y = 8,
  local_size_x_id = 8,
  local_size_y_id = 9,
  local_size_z = 1) in;

#include "blit_push_data.h"
#include "composite.h"

vec4 sampleLayer(uint layerIdx, vec2 uv) {
    if ((c_ycbcrMask & (1 << layerIdx)) != 0)
        return sampleLayer(s_ycbcr_samplers[layerIdx], layerIdx, uv, false);
    return sampleLayer(s_samplers[layerIdx], layerIdx, uv, true);
}

vec4 blendLayer(uint layerIdx, vec2 uv, vec4 background) {
    bool fullScreen = layer_full_screen(layerIdx);
    bool blackBorder = layer_black_border(layerIdx);
    // Transparent borders leave the background unchanged. Cull the whole group
    // before sampling, color conversion, and blending, including on the full path.
    if (!fullScreen && !blackBorder &&
        (c_simple_blend || get_layer_alphamode(layerIdx) != alpha_mode_none)) {
        vec2 first = vec2(gl_WorkGroupID.xy * gl_WorkGroupSize.xy) + u_offset[layerIdx];
        vec2 last = first + vec2(gl_WorkGroupSize.xy - uvec2(1));
        bool aligned = layer_aligned(layerIdx);
        if (!aligned) {
            first *= u_scale[layerIdx];
            last *= u_scale[layerIdx];
        }
        vec2 size = (c_ycbcrMask & (1u << layerIdx)) != 0
            ? textureSize(s_ycbcr_samplers[layerIdx], 0)
            : textureSize(s_samplers[layerIdx], 0);
        if (any(lessThan(max(first, last), vec2(0.0))) ||
            any(greaterThanEqual(min(first, last), size)))
            return background;
    }
    vec4 foreground = sampleLayer(layerIdx, uv);
    if (c_simple_blend)
        return foreground + background * (1.0 - foreground.a);
    return BlendLayer(layerIdx, background, foreground, u_opacity[layerIdx]);
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
        outputValue = sampleLayer(0, uv) * (c_simple_blend ? 1.0 : u_opacity[0]);
    }

    // Keep sampler indices constant on drivers without dynamic image-array indexing.
#define BLEND_LAYER(i) if (c_layerCount > i) outputValue = blendLayer(i, uv, outputValue);
    BLEND_LAYER(1)
    BLEND_LAYER(2)
    BLEND_LAYER(3)
    BLEND_LAYER(4)
    BLEND_LAYER(5)
#undef BLEND_LAYER

    outputValue.rgb = encodeOutputColor(outputValue.rgb);
    // XRGB targets discard alpha, so intermediate alpha calculations are dead.
    if (c_opaque_output)
        outputValue.a = 1.0;
    imageStore(dst, ivec2(coord), outputValue);

    // Indicator to quickly tell if we're in the compositing path or not.
    if (checkDebugFlag(compositedebug_Markers))
        compositing_debug(coord);
}
