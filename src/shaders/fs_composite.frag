#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_scalar_block_layout : require

#define VKR_FRAGMENT 1
#include "descriptor_set.h"
#include "blit_push_data.h"
#include "composite.h"

layout(location = 0) out vec4 color;

void main()
{
    vec2 uv = gl_FragCoord.xy - vec2(0.5);
    if ((c_ycbcrMask & (1u << c_drawLayer)) != 0)
        color = sampleLayer(s_ycbcr_samplers[c_drawLayer], c_drawLayer, uv, false);
    else
        color = sampleLayer(s_samplers[c_drawLayer], c_drawLayer, uv, true);
    color *= u_opacity[c_drawLayer];
    // The sRGB color attachment performs the output transfer after linear blending.
}
