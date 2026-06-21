// Rotate only the final store coordinate (CCW 90-degree steps: 1=90, 2=180,
// 3=270); the scene stays logical so sampling and blending are unchanged.
uvec2 outputLogicalSize(uint rotation) {
    uvec2 size = imageSize(dst);
    return (rotation & 1u) != 0u ? uvec2(size.y, size.x) : size;
}

ivec2 rotateOutputCoord(uvec2 coord, uint rotation) {
    uvec2 size = imageSize(dst);
    switch (rotation) {
        case 1u: return ivec2(coord.y, size.y - 1u - coord.x);
        case 2u: return ivec2(size.x - 1u - coord.x, size.y - 1u - coord.y);
        case 3u: return ivec2(size.x - 1u - coord.y, coord.x);
        default: return ivec2(coord);
    }
}

#ifdef VKR_FRAGMENT
layout(constant_id = 16) const bool c_linear_source = false;
layout(binding = 7, std430) readonly buffer LinearPixels { uint u_linearPixels[]; };
layout(push_constant) uniform LinearSource { uint u_linearStride, u_linearOffset, u_linearWidth, u_linearHeight, u_linearFlags; };
#endif
#include "colorimetry.h"

#include "shaderfilter.h"
#include "alphamode.h"

bool layer_full_screen(uint layerIdx) {
    return c_samplingMask != 0xffffffffu && (c_samplingMask & (1u << layerIdx)) != 0;
}

bool layer_aligned(uint layerIdx) {
    return c_samplingMask != 0xffffffffu && (c_samplingMask & (1u << (layerIdx + VKR_MAX_LAYERS))) != 0;
}

bool layer_black_border(uint layerIdx) {
    uint borders = c_samplingMask == 0xffffffffu ? u_borderMask : c_samplingMask >> (2u * VKR_MAX_LAYERS);
    return (borders & (1u << layerIdx)) != 0;
}

vec4 sampleRegular(sampler2D tex, vec2 coord, uint colorspace) {
    vec4 color = textureLod(tex, coord, 0);
    color.rgb = colorspace_plane_degamma_tf(color.rgb, colorspace);
    return color;
}

// To be considered pseudo-bandlimited, upscaling factor must be at least 2x.

const float bandlimited_PI = 3.14159265359;
const float bandlimited_PI_half = 0.5 * bandlimited_PI;
// size: resolution of sampled texture
// inv_size: inverse resolution of sampled texture
// extent: Screen-space gradient of UV in texels. Typically computed as (texture resolution) / (viewport resolution).
//   If screen is rotated by 90 or 270 degrees, the derivatives need to be computed appropriately.
//   For uniform scaling, none of this matters.
//   extent can be multiplied to achieve LOD bias.
//   extent must be at least 1.0 / 256.0.
vec4 sampleBandLimited(sampler2D samp, vec2 uv, vec2 size, vec2 inv_size, vec2 extent, uint colorspace, bool unnormalized)
{
    // Josh:
    // Clamp to behaviour like 4x scale (0.25).
    //
    // Was defaulted to 2x before (0.5), which is 1px, but gives blurry result
    // on Cave Story (480p) -> 800p on Deck.
    // TODO: Maybe make this configurable?
    const float max_extent = 0.25f;

	// Get base pixel and phase, range [0, 1).
	vec2 pixel = uv * (unnormalized ? vec2(1.0f) : size) - 0.5;
	vec2 base_pixel = floor(pixel);
	vec2 phase = pixel - base_pixel;

	// We can resolve the filter by just sampling a single 2x2 block.
	// Lerp between normal sampling at LOD 0, and bandlimited pixel filter at LOD -1.
	vec2 shift = 0.5 + 0.5 * sin(bandlimited_PI_half * clamp((phase - 0.5) / min(extent, vec2(max_extent)), -1.0, 1.0));
	uv = (base_pixel + 0.5 + shift) * (unnormalized ? vec2(1.0f) : inv_size);

	return sampleRegular(samp, uv, colorspace);
}

uint pseudo_random(uint seed) {
    seed ^= (seed << 13);
    seed ^= (seed >> 17);
    seed ^= (seed << 5);
    return seed * 1664525u + 1013904223u;
}

void compositing_debug(uvec2 coord, uint rotation) {
    uvec2 pos = coord;
    pos.x -= (u_frameId & 2) != 0 ?  128 : 0;
    pos.y -= (u_frameId & 1) != 0 ?  128 : 0;

    if (pos.x >= 40 && pos.x < 120 && pos.y >= 40 && pos.y < 120) {
        vec4 value = vec4(1.0f, 1.0f, 1.0f, 1.0f);
        if (checkDebugFlag(compositedebug_Markers_Partial)) {
            value = vec4(0.0f, 1.0f, 1.0f, 1.0f);
        }
        if (pos.x >= 48 && pos.x < 112 && pos.y >= 48 && pos.y < 112) {
            uint random = pseudo_random(u_frameId.x + (pos.x & ~0x7) + (pos.y & ~0x7) * 50);
            vec4 time = round(unpackUnorm4x8(random)).xyzw;
            if (time.x + time.y + time.z + time.w < 2.0f)
                value = vec4(0.0f, 0.0f, 0.0f, 1.0f);
        }
        imageStore(dst, rotateOutputCoord(coord, rotation), value);
    }
}

// Takes in a scRGB/Linear encoded value and applies color management
// based on the input colorspace.
//
// ie. call colorspace_plane_degamma_tf(color.rgb, colorspace) before
// input to this function.
vec3 apply_layer_color_mgmt(vec3 color, uint layer, uint colorspace, bool encodedInput) {
    if (colorspace == colorspace_passthru || colorspace == colorspace_display_native)
        return color;

    if (c_itm_enable)
    {
        color = bt2446a_inverse_tonemapping(color, u_itmSdrNits, u_itmTargetNits);
        colorspace = colorspace_pq;
    }

    // Shaper + 3D LUT path to match DRM.
    uint plane_eotf = colorspace_to_eotf(colorspace);

    if (layer == 0 && checkDebugFlag(compositedebug_Heatmap))
    {
        // Debug HDR heatmap.
        color = hdr_heatmap(color, colorspace);
        plane_eotf = EOTF_Gamma22;
    }


    // The shaper TF is basically just a regamma to get into something the shaper LUT can handle.
    //
    // Despite naming, degamma + shaper TF are NOT necessarily the inverse of each other. ^^^
    // This gets the color ready to go into the shaper LUT.
    // ie. scRGB -> PQ
    //
    // We also need to do degamma here for non-linear views to blend in linear space.
    // ie. PQ -> PQ would need us to manually do bilinear here.
    bool lut3d_enabled = c_colorMgmtMask == 0xffffffffu
        ? textureQueryLevels(s_shaperLut[plane_eotf]) != 0
        : (c_colorMgmtMask & (1u << (VKR_MAX_LAYERS + plane_eotf))) != 0;
    if (lut3d_enabled)
    {
        // Compact curves already contain the original LUT and its blend TF at
        // every 8-bit code. Other sampling geometries keep the original 3D LUT.
        if (encodedInput && c_colorMgmtMask != 0xffffffffu &&
            (c_colorMgmtMask & (1u << (2 * VKR_MAX_LAYERS + 3 * EOTF_Count + plane_eotf))) != 0) {
            // The nearest, unnormalized sampler selects exact 8-bit entries.
            vec3 coord = color * 255.0 + vec3(0.5, 256.5, 512.5);
            return vec3(textureLod(s_shaperLut[plane_eotf], coord.r, 0.0).r,
                        textureLod(s_shaperLut[plane_eotf], coord.g, 0.0).r,
                        textureLod(s_shaperLut[plane_eotf], coord.b, 0.0).r);
        }
        if (!encodedInput)
            color = colorspace_plane_shaper_tf(color, colorspace);
        if (c_colorMgmtMask == 0xffffffffu || (c_colorMgmtMask & (1u << (VKR_MAX_LAYERS + EOTF_Count + plane_eotf))) == 0)
            color = perform_1dlut(color, s_shaperLut[plane_eotf]);
        else
            color = clamp(color, vec3(0.0), vec3(1.0));
        color = perform_3dlut(color, s_lut3D[plane_eotf]);
        color = colorspace_blend_tf(color, c_output_eotf);
    }

    return color;
}

vec3 apply_layer_color_mgmt(vec3 color, uint layer, uint colorspace) {
    return apply_layer_color_mgmt(color, layer, colorspace, false);
}

// EASU already applied the layer CTM and encoded scRGB as PQ for RCAS.
vec3 apply_layer_color_mgmt_pq(vec3 pq, uint layer, uint colorspace) {
    bool lutEnabled = c_colorMgmtMask == 0xffffffffu
        ? textureQueryLevels(s_shaperLut[EOTF_PQ]) != 0
        : (c_colorMgmtMask & (1u << (VKR_MAX_LAYERS + EOTF_PQ))) != 0;
    if (lutEnabled && !c_itm_enable &&
        !(layer == 0 && checkDebugFlag(compositedebug_Heatmap)))
        return apply_layer_color_mgmt(pq, layer, colorspace, true);

    return apply_layer_color_mgmt(pqToScRGBEncoding(pq), layer, colorspace);
}

vec4 sampleBilinear(sampler2D tex, vec2 coord, uint colorspace, bool unnormalized) {
    vec2 scale = unnormalized ? vec2(1.0) : vec2(textureSize(tex, 0));

    vec2 pixCoord = coord * scale - 0.5f;
    vec2 originPixCoord = floor(pixCoord);

    vec2 gatherUV = (originPixCoord + 1.0f) / scale;

    vec4 red   = textureGather(tex, gatherUV, 0);
    vec4 green = textureGather(tex, gatherUV, 1);
    vec4 blue  = textureGather(tex, gatherUV, 2);
    vec4 alpha = textureGather(tex, gatherUV, 3);

    vec4 c00 = vec4(red.w, green.w, blue.w, alpha.w);
    vec4 c01 = vec4(red.x, green.x, blue.x, alpha.x);
    vec4 c11 = vec4(red.y, green.y, blue.y, alpha.y);
    vec4 c10 = vec4(red.z, green.z, blue.z, alpha.z);

    c00.rgb = colorspace_plane_degamma_tf(c00.rgb, colorspace);
    c01.rgb = colorspace_plane_degamma_tf(c01.rgb, colorspace);
    c11.rgb = colorspace_plane_degamma_tf(c11.rgb, colorspace);
    c10.rgb = colorspace_plane_degamma_tf(c10.rgb, colorspace);

    vec2 filterWeight = pixCoord - originPixCoord;

    vec4 temp0 = mix(c01, c11, filterWeight.x);
    vec4 temp1 = mix(c00, c10, filterWeight.x);
    return mix(temp1, temp0, filterWeight.y);
}

vec4 sampleLayerEx(sampler2D layerSampler, uint offsetLayerIdx, uint colorspaceLayerIdx, vec2 uv, bool unnormalized) {
    vec2 coord = layer_full_screen(offsetLayerIdx) ? uv
        : ((uv + u_offset[offsetLayerIdx]) * u_scale[offsetLayerIdx]);

#ifdef VKR_FRAGMENT
    vec2 texSize = c_linear_source ? vec2(u_linearWidth, u_linearHeight) : textureSize(layerSampler, 0);
#else
    vec2 texSize = textureSize(layerSampler, 0);
#endif

    if (!layer_full_screen(offsetLayerIdx) &&
        (coord.x < 0.0f       || coord.y < 0.0f ||
         coord.x >= texSize.x || coord.y >= texSize.y)) {
        float border = (u_borderMask & (1u << offsetLayerIdx)) != 0 ? 1.0f : 0.0f;

        if (checkDebugFlag(compositedebug_PlaneBorders))
            return vec4(vec3(1.0f, 0.0f, 1.0f) * border, border);

        return vec4(0.0f, 0.0f, 0.0f, border);
    }

    uint colorspace = get_layer_colorspace(colorspaceLayerIdx);
    vec4 color;
    if (layer_aligned(offsetLayerIdx) && unnormalized) {

#ifdef VKR_FRAGMENT
        if (c_linear_source) {
            color = unpackUnorm4x8(u_linearPixels[u_linearOffset + uint(coord.y) * u_linearStride + uint(coord.x)]);
            if ((u_linearFlags & 1u) != 0u) color = color.bgra;
            if ((u_linearFlags & 2u) != 0u) color.a = 1.0;
            if (colorspace == colorspace_linear &&
                (c_colorMgmtMask & (1u << (VKR_MAX_LAYERS + 3 * EOTF_Count + colorspaceLayerIdx))) == 0u)
                color.rgb = colorspace_plane_degamma_tf(color.rgb, colorspace_sRGB);
        } else
#endif
        color = texelFetch(layerSampler, ivec2(coord), 0);
        color.rgb = colorspace_plane_degamma_tf(color.rgb, colorspace);
    } else {
        if (!unnormalized)
            coord /= texSize;
        if (get_layer_shaderfilter(offsetLayerIdx) == filter_pixel) {
            vec2 output_res = texSize / u_scale[offsetLayerIdx];
            vec2 extent = max((texSize / output_res), vec2(1.0 / 256.0));
            color = sampleBandLimited(layerSampler, coord, unnormalized ? vec2(1.0f) : texSize, unnormalized ? vec2(1.0f) : vec2(1.0f) / texSize, extent, colorspace, unnormalized);
        }
        else if (get_layer_shaderfilter(offsetLayerIdx) == filter_linear_emulated) {
            color = sampleBilinear(layerSampler, coord, colorspace, unnormalized);
        }
        else {
            color = sampleRegular(layerSampler, coord, colorspace);
        }
    }
#ifdef VKR_FRAGMENT
    // Fully transparent premultiplied pixels do not touch the tile's color or
    // require LUT work, provided the color transform preserves transparent black.
    if (c_drawLayer > 0 && c_colorMgmtMask != 0xffffffffu &&
        (c_colorMgmtMask & (1u << colorspaceLayerIdx)) == 0 &&
        (c_colorMgmtMask & (1u << (VKR_MAX_LAYERS + 2 * EOTF_Count + colorspace_to_eotf(colorspace)))) != 0 &&
        all(equal(color, vec4(0.0))))
        discard;
#endif
    // JoshA: AMDGPU applies 3x4 CTM like this, where A is 1.0, but it only affects .rgb.
    if ((c_colorMgmtMask & (1u << colorspaceLayerIdx)) != 0)
        color.rgb = vec4(color.rgb, 1.0f) * u_ctm[colorspaceLayerIdx];
    bool encodedInput = c_colorMgmtMask != 0xffffffffu &&
        (c_colorMgmtMask & (1u << (VKR_MAX_LAYERS + 3 * EOTF_Count + colorspaceLayerIdx))) != 0;
    color.rgb = apply_layer_color_mgmt(color.rgb, offsetLayerIdx, colorspace, encodedInput);

    return color;
}

vec4 sampleLayer(sampler2D layerSampler, uint layerIdx, vec2 uv, bool unnormalized) {
    return sampleLayerEx(layerSampler, layerIdx, layerIdx, uv, unnormalized);
}

vec3 encodeOutputColor(vec3 value) {
    return colorspace_output_tf(value, c_output_eotf);
}
