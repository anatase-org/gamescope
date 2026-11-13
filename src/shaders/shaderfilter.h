#ifndef SHADERFILTER_H
#define SHADERFILTER_H

const int shader_filter_max_bits = 4;

uint get_layer_shaderfilter(uint layerIdx) {
    uint filters = c_shaderFilterMask == 0xffffffffu ? u_shaderFilter : c_shaderFilterMask;
    return bitfieldExtract(filters, int(layerIdx) * shader_filter_max_bits, shader_filter_max_bits);
}

#endif
