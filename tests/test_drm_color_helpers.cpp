#include "drm_color_helpers.h"
#include "color_helpers_impl.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#define CHECK(condition) do { if (!(condition)) { std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition); std::abort(); } } while (false)

using namespace drm_color;

namespace
{
template<typename T> T Read(const std::vector<uint8_t> &bytes, size_t index = 0)
{
    T value;
    CHECK(bytes.size() >= (index + 1) * sizeof(T));
    std::memcpy(&value, bytes.data() + index * sizeof(T), sizeof(T));
    return value;
}
Operation Op(uint32_t id, std::string type, uint32_t next, bool bypass = true)
{
    Operation op{id, std::move(type), {}};
    op.properties["NEXT"] = {id * 10, next, true, {}};
    op.properties["DATA"] = {id * 10 + 1, 0, false, {}};
    op.properties["SIZE"] = {id * 10 + 2, 17, true, {}};
    op.properties["LUT1D_INTERPOLATION"] = {id * 10 + 3, 5, true, {{"Linear", 5}}};
    op.properties["LUT3D_INTERPOLATION"] = {id * 10 + 4, 7, true, {{"Tetrahedral", 7}}};
    if (bypass) op.properties["BYPASS"] = {id * 10 + 5, 1, false, {}};
    return op;
}
std::vector<uint16_t> IdentityTable(uint32_t size)
{
    std::vector<uint16_t> result(size * 4);
    for (uint32_t i = 0; i < size; i++)
        for (int c = 0; c < 3; c++) result[4 * i + c] = quantize_lut_value_16bit(float(i) / (size - 1));
    return result;
}
std::vector<uint16_t> IdentityCube(uint32_t edge)
{
    std::vector<uint16_t> result(edge * edge * edge * 4);
    for (uint32_t b = 0; b < edge; b++)
        for (uint32_t g = 0; g < edge; g++)
            for (uint32_t r = 0; r < edge; r++)
            {
                size_t index = 4 * (r + edge * (g + edge * b));
                result[index] = quantize_lut_value_16bit(float(r) / (edge - 1));
                result[index + 1] = quantize_lut_value_16bit(float(g) / (edge - 1));
                result[index + 2] = quantize_lut_value_16bit(float(b) / (edge - 1));
            }
    return result;
}
void Near(glm::vec3 actual, glm::vec3 expected, float epsilon = 0.0001f)
{
    if (!glm::all(glm::lessThanEqual(glm::abs(actual - expected), glm::vec3(epsilon))))
        std::fprintf(stderr, "actual=(%g,%g,%g), expected=(%g,%g,%g), tolerance=%g\n",
            actual.r, actual.g, actual.b, expected.r, expected.g, expected.b, epsilon);
    CHECK(glm::all(glm::lessThanEqual(glm::abs(actual - expected), glm::vec3(epsilon))));
}
glm::vec3 EvaluateStages(std::span<const Stage> stages, glm::vec3 input)
{
    for (const auto &stage : stages)
    {
        if (stage.type == Stage::Type::Curve) input = stage.function(input);
        else if (stage.type == Stage::Type::Lut3D) input = Sample3D(stage.cube, stage.edge, input);
        else if (stage.type == Stage::Type::Matrix)
        {
            glm::vec3 value;
            for (int r = 0; r < 3; r++)
                value[r] = stage.matrix[r * 4] * input.r + stage.matrix[r * 4 + 1] * input.g +
                    stage.matrix[r * 4 + 2] * input.b + stage.matrix[r * 4 + 3];
            input = value;
        }
        else CHECK(false);
    }
    return input;
}
template<typename Entry>
glm::vec3 SampleWireCurve(const std::vector<uint8_t> &blob, glm::vec3 input, bool shared)
{
    const size_t size = blob.size() / sizeof(Entry);
    CHECK(size >= 2);
    glm::vec3 result;
    for (int c = 0; c < 3; c++)
    {
        const float position = clamp01(input[c]) * float(size - 1);
        const size_t lo = std::min(size_t(position), size - 2);
        const Entry a = Read<Entry>(blob, lo), b = Read<Entry>(blob, lo + 1);
        const double channelsA[] = {double(a.red), double(a.green), double(a.blue)};
        const double channelsB[] = {double(b.red), double(b.green), double(b.blue)};
        const int channel = shared ? 1 : c;
        const double scale = sizeof(Entry) == sizeof(Lut32) ? double(UINT32_MAX) : double(UINT16_MAX);
        result[c] = std::lerp(channelsA[channel], channelsB[channel], position - lo) / scale;
    }
    return result;
}
glm::vec3 EvaluateIntelPlane(const Pipeline &pipeline, const Program &program, glm::vec3 input,
    bool quantizeMatrix = false)
{
    for (const auto &op : pipeline.operations)
    {
        const auto bypass = op.properties.at("BYPASS").id;
        auto state = std::find_if(program.begin(), program.end(), [&](const auto &w) { return w.object == op.id && w.property == bypass; });
        CHECK(state != program.end());
        if (state->value) continue;
        auto data = std::find_if(program.begin(), program.end(), [&](const auto &w) { return w.object == op.id && !w.blob.empty(); });
        CHECK(data != program.end());
        if (op.type == "1D LUT")
            input = SampleWireCurve<Lut32>(data->blob, input, true); // kernel programs .green only
        else
        {
            CHECK(op.type == "3x4 Matrix");
            Matrix matrix;
            for (int i = 0; i < 12; i++)
            {
                const uint64_t value = Read<uint64_t>(data->blob, i);
                matrix[i] = double(value & ~(1ULL << 63)) / 4294967296.0 * (value >> 63 ? -1 : 1);
                if (quantizeMatrix && i % 4 != 3)
                {
                    // patchworkan's xelpd_load_plane_csc_matrix uses the
                    // ILK_CSC_COEFF_FP sliding precision window (9 bits).
                    const uint64_t magnitude = value & ~(1ULL << 63);
                    const double absolute = std::abs(matrix[i]);
                    const int bits = absolute < .125 ? 12 : absolute < .25 ? 11 : absolute < .5 ? 10 :
                        absolute < 1 ? 9 : absolute < 2 ? 8 : 7;
                    const uint64_t rounded = std::min((magnitude >> (32 - bits - 3)) + 4, uint64_t(0xfff)) & 0xff8;
                    matrix[i] = double(rounded >> 3) / (1u << bits) * (value >> 63 ? -1 : 1);
                }
            }
            Stage affine = Affine(matrix);
            input = EvaluateStages(std::span(&affine, 1), input);
        }
    }
    return input;
}
void TestWireFormats()
{
    CHECK(*SignMagnitude(1.5) == 0x180000000ULL);
    CHECK(*SignMagnitude(-1.5) == 0x8000000180000000ULL);
    CHECK(*SignMagnitude(-0.0) == 0);
    CHECK(!SignMagnitude(2147483648.0));
    CHECK(!SignMagnitude(std::numeric_limits<double>::infinity()));
    CHECK(!SignMagnitude(std::numeric_limits<double>::quiet_NaN()));
    Matrix matrix{1, 2, 3, .5, 4, -5, 6, -.25, 7, 8, 9, 0};
    auto blob = MatrixBlob(matrix, true);
    CHECK(blob && blob->size() == 12 * sizeof(uint64_t));
    CHECK(Read<uint64_t>(*blob, 3) == 0x80000000ULL);
    CHECK(Read<uint64_t>(*blob, 5) == 0x8000000500000000ULL);
    CHECK(!MatrixBlob(matrix, false));
    matrix[3] = matrix[7] = 0;
    blob = MatrixBlob(matrix, false);
    CHECK(blob && blob->size() == 9 * sizeof(uint64_t));
    CHECK(Read<uint64_t>(*blob, 3) == 4ULL << 32);
    glm::mat3 m{1, 4, 7, 2, 5, 8, 3, 6, 9};
    CHECK(FromMatrix(m)[1] == 2);
    auto curve = SampleCurve(Transfer(Curve::SRGBDecode), 1024, true);
    CHECK(curve && curve->size() == 1024 * sizeof(Lut32));
    CHECK(Read<Lut32>(*curve).red == 0);
    CHECK(Read<Lut32>(*curve, 1023).blue == UINT32_MAX);
    CHECK(Read<Lut32>(*curve, 500).reserved == 0);
    auto narrow = SampleCurve(Transfer(Curve::Gamma22Decode), 256, false);
    CHECK(narrow && Read<Lut16>(*narrow, 255).red == UINT16_MAX);
    CHECK(!SampleCurve(Transfer(Curve::PQDecode), 256, true)); // 125 cannot fit in UNORM
    CHECK(!SampleCurve(Transfer(Curve::PQEncode), 256, true)); // unbounded scRGB input
    CHECK(!SampleCurve(Transfer(Curve::SRGBDecode), 1, true));
    CHECK(!SampleCurve(Transfer(Curve::PQNormalizedEncode), 1024, false));
    CHECK(!SampleCurve(Transfer(Curve::Gamma22Encode), 32, true));
}
void TestUnitsAndInterpolation()
{
    Near(Evaluate(Curve::PQDecode, glm::vec3(1)), glm::vec3(125), .01f);
    Near(Evaluate(Curve::PQEncode, glm::vec3(1)), nits_to_pq(glm::vec3(80)), .00001f);
    Near(Evaluate(Curve::PQNormalizedDecode, glm::vec3(1)), glm::vec3(1), .0001f);
    auto table = IdentityTable(4096);
    auto cube = IdentityCube(17);
    for (auto v : {glm::vec3(.1f, .6f, .8f), glm::vec3(.8f, .1f, .6f), glm::vec3(.6f, .8f, .1f),
                   glm::vec3(.1f, .8f, .6f), glm::vec3(.8f, .6f, .1f), glm::vec3(.6f, .1f, .8f), glm::vec3(1)})
    {
        Near(Sample1D(table, v), v);
        Near(Sample3D(cube, 17, v), v);
    }
    CHECK(Separable(cube, 17));
    cube[4 * (1 + 17 * (2 + 17 * 3))]++;
    CHECK(!Separable(cube, 17)); // even a small cross-channel dependency must survive
}
void TestPipelines()
{
    auto shaper = IdentityTable(4096);
    auto cube = IdentityCube(17);
    cube[4 * (1 + 17 * 2)] = 1000; // arbitrary, nonseparable look
    Pipeline amd{42, 100, {Op(100, "1D Curve", 101), Op(101, "Multiplier", 102),
        Op(102, "3x4 Matrix", 103), Op(103, "1D LUT", 104), Op(104, "3D LUT", 105), Op(105, "1D Curve", 0)}};
    Transform transform;
    transform.managed = true;
    transform.shaper = shaper;
    transform.cube = cube;
    auto variants = BuildVariants(transform);
    CHECK(variants.size() == 1);
    auto compiled = Compile(amd, variants[0]);
    CHECK(compiled);
    bool sawCube = false;
    for (const auto &write : *compiled)
        if (write.object == 104 && !write.blob.empty())
        {
            sawCube = true;
            CHECK(write.blob.size() == 17 * 17 * 17 * sizeof(Lut32));
            CHECK(Read<Lut32>(write.blob, 16).red == UINT32_MAX);
        }
    CHECK(sawCube);
    Pipeline intel{43, 200, {Op(200, "1D LUT", 201), Op(201, "3x4 Matrix", 202),
        Op(202, "3D LUT", 203), Op(203, "1D LUT", 0)}};
    CHECK(Compile(intel, variants[0]));
    Pipeline noCube{43, 200, {Op(200, "1D LUT", 201), Op(201, "3x4 Matrix", 203), Op(203, "1D LUT", 0)}};
    CHECK(!Compile(noCube, variants[0]));
    transform.cube = cube = IdentityCube(17);
    variants = BuildVariants(transform);
    CHECK(variants.size() == 2 && Compile(noCube, variants[1]));
    amd.operations[4].properties["SIZE"].value = 9;
    CHECK(!Compile(amd, variants[0]));
    Pipeline future{1, 10, {Op(10, "Future stage", 0)}};
    CHECK(Compile(future, {}));
    future.operations[0].properties.erase("BYPASS");
    CHECK(!Compile(future, {}));
    future.operations[0].properties["NEXT"].value = 10;
    CHECK(!ValidPipeline(future));
    Pipeline fixed{1, 30, {Op(30, "1D Curve", 0, false)}};
    fixed.operations[0].properties["CURVE_1D_TYPE"] = {301, 23, false, {{"PQ 125 Inverse EOTF", 23}}};
    std::vector<Stage> stages{Transfer(Curve::PQEncode)};
    CHECK(Compile(fixed, stages)); // enum value is deliberately not its index
    CHECK(!Compile(fixed, {})); // non-bypassable NVIDIA curve
    stages = {Transfer(Curve::SRGBDecode)};
    CHECK(!Compile(fixed, stages));
    Pipeline yuv{1, 40, {Op(40, "Fixed Matrix", 0)}};
    yuv.operations[0].properties["FIXED_MATRIX_TYPE"] = {401, 29, false, {{"YCbCr 709 Limited to RGB", 29}}};
    Stage convert;
    convert.type = Stage::Type::FixedMatrix;
    convert.fixedMatrix = "YCbCr 709 Limited to RGB";
    CHECK(Compile(yuv, std::span(&convert, 1)));
    convert.fixedMatrix = "YCbCr 601 Limited to RGB";
    CHECK(!Compile(yuv, std::span(&convert, 1)));
    Pipeline mult{1, 50, {Op(50, "Multiplier", 0)}};
    mult.operations[0].properties["MULTIPLIER"] = {501, 0, false, {}};
    auto scalar = Identity();
    scalar[0] = scalar[5] = scalar[10] = .008;
    auto gain = Affine(scalar);
    CHECK(Compile(mult, std::span(&gain, 1)));
    gain.matrix[3] = .1;
    CHECK(!Compile(mult, std::span(&gain, 1)));
}
void TestCRTCAndTransforms()
{
    auto shaper = IdentityTable(4096);
    auto cube = IdentityCube(17);
    Transform t;
    t.managed = true;
    t.shaper = shaper;
    t.cube = cube;
    auto variants = BuildVariants(t);
    CHECK(!CompileCRTC(variants[0], 1024, true, 1024)); // cannot drop the 3D stage
    auto crtc = CompileCRTC(variants[1], 0, false, 1024);
    CHECK(crtc && crtc->ctm.empty() && crtc->degamma.empty() && !crtc->gamma.empty());
    t.blend = true;
    variants = BuildVariants(t);
    CHECK(variants[1].size() == 1); // curves fuse before linear-light blending
    Near(variants[1][0].function(glm::vec3(.5f)), Evaluate(Curve::SRGBDecode, glm::vec3(.5f)));
    t.blend = false;
    t.ctm = Identity();
    (*t.ctm)[3] = .1;
    variants = BuildVariants(t);
    CHECK(!CompileCRTC(variants[1], 1024, true, 1024)); // legacy matrix cannot carry offsets
    t = {};
    CHECK(BuildVariants(t)[0].empty()); // composed image: identity, no second transform
    CHECK(CompileCRTC({}, 0, false, 0));
    // A dirty/unsupported program must not mutate the discoverable topology.
    Pipeline pipeline{1, 20, {Op(20, "1D LUT", 0)}};
    pipeline.operations[0].properties["SIZE"].value = 1024;
    std::vector<Stage> curves{Transfer(Curve::SRGBDecode)};
    auto first = Compile(pipeline, curves);
    curves = {Transfer(Curve::Gamma22Decode)};
    auto second = Compile(pipeline, curves);
    CHECK(first && second && first->back().blob != second->back().blob);
    CHECK(pipeline.operations[0].properties.at("BYPASS").value == 1);
    // scRGB keeps its PQ encode (including the 125.0 clamp) before any
    // analytic alternative. 1.0 scRGB is 80 nits, not 10,000 nits.
    t.managed = true;
    t.input = Encoding::ScRGB;
    t.output = EOTF_PQ;
    t.shaper = shaper;
    t.cube = cube;
    t.analytic = color_matrix_transform_t{glm::mat3(1), EOTF_PQ, EOTF_PQ, 1};
    variants = BuildVariants(t);
    Near(EvaluateStages(variants.back(), glm::vec3(1)), nits_to_pq(glm::vec3(80)), .0001f);
    Near(EvaluateStages(variants.back(), glm::vec3(250)), glm::vec3(1), .0001f);
    t.analytic->matrix *= .5f;
    variants = BuildVariants(t);
    Near(EvaluateStages(variants.back(), glm::vec3(250)), nits_to_pq(glm::vec3(5000)), .0001f);
}
void TestAnalyticReference()
{
    for (auto source : {EOTF_Gamma22, EOTF_PQ})
        for (auto dest : {EOTF_Gamma22, EOTF_PQ})
        {
            lut1d_t shaper;
            lut3d_t cube;
            tonemapping_t tone;
            tone.g22_luminance = 203.f;
            std::optional<color_matrix_transform_t> analytic;
            calcColorTransform<17>(&shaper, 4096, &cube, displaycolorimetry_709, source,
                displaycolorimetry_2020, dest, displaycolorimetry_709, glm::vec2(0),
                k_EChromaticAdapatationMethod_Bradford, {0, 1, 0, 0}, {.3f, .05f, .6f}, tone, nullptr, .8f, &analytic);
            if (source == EOTF_PQ && dest == EOTF_Gamma22)
            {
                CHECK(!analytic); // shaper clips HDR before the matrix/night mode
                continue;
            }
            CHECK(analytic);
            std::vector<uint16_t> quantizedShaper(shaper.dataR.size() * 4);
            std::vector<uint16_t> quantizedCube(cube.data.size() * 4);
            for (size_t i = 0; i < shaper.dataR.size(); i++)
            {
                quantizedShaper[4 * i] = quantize_lut_value_16bit(shaper.dataR[i]);
                quantizedShaper[4 * i + 1] = quantize_lut_value_16bit(shaper.dataG[i]);
                quantizedShaper[4 * i + 2] = quantize_lut_value_16bit(shaper.dataB[i]);
            }
            for (size_t i = 0; i < cube.data.size(); i++)
                for (int c = 0; c < 3; c++) quantizedCube[4 * i + c] = quantize_lut_value_16bit(cube.data[i][c]);
            Transform transform;
            transform.input = source == EOTF_PQ ? Encoding::PQ : Encoding::SDR;
            transform.output = dest;
            transform.managed = true;
            transform.shaper = quantizedShaper;
            transform.cube = quantizedCube;
            transform.analytic = analytic;
            auto variants = BuildVariants(transform);
            for (glm::vec3 encoded : {glm::vec3(.25f), glm::vec3(.5f), glm::vec3(.75f), glm::vec3(1, 0, 0)})
            {
                auto linear = source == EOTF_PQ ? pq_to_nits(encoded) : glm::pow(encoded, glm::vec3(2.2f)) * tone.g22_luminance;
                linear = analytic->matrix * linear;
                auto expected = dest == EOTF_PQ ? nits_to_pq(linear) : glm::pow(glm::clamp(linear / tone.g22_luminance, glm::vec3(0), glm::vec3(1)), glm::vec3(1.f / 2.2f));
                const auto shaped = Sample1D(quantizedShaper, encoded);
                Near(Sample3D(quantizedCube, 17, shaped), expected, .015f);
                Near(EvaluateStages(variants.back(), encoded), expected, .0002f);
                transform.blend = true;
                auto blended = BuildVariants(transform);
                Near(EvaluateStages(blended.back(), encoded), Evaluate(dest == EOTF_PQ ? Curve::PQNormalizedDecode : Curve::SRGBDecode, expected), .0002f);
            }
            // Saturation-dependent gamut mapping cannot be replaced by a matrix.
            calcColorTransform<17>(nullptr, 0, &cube, displaycolorimetry_709, source,
                displaycolorimetry_2020, dest, displaycolorimetry_2020, glm::vec2(0),
                k_EChromaticAdapatationMethod_Bradford, {0, 1, 0, 1}, {}, tone, nullptr, 1, &analytic);
            CHECK(!analytic);
        }
    lut3d_t cube;
    lut1d_t shaper;
    tonemapping_t tone;
    tone.eOperator = ETonemapOperator_EETF2390_Independent;
    tone.eetf2390.init({.1f, 1000.f}, {.1f, 203.f});
    std::optional<color_matrix_transform_t> analytic;
    // Tone mapping and custom looks must never be silently replaced by a matrix.
    calcColorTransform<17>(&shaper, 4096, &cube, displaycolorimetry_709, EOTF_Gamma22,
        displaycolorimetry_709, EOTF_Gamma22, displaycolorimetry_709, glm::vec2(0),
        k_EChromaticAdapatationMethod_Bradford, {0, 1, 0, 0}, {}, tone, nullptr, 1, &analytic);
    CHECK(!analytic);
    tone.eOperator = ETonemapOperator_None;
    lut3d_t look;
    look.resize(2);
    for (auto &value : look.data) value = glm::vec3(0);
    calcColorTransform<17>(&shaper, 4096, &cube, displaycolorimetry_709, EOTF_Gamma22,
        displaycolorimetry_709, EOTF_Gamma22, displaycolorimetry_709, glm::vec2(0),
        k_EChromaticAdapatationMethod_Bradford, {0, 1, 0, 0}, {}, tone, &look, 1, &analytic);
    CHECK(!analytic);
}

void TestWhitePoint()
{
    lut1d_t shaper;
    lut3d_t cube;
    tonemapping_t tone;
    std::optional<color_matrix_transform_t> analytic;
    calcColorTransform<17>(&shaper, 4096, &cube, displaycolorimetry_709, EOTF_Gamma22,
        displaycolorimetry_709, EOTF_Gamma22, displaycolorimetry_709, {.3457f, .3585f},
        k_EChromaticAdapatationMethod_Bradford, {0, 1, 0, 0}, {}, tone, nullptr, 1, &analytic);
    CHECK(analytic && analytic->matrix != glm::mat3(1));
    for (auto coordinate : {glm::ivec3(4, 8, 12), glm::ivec3(16), glm::ivec3(16, 0, 0)})
    {
        auto encoded = glm::vec3(coordinate) / 16.f;
        auto linear = analytic->matrix * glm::pow(encoded, glm::vec3(2.2f));
        auto expected = glm::pow(glm::clamp(linear, glm::vec3(0), glm::vec3(1)), glm::vec3(1.f / 2.2f));
        Near(cube.data[coordinate.r + 17 * (coordinate.g + 17 * coordinate.b)], expected);
    }
}

void TestIntelSingleChannelNightMode()
{
    auto shaper = IdentityTable(4096);
    auto cube = IdentityCube(17);
    // Native SDR night mode is a separable tint. Intel's plane LUT loaders
    // read only green; accepting this as a shared curve would dim all three
    // channels by 0.5 and discard the intended temperature change.
    for (size_t i = 0; i < cube.size(); i += 4)
    {
        cube[i + 1] = quantize_lut_value_16bit(cube[i + 1] / 65535.f * .5f);
        cube[i + 2] = quantize_lut_value_16bit(cube[i + 2] / 65535.f * .25f);
    }
    Transform transform;
    transform.managed = true;
    transform.shaper = shaper;
    transform.cube = cube;
    auto variants = BuildVariants(transform);
    CHECK(variants.size() >= 2);
    Pipeline intel{42, 100, {Op(100, "1D LUT", 101), Op(101, "3x4 Matrix", 102), Op(102, "1D LUT", 0)}};
    intel.operations[0].singleChannel = true;
    intel.operations[0].properties["SIZE"].value = 128;
    intel.operations[2].singleChannel = true;
    intel.operations[2].properties["SIZE"].value = 32;
    CHECK(!Compile(intel, variants[1]));
    Pipeline postOnly{42, 102, {intel.operations[2]}};
    CHECK(!Compile(postOnly, variants[1]));
    auto matrixVariant = std::find_if(variants.begin(), variants.end(), [](const auto &stages) {
        return stages.size() == 1 && stages[0].type == Stage::Type::Matrix;
    });
    CHECK(matrixVariant != variants.end());
    auto hardwareNight = Compile(intel, *matrixVariant);
    CHECK(hardwareNight);
    bool sawMatrix = false;
    for (const auto &write : *hardwareNight)
    {
        if (write.object != 101) CHECK(write.blob.empty());
        if (!write.blob.empty())
        {
            sawMatrix = true;
            CHECK(Read<uint64_t>(write.blob, 0) == *SignMagnitude(1));
            CHECK(std::abs(double(Read<uint64_t>(write.blob, 5)) / 4294967296.0 - .5) < 2.0 / 65535);
            CHECK(std::abs(double(Read<uint64_t>(write.blob, 10)) / 4294967296.0 - .25) < 2.0 / 65535);
        }
    }
    CHECK(sawMatrix);
    auto rgbPlane = intel;
    rgbPlane.operations[0].singleChannel = false;
    CHECK(Compile(rgbPlane, variants[1]));

    // Shared transfer functions remain eligible for plane offload.
    std::vector<Stage> uniform{Transfer(Curve::Gamma22Decode)};
    CHECK(Compile(intel, uniform));

    // The CRTC LUT has independent RGB entries and preserves the warm tint.
    auto crtc = CompileCRTC(variants[1], 0, false, 1024);
    CHECK(crtc && crtc->degamma.empty() && crtc->ctm.empty());
    auto white = Read<Lut16>(crtc->gamma, 1023);
    CHECK(white.red == UINT16_MAX);
    CHECK(std::abs(int(white.green) - 32768) <= 1);
    CHECK(std::abs(int(white.blue) - 16384) <= 1);

    // On hardware with a 3D LUT, retain the full RGB transform there; only
    // the uniform shaper is placed in the single-channel plane LUT.
    intel.operations[1].properties["NEXT"].value = 103;
    intel.operations.insert(intel.operations.begin() + 2, Op(103, "3D LUT", 102));
    CHECK(Compile(intel, variants[0]));
}

void TestIntelNativeSDRNightScanout()
{
    // Exact topology and sizes from patchworkan/intel_color_pipeline.c.
    Pipeline intel{42, 100, {Op(100, "1D LUT", 101), Op(101, "3x4 Matrix", 102), Op(102, "1D LUT", 0)}};
    intel.operations[0].singleChannel = intel.operations[2].singleChannel = true;
    intel.operations[0].properties["SIZE"].value = 128;
    intel.operations[2].properties["SIZE"].value = 32;
    for (float amount : {.001f, .01f, .1f, .5f, .8f, 1.f})
    {
        lut1d_t shape;
        lut3d_t cube;
        tonemapping_t tone;
        calcColorTransform<17>(&shape, 4096, &cube, displaycolorimetry_709, EOTF_Gamma22,
            displaycolorimetry_709, EOTF_Gamma22, displaycolorimetry_709, glm::vec2(0),
            k_EChromaticAdapatationMethod_Bradford, {0, 1, 0, 0}, {amount, .05f, 1.f}, tone, nullptr, 1);
        std::vector<uint16_t> shaper(shape.dataR.size() * 4), table(cube.data.size() * 4);
        for (size_t i = 0; i < shape.dataR.size(); i++)
        {
            shaper[4 * i] = quantize_lut_value_16bit(shape.dataR[i]);
            shaper[4 * i + 1] = quantize_lut_value_16bit(shape.dataG[i]);
            shaper[4 * i + 2] = quantize_lut_value_16bit(shape.dataB[i]);
        }
        for (size_t i = 0; i < cube.data.size(); i++)
            for (int c = 0; c < 3; c++) table[4 * i + c] = quantize_lut_value_16bit(cube.data[i][c]);
        Transform transform;
        transform.managed = true;
        transform.shaper = shaper;
        transform.cube = table;
        auto variants = BuildVariants(transform);
        auto matrix = std::find_if(variants.begin(), variants.end(), [](const auto &s) {
            return s.size() == 1 && s[0].type == Stage::Type::Matrix;
        });
        CHECK(matrix != variants.end());
        auto program = Compile(intel, *matrix);
        CHECK(program); // hardware plane matrix; no CRTC or 3D LUT required
        for (int i = 0; i < 256; i++)
        {
            const glm::vec3 input(float(i) / 255.f, float(255 - i) / 255.f, float((i * 23) % 256) / 255.f);
            const auto expected = Sample3D(table, 17, Sample1D(shaper, input));
            Near(EvaluateIntelPlane(intel, *program, input), expected, 2.1f / 65535);
            Near(EvaluateIntelPlane(intel, *program, input, true), expected, 2.f / 1023);
        }
        const auto white = EvaluateIntelPlane(intel, *program, glm::vec3(1));
        CHECK(white.r > white.g && white.g > white.b);

        // Alpha-capable buffers/overlays retain linear-light blending. The
        // tint goes in the matrix; only the shared sRGB decode uses the 32 LUT.
        transform.blend = true;
        auto blended = BuildVariants(transform);
        auto blendMatrix = std::find_if(blended.begin(), blended.end(), [](const auto &s) {
            return s.size() == 2 && s[0].type == Stage::Type::Matrix && s[1].type == Stage::Type::Curve;
        });
        CHECK(blendMatrix != blended.end());
        auto blendProgram = Compile(intel, *blendMatrix);
        CHECK(blendProgram);
        const Stage outputStage = Transfer(Curve::SRGBEncode);
        auto output = CompileCRTC(std::span(&outputStage, 1), 0, false, 1024);
        CHECK(output && output->ctm.empty() && output->degamma.empty());
        for (float alpha : {0.f, .3f, 1.f})
        {
            const glm::vec3 base(.8f, .6f, .4f), overlay(.2f, .4f, .7f);
            const auto actual = SampleWireCurve<Lut16>(output->gamma,
                glm::mix(EvaluateIntelPlane(intel, *blendProgram, base), EvaluateIntelPlane(intel, *blendProgram, overlay), alpha), false);
            const auto expected = Evaluate(Curve::SRGBEncode, glm::mix(
                Evaluate(Curve::SRGBDecode, Sample3D(table, 17, Sample1D(shaper, base))),
                Evaluate(Curve::SRGBDecode, Sample3D(table, 17, Sample1D(shaper, overlay))), alpha));
            Near(actual, expected, 2.f / 1023);
        }
    }
}

void TestAffineNightVariantLimits()
{
    auto cube = IdentityCube(17);
    for (size_t i = 0; i < cube.size(); i += 4)
    {
        const float g = cube[i + 1] / 65535.f;
        cube[i + 1] = quantize_lut_value_16bit(.5f * g * g);
        cube[i + 2] /= 4;
    }
    // With only two shaper knots, checking shaper samples alone would miss
    // all of this nonlinear cube axis. The crossing checks must reject it.
    auto shaper = IdentityTable(2);
    Transform t;
    t.managed = true;
    t.shaper = shaper;
    t.cube = cube;
    const auto hasMatrix = [](const auto &variants) {
        return std::any_of(variants.begin(), variants.end(), [](const auto &s) {
            return s.size() == 1 && s[0].type == Stage::Type::Matrix;
        });
    };
    CHECK(!hasMatrix(BuildVariants(t)));

    cube = IdentityCube(17);
    for (size_t i = 0; i < cube.size(); i += 4) cube[i + 2] /= 4;
    t.cube = cube;
    CHECK(hasMatrix(BuildVariants(t)));
    shaper = {0, 0, 0, 0, 65535, 65535, 65535, 0, 65535, 65535, 65535, 0};
    t.shaper = shaper;
    CHECK(!hasMatrix(BuildVariants(t))); // pre-matrix clipping cannot be dropped
    shaper = IdentityTable(2);
    t.shaper = shaper;
    t.ctm = Identity();
    (*t.ctm)[3] = .25;
    CHECK(!hasMatrix(BuildVariants(t))); // retain clamping after a layer CTM
}
}

int main()
{
    TestWireFormats();
    TestUnitsAndInterpolation();
    TestPipelines();
    TestCRTCAndTransforms();
    TestAnalyticReference();
    TestWhitePoint();
    TestIntelSingleChannelNightMode();
    TestIntelNativeSDRNightScanout();
    TestAffineNightVariantLimits();
}
