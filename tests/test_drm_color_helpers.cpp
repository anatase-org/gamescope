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
}

int main()
{
    TestWireFormats();
    TestUnitsAndInterpolation();
    TestPipelines();
    TestCRTCAndTransforms();
    TestAnalyticReference();
    TestWhitePoint();
}
