#include "drm_color_helpers.h"

#include <bit>
#include <cstring>
#include <limits>
#include <unordered_set>

namespace drm_color
{
namespace
{
bool Finite(double value)
{
    // The project builds with -ffast-math; std::isfinite can be optimized away.
    return (std::bit_cast<uint64_t>(value) & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL;
}
template<typename T> std::vector<uint8_t> Bytes(std::span<const T> values)
{
    std::vector<uint8_t> result(values.size_bytes());
    std::memcpy(result.data(), values.data(), result.size());
    return result;
}
const Property *Find(const Operation &op, const char *name)
{
    auto it = op.properties.find(name);
    return it == op.properties.end() ? nullptr : &it->second;
}
glm::vec3 CubeValue(std::span<const uint16_t> table, uint32_t index)
{
    return glm::vec3(table[4 * index], table[4 * index + 1], table[4 * index + 2]) / 65535.f;
}
}

Matrix Identity() { return {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}; }
Matrix FromMatrix(const glm::mat3 &matrix)
{
    Matrix result{};
    for (int row = 0; row < 3; row++)
        for (int col = 0; col < 3; col++)
            result[row * 4 + col] = matrix[col][row];
    return result;
}
std::optional<uint64_t> SignMagnitude(double value)
{
    if (!Finite(value) || std::abs(value) >= 2147483648.0)
        return std::nullopt;
    const uint64_t magnitude = uint64_t(std::round(std::abs(value) * 4294967296.0));
    return magnitude | (value < 0.0 && magnitude ? 1ULL << 63 : 0);
}
std::optional<std::vector<uint8_t>> MatrixBlob(const Matrix &matrix, bool affine)
{
    std::vector<uint64_t> values;
    for (int row = 0; row < 3; row++)
    {
        if (!affine && matrix[4 * row + 3] != 0.0)
            return std::nullopt;
        for (int col = 0; col < (affine ? 4 : 3); col++)
        {
            auto value = SignMagnitude(matrix[4 * row + col]);
            if (!value) return std::nullopt;
            values.push_back(*value);
        }
    }
    return Bytes<uint64_t>(values);
}

glm::vec3 Evaluate(Curve curve, glm::vec3 value)
{
    value = glm::max(value, glm::vec3(0.f));
    switch (curve)
    {
        case Curve::SRGBDecode:
            return {srgb_to_linear(value.r), srgb_to_linear(value.g), srgb_to_linear(value.b)};
        case Curve::SRGBEncode:
            return {linear_to_srgb(value.r), linear_to_srgb(value.g), linear_to_srgb(value.b)};
        case Curve::Gamma22Decode: return glm::pow(value, glm::vec3(2.2f));
        case Curve::Gamma22Encode: return glm::pow(value, glm::vec3(1.f / 2.2f));
        case Curve::PQDecode: return pq_to_nits(value) / 80.f;
        case Curve::PQEncode: return nits_to_pq(value * 80.f);
        case Curve::PQNormalizedDecode: return pq_to_nits(value) / 10000.f;
        case Curve::PQNormalizedEncode: return nits_to_pq(value * 10000.f);
    }
    return {};
}
std::string CurveName(Curve curve)
{
    switch (curve)
    {
        case Curve::SRGBDecode: return "sRGB EOTF";
        case Curve::SRGBEncode: return "sRGB Inverse EOTF";
        case Curve::Gamma22Decode: return "Gamma 2.2";
        case Curve::Gamma22Encode: return "Gamma 2.2 Inverse";
        case Curve::PQDecode: return "PQ 125 EOTF";
        case Curve::PQEncode: return "PQ 125 Inverse EOTF";
        default: return {};
    }
}
Stage Transfer(Curve curve)
{
    Stage result;
    result.function = [curve](glm::vec3 value) { return Evaluate(curve, value); };
    result.curve = CurveName(curve);
    result.bounded = curve != Curve::PQEncode;
    return result;
}
Stage Affine(const Matrix &matrix)
{
    Stage result;
    result.type = Stage::Type::Matrix;
    result.matrix = matrix;
    return result;
}
glm::vec3 Sample1D(std::span<const uint16_t> table, glm::vec3 input)
{
    glm::vec3 result{};
    const size_t size = table.size() / 4;
    if (size < 2) return result;
    input = glm::clamp(input, glm::vec3(0.f), glm::vec3(1.f)) * float(size - 1);
    for (int c = 0; c < 3; c++)
    {
        size_t lo = std::min(size_t(input[c]), size - 2);
        result[c] = std::lerp(float(table[4 * lo + c]), float(table[4 * (lo + 1) + c]), input[c] - float(lo)) / 65535.f;
    }
    return result;
}
Stage Table1D(std::span<const uint16_t> table)
{
    Stage result;
    result.function = [table](glm::vec3 value) { return Sample1D(table, value); };
    return result;
}
glm::vec3 Sample3D(std::span<const uint16_t> table, uint32_t edge, glm::vec3 input)
{
    if (edge < 2 || table.size() != size_t(edge) * edge * edge * 4) return {};
    glm::vec3 pos = glm::clamp(input, glm::vec3(0.f), glm::vec3(1.f)) * float(edge - 1);
    glm::uvec3 lo = glm::min(glm::uvec3(pos), glm::uvec3(edge - 2));
    glm::vec3 f = pos - glm::vec3(lo);
    const uint32_t stride[] = {1, edge, edge * edge};
    std::array<int, 3> order{0, 1, 2};
    std::sort(order.begin(), order.end(), [&](int a, int b) { return f[a] > f[b]; });
    uint32_t index = lo.r + edge * (lo.g + edge * lo.b);
    glm::vec3 previous = CubeValue(table, index), result = previous;
    for (int c : order)
    {
        index += stride[c];
        glm::vec3 next = CubeValue(table, index);
        result += f[c] * (next - previous);
        previous = next;
    }
    return result;
}
Stage Table3D(std::span<const uint16_t> table, uint32_t edge)
{
    Stage result;
    result.type = Stage::Type::Lut3D;
    result.cube = table;
    result.edge = edge;
    return result;
}
Stage Compose(const Stage &first, const Stage &second)
{
    Stage result;
    result.function = [a = first.function, b = second.function](glm::vec3 v) { return b(a(v)); };
    result.bounded = first.bounded && second.bounded;
    return result;
}
std::optional<Stage> Separable(std::span<const uint16_t> cube, uint32_t edge)
{
    if (edge < 2 || cube.size() != size_t(edge) * edge * edge * 4) return std::nullopt;
    const uint32_t stride[] = {1, edge, edge * edge};
    for (uint32_t b = 0; b < edge; b++)
        for (uint32_t g = 0; g < edge; g++)
            for (uint32_t r = 0; r < edge; r++)
            {
                const uint32_t coord[] = {r, g, b};
                const uint32_t index = r + edge * (g + edge * b);
                for (int c = 0; c < 3; c++)
                    if (cube[4 * index + c] != cube[4 * stride[c] * coord[c] + c])
                        return std::nullopt;
            }
    Stage result;
    result.function = [cube, edge](glm::vec3 v) { return Sample3D(cube, edge, v); };
    return result;
}
std::optional<std::vector<uint8_t>> SampleCurve(const Stage &stage, uint32_t size, bool wide)
{
    if (stage.type != Stage::Type::Curve || !stage.bounded || !stage.function || size < 2 || size > 65536)
        return std::nullopt;
    std::vector<Lut32> lut32;
    std::vector<Lut16> lut16;
    std::vector<glm::vec3> samples;
    samples.reserve(size);
    for (uint32_t i = 0; i < size; i++)
    {
        const glm::vec3 value = stage.function(glm::vec3(float(i) / float(size - 1)));
        uint32_t channels[3];
        for (int c = 0; c < 3; c++)
        {
            if (!Finite(value[c]) || value[c] < -0.000001f || value[c] > 1.000001f) return std::nullopt;
            channels[c] = uint32_t(std::round(double(clamp01(value[c])) * (wide ? 4294967295.0 : 65535.0)));
        }
        if (wide) lut32.push_back({channels[0], channels[1], channels[2], 0});
        else lut16.push_back({uint16_t(channels[0]), uint16_t(channels[1]), uint16_t(channels[2]), 0});
        samples.push_back(glm::vec3(channels[0], channels[1], channels[2]) / (wide ? 4294967295.f : 65535.f));
    }
    // Uniform linear-light LUTs are often too coarse for inverse gamma/PQ
    // near black. A valid atomic commit does not establish color fidelity.
    // Reject curves which exceed one 10-bit code in the sampled model, rather
    // than silently accepting severe banding on matrix-only hardware.
    const uint32_t checks = std::max(16384u, size * 4);
    for (uint32_t i = 1; i < checks; i++)
    {
        const float x = float(i) / checks;
        const float position = x * (size - 1);
        const uint32_t lo = std::min(uint32_t(position), size - 2);
        const glm::vec3 interpolated = glm::mix(samples[lo], samples[lo + 1], position - lo);
        const glm::vec3 expected = stage.function(glm::vec3(x));
        for (int c = 0; c < 3; c++)
            if (!Finite(expected[c]) || std::abs(interpolated[c] - expected[c]) > 1.f / 1023.f)
                return std::nullopt;
    }
    return wide ? Bytes<Lut32>(lut32) : Bytes<Lut16>(lut16);
}

bool ValidPipeline(const Pipeline &pipeline)
{
    if (!pipeline.root || pipeline.operations.empty() || pipeline.operations.front().id != pipeline.root) return false;
    std::unordered_set<uint32_t> seen;
    for (size_t i = 0; i < pipeline.operations.size(); i++)
    {
        const auto &op = pipeline.operations[i];
        const auto *next = Find(op, "NEXT");
        if (!op.id || !seen.insert(op.id).second || !next ||
            next->value != (i + 1 < pipeline.operations.size() ? pipeline.operations[i + 1].id : 0)) return false;
    }
    return true;
}

std::vector<std::vector<Stage>> BuildVariants(const Transform &t)
{
    std::vector<std::vector<Stage>> variants;
    std::vector<Stage> prefix;
    if (t.input == Encoding::ScRGB || t.input == Encoding::Passthrough)
    {
        if (t.ctm) prefix.push_back(Affine(*t.ctm));
        if (t.input == Encoding::ScRGB && t.managed) prefix.push_back(Transfer(Curve::PQEncode));
    }
    else if (t.ctm)
    {
        prefix.push_back(Transfer(t.input == Encoding::PQ ? Curve::PQDecode : Curve::SRGBDecode));
        prefix.push_back(Affine(*t.ctm));
        prefix.push_back(Transfer(t.input == Encoding::PQ ? Curve::PQEncode : Curve::SRGBEncode));
    }
    auto finish = [&](std::vector<Stage> stages)
    {
        if (t.blend)
            stages.push_back(Transfer(t.output == EOTF_PQ ? Curve::PQNormalizedDecode : Curve::SRGBDecode));
        // Adjacent independent curves can share a single hardware 1D LUT.
        for (size_t i = 1; i < stages.size();)
        {
            if (stages[i - 1].type == Stage::Type::Curve && stages[i].type == Stage::Type::Curve &&
                stages[i - 1].bounded && stages[i].bounded)
            {
                stages[i - 1] = Compose(stages[i - 1], stages[i]);
                stages.erase(stages.begin() + i);
            }
            else ++i;
        }
        variants.push_back(std::move(stages));
    };
    if (!t.managed || t.input == Encoding::Passthrough)
    {
        finish(prefix);
        return variants;
    }
    if (t.shaper.size() < 8 || t.cube.size() != size_t(t.edge) * t.edge * t.edge * 4)
        return variants;
    auto full = prefix;
    full.push_back(Table1D(t.shaper));
    full.push_back(Table3D(t.cube, t.edge));
    finish(std::move(full));
    if (auto separable = Separable(t.cube, t.edge))
    {
        auto simple = prefix;
        simple.push_back(Compose(Table1D(t.shaper), *separable));
        finish(std::move(simple));
    }
    // Arbitrary look LUTs and nonlinear gamut/tone mappings never reach this
    // branch: the generator supplies an analytic description only when exact.
    if (t.analytic)
    {
        const auto &a = *t.analytic;
        // Retain the prefix even for analytic programs: scRGB -> PQ clips at
        // 10,000 nits, and a layer CTM precedes the LUT's source EOTF decode.
        // Combining its matrix across those nonlinear stages would be wrong.
        std::vector<Stage> analytic = prefix;
        analytic.push_back(Transfer(a.sourceEOTF == EOTF_PQ ? Curve::PQNormalizedDecode : Curve::Gamma22Decode));
        const double sourceNits = a.sourceEOTF == EOTF_PQ ? 10000.0 : a.sdrLuminance;
        const double destNits = a.destEOTF == EOTF_PQ ? 10000.0 : a.sdrLuminance;
        Matrix matrix = FromMatrix(a.matrix);
        for (double &value : matrix) value *= sourceNits / destNits;
        if (matrix != Identity()) analytic.push_back(Affine(matrix));
        analytic.push_back(Transfer(a.destEOTF == EOTF_PQ ? Curve::PQNormalizedEncode : Curve::Gamma22Encode));
        finish(std::move(analytic));
    }
    return variants;
}

std::optional<Program> Compile(const Pipeline &pipeline, std::span<const Stage> stages)
{
    if (!ValidPipeline(pipeline)) return std::nullopt;
    Program program;
    auto match = [&](const Operation &op, const Stage &stage) -> std::optional<Write>
    {
        const Property *data = Find(op, "DATA");
        std::optional<std::vector<uint8_t>> blob;
        if (stage.type == Stage::Type::Curve && op.type == "1D Curve")
        {
            const auto *curve = Find(op, "CURVE_1D_TYPE");
            if (curve && !stage.curve.empty() && curve->enums.contains(stage.curve))
                return Write{op.id, curve->id, curve->enums.at(stage.curve), {}};
        }
        if (stage.type == Stage::Type::FixedMatrix && op.type == "Fixed Matrix")
        {
            const auto *matrix = Find(op, "FIXED_MATRIX_TYPE");
            if (matrix && matrix->enums.contains(stage.fixedMatrix))
                return Write{op.id, matrix->id, matrix->enums.at(stage.fixedMatrix), {}};
        }
        if (stage.type == Stage::Type::Matrix && op.type == "Multiplier")
        {
            auto scalar = Identity();
            scalar[0] = scalar[5] = scalar[10] = stage.matrix[0];
            const auto *multiplier = Find(op, "MULTIPLIER");
            auto value = SignMagnitude(stage.matrix[0]);
            if (scalar == stage.matrix && multiplier && value)
                return Write{op.id, multiplier->id, *value, {}};
        }
        if (!data || data->immutable) return std::nullopt;
        if (stage.type == Stage::Type::Matrix && op.type == "3x4 Matrix")
            blob = MatrixBlob(stage.matrix, true);
        const auto *size = Find(op, "SIZE");
        if (stage.type == Stage::Type::Curve && op.type == "1D LUT" && size)
        {
            const auto *interpolation = Find(op, "LUT1D_INTERPOLATION");
            if (!interpolation || !interpolation->enums.contains("Linear")) return std::nullopt;
            if (interpolation->immutable && interpolation->value != interpolation->enums.at("Linear")) return std::nullopt;
            if (size->value > 65536) return std::nullopt;
            blob = SampleCurve(stage, uint32_t(size->value), true);
        }
        if (stage.type == Stage::Type::Lut3D && op.type == "3D LUT" && size)
        {
            const auto *interpolation = Find(op, "LUT3D_INTERPOLATION");
            // Do not silently introduce a second interpolation grid.
            if (!interpolation || !interpolation->enums.contains("Tetrahedral") || size->value != stage.edge)
                return std::nullopt;
            if (interpolation->immutable && interpolation->value != interpolation->enums.at("Tetrahedral")) return std::nullopt;
            if (stage.edge < 2 || stage.cube.size() != size_t(stage.edge) * stage.edge * stage.edge * 4) return std::nullopt;
            std::vector<Lut32> values;
            for (size_t i = 0; i < stage.cube.size(); i += 4)
                values.push_back({uint32_t(stage.cube[i]) * 65537u, uint32_t(stage.cube[i + 1]) * 65537u,
                    uint32_t(stage.cube[i + 2]) * 65537u, 0});
            blob = Bytes<Lut32>(values);
        }
        if (!blob) return std::nullopt;
        return Write{op.id, data->id, 0, std::move(*blob)};
    };
    auto search = [&](auto &&self, size_t operation, size_t stage) -> bool
    {
        if (operation == pipeline.operations.size()) return stage == stages.size();
        const Operation &op = pipeline.operations[operation];
        const Property *bypass = Find(op, "BYPASS");
        const size_t start = program.size();
        if (stage < stages.size() && (!bypass || !bypass->immutable || bypass->value == 0))
        {
            if (auto write = match(op, stages[stage]))
            {
                if (bypass && !bypass->immutable) program.push_back({op.id, bypass->id, 0, {}});
                const char *name = op.type == "1D LUT" ? "LUT1D_INTERPOLATION" : "LUT3D_INTERPOLATION";
                if (op.type == "1D LUT" || op.type == "3D LUT")
                {
                    const auto *interpolation = Find(op, name);
                    if (interpolation && !interpolation->immutable)
                        program.push_back({op.id, interpolation->id,
                            interpolation->enums.at(op.type == "1D LUT" ? "Linear" : "Tetrahedral"), {}});
                }
                program.push_back(std::move(*write));
                if (self(self, operation + 1, stage + 1)) return true;
                program.resize(start);
            }
        }
        if (bypass && (!bypass->immutable || bypass->value == 1))
        {
            if (!bypass->immutable) program.push_back({op.id, bypass->id, 1, {}});
            if (self(self, operation + 1, stage)) return true;
            program.resize(start);
        }
        return false;
    };
    if (!search(search, 0, 0)) return std::nullopt;
    return program;
}

std::optional<CrtcProgram> CompileCRTC(std::span<const Stage> stages,
    uint32_t degammaSize, bool hasCTM, uint32_t gammaSize)
{
    CrtcProgram result;
    size_t i = 0;
    // A single curve can run entirely in GAMMA_LUT.
    if (i + 1 < stages.size() && stages[i].type == Stage::Type::Curve)
    {
        auto blob = SampleCurve(stages[i++], degammaSize, false);
        if (!blob) return std::nullopt;
        result.degamma = std::move(*blob);
    }
    if (i < stages.size() && stages[i].type == Stage::Type::Matrix)
    {
        if (!hasCTM) return std::nullopt;
        auto blob = MatrixBlob(stages[i++].matrix, false);
        if (!blob) return std::nullopt;
        result.ctm = std::move(*blob);
    }
    if (i < stages.size() && stages[i].type == Stage::Type::Curve)
    {
        auto blob = SampleCurve(stages[i++], gammaSize, false);
        if (!blob) return std::nullopt;
        result.gamma = std::move(*blob);
    }
    if (i != stages.size()) return std::nullopt;
    return result;
}
}
