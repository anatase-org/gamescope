#pragma once

#include "color_helpers.h"

#include <array>
#include <functional>
#include <map>
#include <span>
#include <string>

// This module has no DRM device dependency. In particular, compiling a program
// must not change committed state while libliftoff explores plane assignments.
namespace drm_color
{
struct Property
{
    uint32_t id = 0;
    uint64_t value = 0;
    bool immutable = false;
    std::map<std::string, uint64_t> enums;
};
using Properties = std::map<std::string, Property>;

struct Operation
{
    uint32_t id = 0;
    std::string type;
    Properties properties;
    // Some drivers expose a shared hardware curve through an RGB LUT blob.
    // These operations must receive equal red, green, and blue entries.
    bool singleChannel = false;
};
struct Pipeline
{
    uint32_t plane = 0;
    uint64_t root = 0; // COLOR_PIPELINE enum value, not its enum index
    std::vector<Operation> operations;
};

using Matrix = std::array<double, 12>; // row-major affine 3x4
Matrix Identity();
Matrix FromMatrix(const glm::mat3 &matrix);
std::optional<uint64_t> SignMagnitude(double value);
std::optional<std::vector<uint8_t>> MatrixBlob(const Matrix &matrix, bool affine);

enum class Curve { SRGBDecode, SRGBEncode, Gamma22Decode, Gamma22Encode,
                   PQDecode, PQEncode, PQNormalizedDecode, PQNormalizedEncode };
glm::vec3 Evaluate(Curve curve, glm::vec3 value);
std::string CurveName(Curve curve);

struct Stage
{
    enum class Type { Curve, Matrix, Lut3D, FixedMatrix } type = Type::Curve;
    std::function<glm::vec3(glm::vec3)> function;
    std::string curve;
    bool bounded = true; // unbounded scRGB must use a fixed curve, never a UNORM LUT
    Matrix matrix = Identity();
    std::span<const uint16_t> cube;
    uint32_t edge = 0;
    std::string fixedMatrix;
};
Stage Transfer(Curve curve);
Stage Affine(const Matrix &matrix);
Stage Table1D(std::span<const uint16_t> table);
Stage Table3D(std::span<const uint16_t> table, uint32_t edge);
Stage Compose(const Stage &first, const Stage &second);
std::optional<Stage> Separable(std::span<const uint16_t> cube, uint32_t edge);
glm::vec3 Sample1D(std::span<const uint16_t> table, glm::vec3 input);
glm::vec3 Sample3D(std::span<const uint16_t> table, uint32_t edge, glm::vec3 input);

// Wire layouts are intentionally private, so CRTC-only builds still work with
// libdrm versions predating drm_color_lut32 and drm_color_ctm_3x4.
struct Lut32 { uint32_t red, green, blue, reserved; };
struct Lut16 { uint16_t red, green, blue, reserved; };
std::optional<std::vector<uint8_t>> SampleCurve(const Stage &stage, uint32_t size, bool wide,
    bool singleChannel = false);

enum class Encoding { SDR, PQ, ScRGB, Passthrough };
struct Transform
{
    Encoding input = Encoding::SDR;
    EOTF output = EOTF_Gamma22;
    bool managed = false;
    bool blend = false;
    std::span<const uint16_t> shaper;
    std::span<const uint16_t> cube;
    uint32_t edge = 17;
    std::optional<color_matrix_transform_t> analytic;
    std::optional<Matrix> ctm;
};
std::vector<std::vector<Stage>> BuildVariants(const Transform &transform);

struct Write
{
    uint32_t object = 0;
    uint32_t property = 0;
    uint64_t value = 0;
    std::vector<uint8_t> blob;
};
using Program = std::vector<Write>;

// Missing BYPASS is significant: such an operation must be programmed, including
// non-bypassable curves in NVIDIA's ICtCp pipelines.
std::optional<Program> Compile(const Pipeline &pipeline, std::span<const Stage> stages);
bool ValidPipeline(const Pipeline &pipeline);
struct CrtcProgram
{
    std::vector<uint8_t> degamma, ctm, gamma;
};
std::optional<CrtcProgram> CompileCRTC(std::span<const Stage> stages,
    uint32_t degammaSize, bool hasCTM, uint32_t gammaSize);
}
