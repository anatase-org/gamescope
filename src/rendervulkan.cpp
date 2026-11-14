// Initialize Vulkan and composite stuff with a compute queue

#include <cassert>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <algorithm>
#include <array>
#include <bitset>
#include <deque>
#include <thread>
#include <dlfcn.h>
#include "vulkan_include.h"
#include "Utils/Algorithm.h"

#if defined(__linux__)
#include <sys/sysmacros.h>
#endif

// Used to remove the config struct alignment specified by the NIS header
#define NIS_ALIGNED(x)
// NIS_Config needs to be included before the X11 headers because of conflicting defines introduced by X11
#include "shaders/NVIDIAImageScaling/NIS/NIS_Config.h"

#include <drm_fourcc.h>
#include "hdmi.h"
#if HAVE_DRM
#include "drm_include.h"
#endif
#include "wlr_begin.hpp"
#include <wlr/render/drm_format_set.h>
#include "wlr_end.hpp"

#include "rendervulkan.hpp"
#include <glm/gtc/packing.hpp>
#include "main.hpp"
#include "steamcompmgr.hpp"
#include "frame_generation_config.hpp"
#include "frame_generation_pacing.hpp"
#include "frame_generation_vulkan.hpp"
#include "log.hpp"
#include "Utils/Process.h"

#include "cs_composite_blit.h"
#include "vs_composite.h"
#include "fs_composite.h"
#include "cs_composite_blit_rgb10a2.h"
#include "cs_composite_blur.h"
#include "cs_composite_blur_cond.h"
#include "cs_composite_rcas.h"
#include "cs_composite_rcas_1px.h"
#include "cs_composite_rcas_fp16.h"
#include "cs_composite_rcas_fp16_1px.h"
#include "cs_composite_rcas_rgba16f.h"
#include "cs_composite_rcas_rgba16f_1px.h"
#include "cs_composite_rcas_rgb10a2.h"
#include "cs_composite_rcas_rgb10a2_1px.h"
#include "cs_easu.h"
#include "cs_easu_fp16.h"
#include "cs_easu_rgba16f.h"
#include "cs_easu_fp16_rgba16f.h"
#include "cs_gaussian_blur_horizontal.h"
#include "cs_nis.h"
#include "cs_nis_fp16.h"
#include "cs_nis_hdr_linear.h"
#include "cs_nis_hdr_linear_fp16.h"
#include "cs_rgb_to_nv12.h"
#include "cs_sgsr.h"
#include "cs_composite_blur_rgba16f.h"
#include "cs_composite_blur_rgb10a2.h"
#include "cs_composite_blur_cond_rgba16f.h"
#include "cs_composite_blur_cond_rgb10a2.h"
#include "cs_gaussian_blur_horizontal_rgba16f.h"
#include "cs_gaussian_blur_horizontal_rgb10a2.h"
#include "cs_composite_blit_rgba16f.h"
#include "cs_nis_rgb10a2.h"
#include "cs_nis_fp16_rgb10a2.h"
#include "cs_composite_blit_formatless.h"
#include "cs_composite_blit_rgb10a2_formatless.h"
#include "cs_composite_blur_formatless.h"
#include "cs_composite_blur_cond_formatless.h"
#include "cs_composite_rcas_formatless.h"
#include "cs_composite_rcas_1px_formatless.h"
#include "cs_composite_rcas_fp16_formatless.h"
#include "cs_composite_rcas_fp16_1px_formatless.h"
#include "cs_composite_rcas_rgba16f_formatless.h"
#include "cs_composite_rcas_rgba16f_1px_formatless.h"
#include "cs_composite_rcas_rgb10a2_formatless.h"
#include "cs_composite_rcas_rgb10a2_1px_formatless.h"
#include "cs_easu_formatless.h"
#include "cs_easu_fp16_formatless.h"
#include "cs_easu_rgba16f_formatless.h"
#include "cs_easu_fp16_rgba16f_formatless.h"
#include "cs_gaussian_blur_horizontal_formatless.h"
#include "cs_nis_formatless.h"
#include "cs_nis_fp16_formatless.h"
#include "cs_nis_hdr_linear_formatless.h"
#include "cs_nis_hdr_linear_fp16_formatless.h"
#include "cs_rgb_to_nv12_formatless.h"
#include "cs_sgsr_formatless.h"
#include "cs_composite_blur_rgba16f_formatless.h"
#include "cs_composite_blur_rgb10a2_formatless.h"
#include "cs_composite_blur_cond_rgba16f_formatless.h"
#include "cs_composite_blur_cond_rgb10a2_formatless.h"
#include "cs_gaussian_blur_horizontal_rgba16f_formatless.h"
#include "cs_gaussian_blur_horizontal_rgb10a2_formatless.h"
#include "cs_composite_blit_rgba16f_formatless.h"
#include "cs_nis_rgb10a2_formatless.h"
#include "cs_nis_fp16_rgb10a2_formatless.h"


gamescope::ConVar<bool> cv_raster_damage("raster_damage", true, "Reuse input shadows and output contents outside client damage.");


struct CompositeState_t
{
	struct Layer
	{
		uint64_t texture = 0, version = 0;
		SurfaceDamage damage;
		std::array<float, 5> geometry{}; // scale, pixel-center offset, opacity
		std::array<uint32_t, 2> size{};
		std::array<uint32_t, 6> flags{}; // colorspace, filter, alpha mode, black border, format, applied color management
		std::array<float, 12> ctm{};
		bool operator==(const Layer &) const = default;
	};
	std::array<Layer, k_nMaxLayers> layers{};
	std::array<uint64_t, EOTF_Count * 4> luts{};
	CompositeShaderConfig_t config;
	EOTF outputTF;
	int layerCount;
};

static CompositeState_t composite_state(const FrameInfo_t *frame, EOTF outputTF, CompositeShaderConfig_t config)
{
	CompositeState_t state{};
	state.config = config;
	state.outputTF = outputTF;
	state.layerCount = frame->layers.count();
	for (int i = 0; i < frame->layers.count(); ++i)
	{
		const auto &layer = frame->layers[i];
		auto &s = state.layers[i];
		s.texture = layer.tex->textureID;
		s.version = layer.tex->contentVersion.load(std::memory_order_relaxed);
		if (layer.damage.texture == s.texture)
			s.damage = layer.damage;
		const auto offset = layer.offsetPixelCenter();
		s.geometry = {layer.scale.x, layer.scale.y, offset.x, offset.y, layer.opacity};
		s.size = {layer.tex->width(), layer.tex->height()};
		s.flags = {uint32_t(layer.colorspace), uint32_t(layer.filter), uint32_t(layer.eAlphaBlendingMode), uint32_t(layer.blackBorder), uint32_t(layer.tex->format()), uint32_t(layer.outputColorMgmtApplied)};
		if (layer.ctm)
		{
			const auto &matrix = layer.ctm->View<glm::mat3x4>();
			memcpy(s.ctm.data(), &matrix, sizeof(matrix));
		}
	}
	for (int i = 0; i < EOTF_Count; ++i)
	{
		if (frame->shaperLut[i])
		{
			state.luts[4 * i] = frame->shaperLut[i]->textureID;
			state.luts[4 * i + 1] = frame->shaperLut[i]->contentVersion.load(std::memory_order_relaxed);
		}
		if (frame->lut3D[i])
		{
			state.luts[4 * i + 2] = frame->lut3D[i]->textureID;
			state.luts[4 * i + 3] = frame->lut3D[i]->contentVersion.load(std::memory_order_relaxed);
		}
	}
	return state;
}

static VkRect2D composite_layer_bounds(const CompositeState_t::Layer &s, uint32_t width, uint32_t height)
{
	if (s.flags[3])
		return {{0, 0}, {width, height}};
	const float x = float(s.size[0]) / s.geometry[0], y = float(s.size[1]) / s.geometry[1];
	const int left = std::clamp(int(floorf(std::min(0.0f, x) - s.geometry[2])), 0, int(width));
	const int top = std::clamp(int(floorf(std::min(0.0f, y) - s.geometry[3])), 0, int(height));
	const int right = std::clamp(int(ceilf(std::max(0.0f, x) - s.geometry[2])) + 1, left, int(width));
	const int bottom = std::clamp(int(ceilf(std::max(0.0f, y) - s.geometry[3])) + 1, top, int(height));
	return {{left, top}, {uint32_t(right - left), uint32_t(bottom - top)}};
}

static VkRect2D intersect_rect(VkRect2D a, VkRect2D b)
{
	const int x = std::max(a.offset.x, b.offset.x), y = std::max(a.offset.y, b.offset.y);
	const int right = std::min(a.offset.x + int(a.extent.width), b.offset.x + int(b.extent.width));
	const int bottom = std::min(a.offset.y + int(a.extent.height), b.offset.y + int(b.extent.height));
	return {{x, y}, {uint32_t(std::max(0, right - x)), uint32_t(std::max(0, bottom - y))}};
}

static VkRect2D union_rect(VkRect2D a, VkRect2D b)
{
	if (!a.extent.width || !a.extent.height) return b;
	if (!b.extent.width || !b.extent.height) return a;
	const int x = std::min(a.offset.x, b.offset.x), y = std::min(a.offset.y, b.offset.y);
	const int right = std::max(a.offset.x + int(a.extent.width), b.offset.x + int(b.extent.width));
	const int bottom = std::max(a.offset.y + int(a.extent.height), b.offset.y + int(b.extent.height));
	return {{x, y}, {uint32_t(right - x), uint32_t(bottom - y)}};
}

static VkRect2D composite_damage_bounds(const CompositeState_t::Layer &s, wlr_box damage, uint32_t width, uint32_t height)
{
	if (damage.width <= 0 || damage.height <= 0)
		return {};
	// Include filtering footprints and fractional output positions. The normal
	// raster path excludes wider reconstruction filters and rotations.
	const float x1 = (damage.x - 1.0f) / s.geometry[0] - s.geometry[2];
	const float x2 = (damage.x + damage.width + 1.0f) / s.geometry[0] - s.geometry[2];
	const float y1 = (damage.y - 1.0f) / s.geometry[1] - s.geometry[3];
	const float y2 = (damage.y + damage.height + 1.0f) / s.geometry[1] - s.geometry[3];
	const int left = std::clamp(int(floorf(std::min(x1, x2))), 0, int(width));
	const int top = std::clamp(int(floorf(std::min(y1, y2))), 0, int(height));
	const int right = std::clamp(int(ceilf(std::max(x1, x2))) + 1, left, int(width));
	const int bottom = std::clamp(int(ceilf(std::max(y1, y2))) + 1, top, int(height));
	return {{left, top}, {uint32_t(right - left), uint32_t(bottom - top)}};
}

gamescope::ConVar<bool> cv_raster_linear_read("raster_linear_read", true, "Read compatible linear inputs without a sampled shadow copy.");

#define A_CPU
#include "shaders/ffx_a.h"
#include "shaders/ffx_fsr1.h"

#include "reshade_effect_manager.hpp"

ReshadeEffectPipeline *g_pLastReshadeEffect = nullptr;

extern bool g_bWasPartialComposite;
extern bool g_bAllowDeferredBackend;

static constexpr mat3x4 g_rgb2yuv_srgb_to_bt601_limited = {{
  { 0.257f, 0.504f, 0.098f, 0.0625f },
  { -0.148f, -0.291f, 0.439f, 0.5f },
  { 0.439f, -0.368f, -0.071f, 0.5f },
}};

static constexpr mat3x4 g_rgb2yuv_srgb_to_bt601 = {{
  { 0.299f, 0.587f, 0.114f, 0.0f },
  { -0.169f, -0.331f, 0.500f, 0.5f },
  { 0.500f, -0.419f, -0.081f, 0.5f },
}};

static constexpr mat3x4 g_rgb2yuv_srgb_to_bt709_limited = {{
  { 0.1826f, 0.6142f, 0.0620f, 0.0625f },
  { -0.1006f, -0.3386f, 0.4392f, 0.5f },
  { 0.4392f, -0.3989f, -0.0403f, 0.5f },
}};

static constexpr mat3x4 g_rgb2yuv_srgb_to_bt709_full = {{
  { 0.2126f, 0.7152f, 0.0722f, 0.0f },
  { -0.1146f, -0.3854f, 0.5000f, 0.5f },
  { 0.5000f, -0.4542f, -0.0458f, 0.5f },
}};

static const mat3x4& colorspace_to_conversion_from_srgb_matrix(EStreamColorspace colorspace) {
	switch (colorspace) {
		default:
		case k_EStreamColorspace_BT601:			return g_rgb2yuv_srgb_to_bt601_limited;
		case k_EStreamColorspace_BT601_Full:	return g_rgb2yuv_srgb_to_bt601;
		case k_EStreamColorspace_BT709:			return g_rgb2yuv_srgb_to_bt709_limited;
		case k_EStreamColorspace_BT709_Full:	return g_rgb2yuv_srgb_to_bt709_full;
	}
}

PFN_vkGetInstanceProcAddr g_pfn_vkGetInstanceProcAddr;
PFN_vkCreateInstance g_pfn_vkCreateInstance;

static VkResult vulkan_load_module()
{
	static VkResult s_result = []()
	{
		void* pModule = dlopen( "libvulkan.so.1", RTLD_NOW | RTLD_LOCAL );
		if ( !pModule )
			pModule = dlopen( "libvulkan.so", RTLD_NOW | RTLD_LOCAL );
		if ( !pModule )
			return VK_ERROR_INITIALIZATION_FAILED;

		g_pfn_vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)dlsym( pModule, "vkGetInstanceProcAddr" );
		if ( !g_pfn_vkGetInstanceProcAddr )
			return VK_ERROR_INITIALIZATION_FAILED;

		g_pfn_vkCreateInstance = (PFN_vkCreateInstance) g_pfn_vkGetInstanceProcAddr( nullptr, "vkCreateInstance" );
		if ( !g_pfn_vkCreateInstance )
			return VK_ERROR_INITIALIZATION_FAILED;

		return VK_SUCCESS;
	}();

	return s_result;
}

VulkanOutput_t g_output;

uint32_t g_uCompositeDebug = 0u;
gamescope::ConVar<uint32_t> cv_composite_debug{ "composite_debug", 0, "Debug composition flags" };
gamescope::ConVar<int> cv_fsr_rcas_fp16{ "fsr_rcas_fp16", -1, "FSR RCAS FP16 mode: -1 = automatic on AMD, 0 = disabled, 1 = enabled when supported" };

static std::map< VkFormat, std::map< uint64_t, VkDrmFormatModifierPropertiesEXT > > DRMModifierProps = {};
static std::unordered_map<uint32_t, std::vector<uint64_t>> s_SampledModifierFormats = {};
static struct wlr_drm_format_set sampledShmFormats = {};
static struct wlr_drm_format_set sampledDRMFormats = {};

std::span<const uint64_t> GetSupportedSampleModifiers( uint32_t uDrmFormat )
{
	auto iter = s_SampledModifierFormats.find( uDrmFormat );
	if ( iter == s_SampledModifierFormats.end() )
		return std::span<const uint64_t>{};

	return std::span<const uint64_t>{ iter->second.begin(), iter->second.end() };
}

static LogScope vk_log("vulkan");

static void vk_errorf(VkResult result, const char *fmt, ...) {
	static char buf[1024];
	va_list args;
	va_start(args, fmt);
	vsnprintf(buf, sizeof(buf), fmt, args);
	va_end(args);

	vk_log.errorf("%s (VkResult: %d)", buf, result);
}

// For when device is up and it would be totally fatal to fail
#define vk_check( x ) \
	do \
	{ \
		VkResult check_res = VK_SUCCESS; \
		if ( ( check_res = ( x ) ) != VK_SUCCESS ) \
		{ \
			vk_errorf( check_res, #x " failed!" ); \
			abort(); \
		} \
	} while ( 0 )

template<typename Target, typename Base>
Target *pNextFind(const Base *base, VkStructureType sType)
{
	for ( ; base; base = (const Base *)base->pNext )
	{
		if (base->sType == sType)
			return (Target *) base;
	}
	return nullptr;
}

#define VK_STRUCTURE_TYPE_WSI_IMAGE_CREATE_INFO_MESA (VkStructureType)1000001002
#define VK_STRUCTURE_TYPE_WSI_MEMORY_ALLOCATE_INFO_MESA (VkStructureType)1000001003

struct wsi_image_create_info {
	VkStructureType sType;
	const void *pNext;
	bool scanout;

	uint32_t modifier_count;
	const uint64_t *modifiers;
};

struct wsi_memory_allocate_info {
    VkStructureType sType;
    const void *pNext;
    bool implicit_sync;
};

// DRM doesn't always have 32bit floating point formats, so add our own if necessary

#ifndef DRM_FORMAT_ABGR32323232F
#define DRM_FORMAT_ABGR32323232F fourcc_code('A', 'B', '8', 'F')
#endif

#ifndef DRM_FORMAT_R16F
#define DRM_FORMAT_R16F fourcc_code('R', '1', '6', 'F')
#endif

#ifndef DRM_FORMAT_R32F
#define DRM_FORMAT_R32F fourcc_code('R', '3', '2', 'F')
#endif

struct {
	uint32_t DRMFormat;
	VkFormat vkFormat;
	VkFormat vkFormatSrgb;
	uint32_t bpp;
	bool bHasAlpha;
	bool internal;
} s_DRMVKFormatTable[] = {
	{ DRM_FORMAT_ARGB8888, VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_B8G8R8A8_SRGB, 4, true, false },
	{ DRM_FORMAT_XRGB8888, VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_B8G8R8A8_SRGB, 4, false, false },
	{ DRM_FORMAT_ABGR8888, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SRGB, 4, true, false },
	{ DRM_FORMAT_XBGR8888, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SRGB, 4, false, false },
	{ DRM_FORMAT_RGB565, VK_FORMAT_R5G6B5_UNORM_PACK16, VK_FORMAT_R5G6B5_UNORM_PACK16, 1, false, false },
	{ DRM_FORMAT_NV12, VK_FORMAT_G8_B8R8_2PLANE_420_UNORM, VK_FORMAT_G8_B8R8_2PLANE_420_UNORM, 0, false, false },
	{ DRM_FORMAT_ABGR16161616F, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT, 8, true, false },
	{ DRM_FORMAT_XBGR16161616F, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT, 8, false, false },
	{ DRM_FORMAT_ABGR16161616, VK_FORMAT_R16G16B16A16_UNORM, VK_FORMAT_R16G16B16A16_UNORM, 8, true, false },
	{ DRM_FORMAT_XBGR16161616, VK_FORMAT_R16G16B16A16_UNORM, VK_FORMAT_R16G16B16A16_UNORM, 8, false, false },
	{ DRM_FORMAT_ABGR2101010, VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_FORMAT_A2B10G10R10_UNORM_PACK32, 4, true, false },
	{ DRM_FORMAT_XBGR2101010, VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_FORMAT_A2B10G10R10_UNORM_PACK32, 4, false, false },
	{ DRM_FORMAT_ARGB2101010, VK_FORMAT_A2R10G10B10_UNORM_PACK32, VK_FORMAT_A2R10G10B10_UNORM_PACK32, 4, true, false },
	{ DRM_FORMAT_XRGB2101010, VK_FORMAT_A2R10G10B10_UNORM_PACK32, VK_FORMAT_A2R10G10B10_UNORM_PACK32, 4, false, false },

	{ DRM_FORMAT_R8, VK_FORMAT_R8_UNORM, VK_FORMAT_R8_UNORM, 1, false, true },
	{ DRM_FORMAT_R16, VK_FORMAT_R16_UNORM, VK_FORMAT_R16_UNORM, 2, false, true },
	{ DRM_FORMAT_GR88, VK_FORMAT_R8G8_UNORM, VK_FORMAT_R8G8_UNORM, 2, false, true },
	{ DRM_FORMAT_GR1616, VK_FORMAT_R16G16_UNORM, VK_FORMAT_R16G16_UNORM, 4, false, true },
	{ DRM_FORMAT_ABGR32323232F, VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R32G32B32A32_SFLOAT, 16,true, true },
	{ DRM_FORMAT_R16F, VK_FORMAT_R16_SFLOAT, VK_FORMAT_R16_SFLOAT, 2, false, true },
	{ DRM_FORMAT_R32F, VK_FORMAT_R32_SFLOAT, VK_FORMAT_R32_SFLOAT, 4, false, true },
	{ DRM_FORMAT_INVALID, VK_FORMAT_UNDEFINED, VK_FORMAT_UNDEFINED, false, true },
};

uint32_t VulkanFormatToDRM( VkFormat vkFormat, std::optional<bool> obHasAlphaOverride )
{
	for ( int i = 0; s_DRMVKFormatTable[i].vkFormat != VK_FORMAT_UNDEFINED; i++ )
	{
		if ( ( s_DRMVKFormatTable[i].vkFormat == vkFormat || s_DRMVKFormatTable[i].vkFormatSrgb == vkFormat ) && ( !obHasAlphaOverride || s_DRMVKFormatTable[i].bHasAlpha == *obHasAlphaOverride ) )
		{
			return s_DRMVKFormatTable[i].DRMFormat;
		}
	}
	
	return DRM_FORMAT_INVALID;
}

VkFormat DRMFormatToVulkan( uint32_t nDRMFormat, bool bSrgb )
{
	for ( int i = 0; s_DRMVKFormatTable[i].vkFormat != VK_FORMAT_UNDEFINED; i++ )
	{
		if ( s_DRMVKFormatTable[i].DRMFormat == nDRMFormat )
		{
			return bSrgb ? s_DRMVKFormatTable[i].vkFormatSrgb : s_DRMVKFormatTable[i].vkFormat;
		}
	}
	
	return VK_FORMAT_UNDEFINED;
}

bool DRMFormatHasAlpha( uint32_t nDRMFormat )
{
	for ( int i = 0; s_DRMVKFormatTable[i].vkFormat != VK_FORMAT_UNDEFINED; i++ )
	{
		if ( s_DRMVKFormatTable[i].DRMFormat == nDRMFormat )
		{
			return s_DRMVKFormatTable[i].bHasAlpha;
		}
	}
	
	return false;
}

uint32_t DRMFormatGetBPP( uint32_t nDRMFormat )
{
	for ( int i = 0; s_DRMVKFormatTable[i].vkFormat != VK_FORMAT_UNDEFINED; i++ )
	{
		if ( s_DRMVKFormatTable[i].DRMFormat == nDRMFormat )
		{
			return s_DRMVKFormatTable[i].bpp;
		}
	}

	return false;
}

bool CVulkanDevice::BInit(VkInstance instance, VkSurfaceKHR surface)
{
	assert(instance);
	assert(!m_bInitialized);

	g_output.surface = surface;

	m_instance = instance;
	#define VK_FUNC(x) vk.x = (PFN_vk##x) g_pfn_vkGetInstanceProcAddr(instance, "vk"#x);
	VULKAN_INSTANCE_FUNCTIONS
	#undef VK_FUNC

	if (!selectPhysDev(surface))
		return false;
	if (!createDevice())
		return false;
	if (!createLayouts())
		return false;
	if (!createPools())
		return false;
	if (!createShaders())
		return false;
	if (!createScratchResources())
		return false;

	m_bInitialized = true;

	if ( !isV3dv() )
	{
		m_pipelineThread = std::jthread([this](std::stop_token st){compileAllPipelines(st);});
	}

	g_reshadeManager.init(this);

	return true;
}

extern bool env_to_bool(const char *env);

bool CVulkanDevice::selectPhysDev(VkSurfaceKHR surface)
{
	uint32_t deviceCount = 0;
	vk.EnumeratePhysicalDevices(instance(), &deviceCount, nullptr);
	std::vector<VkPhysicalDevice> physDevs(deviceCount);
	vk.EnumeratePhysicalDevices(instance(), &deviceCount, physDevs.data());
	if (deviceCount < physDevs.size())
		physDevs.resize(deviceCount);

	bool bTryComputeOnly = true;

	// In theory vkBasalt might want to filter out compute-only queue families to force our hand here
	const char *pchEnableVkBasalt = getenv( "ENABLE_VKBASALT" );
	if ( pchEnableVkBasalt != nullptr && pchEnableVkBasalt[0] == '1' )
	{
		bTryComputeOnly = false;
	}

	for (auto cphysDev : physDevs)
	{
		VkPhysicalDeviceProperties deviceProperties;
		vk.GetPhysicalDeviceProperties(cphysDev, &deviceProperties);

		if (deviceProperties.apiVersion < VK_API_VERSION_1_2)
			continue;

		uint32_t queueFamilyCount = 0;
		vk.GetPhysicalDeviceQueueFamilyProperties(cphysDev, &queueFamilyCount, nullptr);
		std::vector<VkQueueFamilyProperties> queueFamilyProperties(queueFamilyCount);
		vk.GetPhysicalDeviceQueueFamilyProperties(cphysDev, &queueFamilyCount, queueFamilyProperties.data());

		uint32_t generalIndex = ~0u;
		uint32_t computeOnlyIndex = ~0u;
		for (uint32_t i = 0; i < queueFamilyCount; ++i) {
			const VkQueueFlags generalBits = VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT;
			if ((queueFamilyProperties[i].queueFlags & generalBits) == generalBits )
				generalIndex = std::min(generalIndex, i);
			else if (bTryComputeOnly && queueFamilyProperties[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
				computeOnlyIndex = std::min(computeOnlyIndex, i);
		}

		if (generalIndex != ~0u || computeOnlyIndex != ~0u)
		{
			// Select the device if it's the first one or the preferred one
			if (!m_physDev ||
			    (g_preferVendorID == deviceProperties.vendorID && g_preferDeviceID == deviceProperties.deviceID))
			{
				// if we have a surface, check that the queue family can actually present on it
				if (surface) {
					VkBool32 canPresent = false;
					vk.GetPhysicalDeviceSurfaceSupportKHR( cphysDev, generalIndex, surface, &canPresent );
					if ( !canPresent )
					{
						vk_log.infof( "physical device %04x:%04x queue doesn't support presenting on our surface, testing next one..", deviceProperties.vendorID, deviceProperties.deviceID );
						continue;
					}
					if (computeOnlyIndex != ~0u)
					{
						vk.GetPhysicalDeviceSurfaceSupportKHR( cphysDev, computeOnlyIndex, surface, &canPresent );
						if ( !canPresent )
						{
							vk_log.infof( "physical device %04x:%04x compute queue doesn't support presenting on our surface, using graphics queue", deviceProperties.vendorID, deviceProperties.deviceID );
							computeOnlyIndex = ~0u;
						}
					}
				}

				m_queueFamily = computeOnlyIndex == ~0u ? generalIndex : computeOnlyIndex;
				m_generalQueueFamily = generalIndex;
				m_physDev = cphysDev;

				/* When Intel uses compute-only queue for Gamescope composition, some games
				 * experience performance loss. Using the general queue alleviates the issue
				 * for now.
				 * See: https://gitlab.freedesktop.org/drm/xe/kernel/-/issues/4452
				 *
				 * TODO: Remove vendorID check for Intel once issue is resolved.
				 */
				if (deviceProperties.vendorID == 0x8086) /* Intel */
				{
					vk_log.infof("Intel device detected, forcing general queue family instead of compute-only queue");
					m_queueFamily = generalIndex;
				}
				else if ( GetBackend()->UsesBackendAllocatedScanout() || env_to_bool( getenv( "GAMESCOPE_FORCE_GENERAL_QUEUE" ) ) )
					m_queueFamily = generalIndex;
			}
		}
	}

	if (!m_physDev)
	{
		vk_log.errorf("failed to find physical device");
		return false;
	}

	VkPhysicalDeviceDriverProperties driverProps = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES,
	};
	VkPhysicalDeviceProperties2 props2 = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
		.pNext = &driverProps,
	};
	vk.GetPhysicalDeviceProperties2( m_physDev, &props2 );
	const VkPhysicalDeviceProperties &props = props2.properties;
	m_uVendorID = props.vendorID;
	m_driverID = driverProps.driverID;
	m_uUniformBufferAlignment = std::max<uint32_t>(16, props.limits.minUniformBufferOffsetAlignment);
	vk_log.infof( "selecting physical device '%s': queue family %x (general queue family %x)", props.deviceName, m_queueFamily, m_generalQueueFamily );

	// Venus currently aborts when importing DMA-BUFs from the guest's virtio-gpu
	// device. Don't advertise the zero-copy client path there, so Wayland and
	// Xwayland clients use wl_shm and Gamescope's existing upload path instead.
	// Keep DMA-BUFs enabled for native drivers, where this is a critical fast path.
	if ( driverProps.driverID == VK_DRIVER_ID_MESA_VENUS )
	{
		m_bSupportsClientDmabufs = false;
		vk_log.infof( "Venus driver detected, disabling client DMA-BUF imports" );
	}

	return true;
}

bool CVulkanDevice::createDevice()
{
	uint32_t supportedExtensionCount;
	vk.EnumerateDeviceExtensionProperties( physDev(), NULL, &supportedExtensionCount, NULL );

	m_supportedExts.resize(supportedExtensionCount);
	vk.EnumerateDeviceExtensionProperties( physDev(), NULL, &supportedExtensionCount, m_supportedExts.data() );

	if ( !GetBackend()->ValidPhysicalDevice( physDev() ) ) {
		vk_log.errorf( "not a valid physical device" );
		return false;
	}

	vk.GetPhysicalDeviceMemoryProperties( physDev(), &m_memoryProperties );

	bool hasDrmProps = vulkan_has_drm_props();
	bool supportsForeignQueue = false;
	bool supportsHDRMetadata = false;
	for (const auto& ext : m_supportedExts) {
		if ( strcmp(ext.extensionName, VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME) == 0 )
			m_bSupportsModifiers = true;

		if ( strcmp(ext.extensionName, VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME) == 0 )
			supportsForeignQueue = true;

		if ( strcmp(ext.extensionName, VK_EXT_HDR_METADATA_EXTENSION_NAME) == 0 )
			supportsHDRMetadata = true;
	}

	vk_log.infof( "physical device %s DRM format modifiers", m_bSupportsModifiers ? "supports" : "does not support" );

	if ( !hasDrmProps ) {
		// This could happen when e.g. running the lavapipe driver
		// (without an actual physical device)
		vk_log.warnf( "physical device doesn't support VK_EXT_physical_device_drm" );
	} else {
#if HAVE_DRM
		VkPhysicalDeviceDrmPropertiesEXT drmProps = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRM_PROPERTIES_EXT,
		};
		VkPhysicalDeviceProperties2 props2 = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
			.pNext = &drmProps,
		};
		vk.GetPhysicalDeviceProperties2( physDev(), &props2 );

		if ( !GetBackend()->UsesVulkanSwapchain() && !drmProps.hasPrimary ) {
			vk_log.errorf( "physical device has no primary node" );
			return false;
		}
		if ( !drmProps.hasRender ) {
			vk_log.errorf( "physical device has no render node" );
			return false;
		}

		dev_t renderDevId = makedev( drmProps.renderMajor, drmProps.renderMinor );
		drmDevice *drmDev = nullptr;
		if (drmGetDeviceFromDevId(renderDevId, 0, &drmDev) != 0) {
			vk_log.errorf( "drmGetDeviceFromDevId() failed" );
			return false;
		}
		assert(drmDev->available_nodes & (1 << DRM_NODE_RENDER));
		const char *drmRenderName = drmDev->nodes[DRM_NODE_RENDER];

		m_drmRendererFd = open( drmRenderName, O_RDWR | O_CLOEXEC );
		drmFreeDevice(&drmDev);
		if ( m_drmRendererFd < 0 ) {
			vk_log.errorf_errno( "failed to open DRM render node" );
			return false;
		}

		if ( drmProps.hasPrimary ) {
			m_bHasDrmPrimaryDevId = true;
			m_drmPrimaryDevId = makedev( drmProps.primaryMajor, drmProps.primaryMinor );
		}
#else
		vk_log.warnf( "built without DRM support" );
#endif
	}

	if ( m_bSupportsModifiers && !supportsForeignQueue ) {
		vk_log.infof( "The vulkan driver does not support foreign queues,"
		              " disabling modifier support.");
		m_bSupportsModifiers = false;
	}

	{
		VkPhysicalDeviceVulkan13Features vulkan13Features = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
		};
		VkPhysicalDeviceVulkan12Features vulkan12Features = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
			.pNext = &vulkan13Features,
		};
		VkPhysicalDeviceFeatures2 features2 = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
			.pNext = &vulkan12Features,
		};
		vk.GetPhysicalDeviceFeatures2( physDev(), &features2 );

		m_bSupportsFp16 = vulkan12Features.shaderFloat16 && features2.features.shaderInt16;
		VkPhysicalDeviceShaderIntegerDotProductProperties dotProductProperties = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES,
		};
		VkPhysicalDeviceProperties2 properties2 = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
			.pNext = &dotProductProperties,
		};
		vk.GetPhysicalDeviceProperties2( physDev(), &properties2 );
		m_bSupportsIntegerDotProduct = vulkan12Features.shaderInt8 &&
			vulkan13Features.shaderIntegerDotProduct &&
			properties2.properties.apiVersion >= VK_API_VERSION_1_3 &&
			dotProductProperties.integerDotProduct8BitUnsignedAccelerated;
		m_bSupportsStorageImageReadWithoutFormat = features2.features.shaderStorageImageReadWithoutFormat;
		m_bSupportsStorageImageWriteWithoutFormat = features2.features.shaderStorageImageWriteWithoutFormat;
	}

	float queuePriorities = 1.0f;

	VkDeviceQueueGlobalPriorityCreateInfoEXT queueCreateInfoEXT = {
		.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO_EXT,
		.pNext = nullptr,
		.globalPriority = VK_QUEUE_GLOBAL_PRIORITY_REALTIME_EXT
	};

	VkDeviceQueueCreateInfo queueCreateInfos[2] = 
	{
		{
			.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
			.pNext = gamescope::Process::HasCapSysNice() ? &queueCreateInfoEXT : nullptr,
			.queueFamilyIndex = m_queueFamily,
			.queueCount = 1,
			.pQueuePriorities = &queuePriorities
		},
		{
			.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
			.pNext = gamescope::Process::HasCapSysNice() ? &queueCreateInfoEXT : nullptr,
			.queueFamilyIndex = m_generalQueueFamily,
			.queueCount = 1,
			.pQueuePriorities = &queuePriorities
		},
	};

	std::vector< const char * > enabledExtensions;

	if ( GetBackend()->UsesVulkanSwapchain() )
	{
		enabledExtensions.push_back( VK_KHR_SWAPCHAIN_EXTENSION_NAME );
		enabledExtensions.push_back( VK_KHR_SWAPCHAIN_MUTABLE_FORMAT_EXTENSION_NAME );

		enabledExtensions.push_back( VK_KHR_PRESENT_ID_EXTENSION_NAME );
		enabledExtensions.push_back( VK_KHR_PRESENT_WAIT_EXTENSION_NAME );

		if ( supportsHDRMetadata )
			enabledExtensions.push_back( VK_EXT_HDR_METADATA_EXTENSION_NAME );
	}

	if ( m_bSupportsModifiers )
	{
		enabledExtensions.push_back( VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME );
		enabledExtensions.push_back( VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME );
	}

	enabledExtensions.push_back( VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME );
	enabledExtensions.push_back( VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME );

	enabledExtensions.push_back( VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME );

	enabledExtensions.push_back( VK_EXT_ROBUSTNESS_2_EXTENSION_NAME );
#if 0
	enabledExtensions.push_back( VK_KHR_MAINTENANCE_5_EXTENSION_NAME );
#endif

	for ( auto& extension : GetBackend()->GetDeviceExtensions( physDev() ) )
		enabledExtensions.push_back( extension );

	uint32_t devExtPropCount = 0;
	vk.EnumerateDeviceExtensionProperties( physDev(), nullptr, &devExtPropCount, nullptr );
	std::vector<VkExtensionProperties> devExtProp( devExtPropCount );
	vk.EnumerateDeviceExtensionProperties( physDev(), nullptr, &devExtPropCount, devExtProp.data() );
	bool anyMissing = false;
	for ( auto& requiredExt : enabledExtensions ) {
		bool extFound = false;
		for ( auto & availableExt : devExtProp ) {
			if ( strcmp( requiredExt, availableExt.extensionName ) == 0 ) {
				extFound = true;
				break;
			}
		}
		if ( !extFound ) {
			vk_log.errorf( "Missing required extension: %s", requiredExt );
			anyMissing = true;
		}
	}
	if ( anyMissing )
		return false;

#if 0
	VkPhysicalDeviceMaintenance5FeaturesKHR maintenance5 = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES_KHR,
		.maintenance5 = VK_TRUE,
	};
#endif

	VkPhysicalDeviceVulkan13Features features13 = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
#if 0
		.pNext = &maintenance5,
#endif
		.dynamicRendering = VK_TRUE,
		.shaderIntegerDotProduct = m_bSupportsIntegerDotProduct,
	};

	VkPhysicalDevicePresentWaitFeaturesKHR presentWaitFeatures = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_WAIT_FEATURES_KHR,
		.pNext = &features13,
		.presentWait = VK_TRUE,
	};

	VkPhysicalDevicePresentIdFeaturesKHR presentIdFeatures = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR,
		.pNext = &presentWaitFeatures,
		.presentId = VK_TRUE,
	};

	VkPhysicalDeviceFeatures2 features2 = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
		.pNext = &presentIdFeatures,
	};
	features2.features.shaderInt16 = m_bSupportsFp16;
	features2.features.shaderStorageImageReadWithoutFormat = m_bSupportsStorageImageReadWithoutFormat;
	features2.features.shaderStorageImageWriteWithoutFormat = m_bSupportsStorageImageWriteWithoutFormat;

	VkDeviceCreateInfo deviceCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		.pNext = &features2,
		.queueCreateInfoCount = m_queueFamily == m_generalQueueFamily ? 1u : 2u,
		.pQueueCreateInfos = queueCreateInfos,
		.enabledExtensionCount = (uint32_t)enabledExtensions.size(),
		.ppEnabledExtensionNames = enabledExtensions.data(),
	};

	VkPhysicalDeviceVulkan12Features vulkan12Features = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
		.pNext = std::exchange(features2.pNext, &vulkan12Features),
		.shaderFloat16 = m_bSupportsFp16,
		.shaderInt8 = m_bSupportsIntegerDotProduct,
		.scalarBlockLayout = VK_TRUE,
		.timelineSemaphore = VK_TRUE,
	};

	VkPhysicalDeviceSamplerYcbcrConversionFeatures ycbcrFeatures = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES,
		.pNext = std::exchange(features2.pNext, &ycbcrFeatures),
		.samplerYcbcrConversion = VK_TRUE,
	};

	VkPhysicalDeviceRobustness2FeaturesEXT robustness2Features = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT,
		.pNext = std::exchange(features2.pNext, &robustness2Features),
		.nullDescriptor = VK_TRUE,
	};

	VkResult res = vk.CreateDevice(physDev(), &deviceCreateInfo, nullptr, &m_device);
	if ( res == VK_ERROR_NOT_PERMITTED_KHR && gamescope::Process::HasCapSysNice() )
	{
		fprintf(stderr, "vkCreateDevice failed with a high-priority queue (general + compute). Falling back to regular priority (general).\n");
		queueCreateInfos[1].pNext = nullptr;
		res = vk.CreateDevice(physDev(), &deviceCreateInfo, nullptr, &m_device);


		if ( res == VK_ERROR_NOT_PERMITTED_KHR && gamescope::Process::HasCapSysNice() )
		{
			fprintf(stderr, "vkCreateDevice failed with a high-priority queue (compute). Falling back to regular priority (all).\n");
			queueCreateInfos[0].pNext = nullptr;
			res = vk.CreateDevice(physDev(), &deviceCreateInfo, nullptr, &m_device);
		}
	}

	if ( res != VK_SUCCESS )
	{
		vk_errorf( res, "vkCreateDevice failed" );
		return false;
	}

	#define VK_FUNC(x) vk.x = (PFN_vk##x) vk.GetDeviceProcAddr(device(), "vk"#x);
	VULKAN_DEVICE_FUNCTIONS
	#undef VK_FUNC

	vk.GetDeviceQueue(device(), m_queueFamily, 0, &m_queue);
	if ( m_queueFamily == m_generalQueueFamily )
		m_generalQueue = m_queue;
	else
		vk.GetDeviceQueue(device(), m_generalQueueFamily, 0, &m_generalQueue);

	return true;
}

static VkSamplerYcbcrModelConversion colorspaceToYCBCRModel( EStreamColorspace colorspace )
{
	switch (colorspace)
	{
		default:
		case k_EStreamColorspace_Unknown:
			return VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_709;

		case k_EStreamColorspace_BT601:
		case k_EStreamColorspace_BT601_Full:
			return VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_601;

		case k_EStreamColorspace_BT709:
		case k_EStreamColorspace_BT709_Full:
			return VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_709;
	}
}

static VkSamplerYcbcrRange colorspaceToYCBCRRange( EStreamColorspace colorspace )
{
	switch (colorspace)
	{
		default:
		case k_EStreamColorspace_Unknown:
			return VK_SAMPLER_YCBCR_RANGE_ITU_FULL;

		case k_EStreamColorspace_BT709:
		case k_EStreamColorspace_BT601:
			return VK_SAMPLER_YCBCR_RANGE_ITU_NARROW;

		case k_EStreamColorspace_BT601_Full:
		case k_EStreamColorspace_BT709_Full:
			return VK_SAMPLER_YCBCR_RANGE_ITU_FULL;
	}
}

bool CVulkanDevice::createLayouts()
{
	VkFormatProperties nv12Properties;
	vk.GetPhysicalDeviceFormatProperties(physDev(), VK_FORMAT_G8_B8R8_2PLANE_420_UNORM, &nv12Properties);
	bool cosited = nv12Properties.optimalTilingFeatures & VK_FORMAT_FEATURE_COSITED_CHROMA_SAMPLES_BIT;

	VkSamplerYcbcrConversionCreateInfo ycbcrSamplerConversionCreateInfo = 
	{
		.sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO,
		.format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
		.ycbcrModel = colorspaceToYCBCRModel( g_ForcedNV12ColorSpace ),
		.ycbcrRange = colorspaceToYCBCRRange( g_ForcedNV12ColorSpace ),
		.xChromaOffset = cosited ? VK_CHROMA_LOCATION_COSITED_EVEN : VK_CHROMA_LOCATION_MIDPOINT,
		.yChromaOffset = cosited ? VK_CHROMA_LOCATION_COSITED_EVEN : VK_CHROMA_LOCATION_MIDPOINT,
		.chromaFilter = VK_FILTER_LINEAR,
		.forceExplicitReconstruction = VK_FALSE,
	};

	vk.CreateSamplerYcbcrConversion( device(), &ycbcrSamplerConversionCreateInfo, nullptr, &m_ycbcrConversion );

	VkSamplerYcbcrConversionInfo ycbcrSamplerConversionInfo = {
		.sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO,
		.conversion = m_ycbcrConversion,
	};

	VkSamplerCreateInfo ycbcrSamplerInfo = {
		.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.pNext = &ycbcrSamplerConversionInfo,
		.magFilter = VK_FILTER_LINEAR,
		.minFilter = VK_FILTER_LINEAR,
		.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK,
	};
	
	vk.CreateSampler( device(), &ycbcrSamplerInfo, nullptr, &m_ycbcrSampler );

	// Create an array of our ycbcrSampler to fill up
	std::array<VkSampler, VKR_SAMPLER_SLOTS> ycbcrSamplers;
	for (auto& sampler : ycbcrSamplers)
		sampler = m_ycbcrSampler;

	std::array<VkDescriptorSetLayoutBinding, 8 > layoutBindings = {
		VkDescriptorSetLayoutBinding {
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		VkDescriptorSetLayoutBinding {
			.binding = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		VkDescriptorSetLayoutBinding {
			.binding = 2,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		VkDescriptorSetLayoutBinding {
			.binding = 3,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = VKR_SAMPLER_SLOTS,
			.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		VkDescriptorSetLayoutBinding {
			.binding = 4,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = VKR_SAMPLER_SLOTS,
			.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
			.pImmutableSamplers = ycbcrSamplers.data(),
		},
		VkDescriptorSetLayoutBinding {
			.binding = 5,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = VKR_LUT3D_COUNT,
			.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		VkDescriptorSetLayoutBinding {
			.binding = 6,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = VKR_LUT3D_COUNT,
			.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		VkDescriptorSetLayoutBinding {
			.binding = 7, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
	};

	VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo =
	{
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = (uint32_t)layoutBindings.size(),
		.pBindings = layoutBindings.data()
	};

	VkResult res = vk.CreateDescriptorSetLayout(device(), &descriptorSetLayoutCreateInfo, 0, &m_descriptorSetLayout);
	if ( res != VK_SUCCESS )
	{
		vk_errorf( res, "vkCreateDescriptorSetLayout failed" );
		return false;
	}

	VkPushConstantRange ranges[] = {
		{VK_SHADER_STAGE_FRAGMENT_BIT, 0, 5 * sizeof(uint32_t)},
		{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t)},
	};
	VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &m_descriptorSetLayout,
		.pushConstantRangeCount = 2, .pPushConstantRanges = ranges,
	};
	
	res = vk.CreatePipelineLayout(device(), &pipelineLayoutCreateInfo, nullptr, &m_pipelineLayout);
	if ( res != VK_SUCCESS )
	{
		vk_errorf( res, "vkCreatePipelineLayout failed" );
		return false;
	}

	return true;
}

bool CVulkanDevice::createPools()
{
	VkCommandPoolCreateInfo commandPoolCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
		.queueFamilyIndex = m_queueFamily,
	};

	VkResult res = vk.CreateCommandPool(device(), &commandPoolCreateInfo, nullptr, &m_commandPool);
	if ( res != VK_SUCCESS )
	{
		vk_errorf( res, "vkCreateCommandPool failed" );
		return false;
	}

	VkCommandPoolCreateInfo generalCommandPoolCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
		.queueFamilyIndex = m_generalQueueFamily,
	};

	res = vk.CreateCommandPool(device(), &generalCommandPoolCreateInfo, nullptr, &m_generalCommandPool);
	if ( res != VK_SUCCESS )
	{
		vk_errorf( res, "vkCreateCommandPool failed" );
		return false;
	}

	VkPhysicalDeviceImageFormatInfo2 imageFormatInfo = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
		.format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
		.type = VK_IMAGE_TYPE_2D,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = VK_IMAGE_USAGE_SAMPLED_BIT,
	};

	VkSamplerYcbcrConversionImageFormatProperties ycbcrProps = {
		.sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_IMAGE_FORMAT_PROPERTIES,
	};

	VkImageFormatProperties2 imageFormatProps = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
		.pNext = &ycbcrProps,
	};

	res = vk.GetPhysicalDeviceImageFormatProperties2( physDev(), &imageFormatInfo, &imageFormatProps );

	VkDescriptorPoolSize poolSizes[4] {
		{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, uint32_t(m_descriptorSets.size()) },
		{
			VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			uint32_t(m_descriptorSets.size()),
		},
		{
			VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
			uint32_t(m_descriptorSets.size()) * 2,
		},
		{
			VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			uint32_t(m_descriptorSets.size()) * (((ycbcrProps.combinedImageSamplerDescriptorCount + 1) * VKR_SAMPLER_SLOTS) + (2 * VKR_LUT3D_COUNT)),
		},
	};
	if ( m_driverID == VK_DRIVER_ID_MESA_V3DV )
	{
		// V3DV applies the YCbCr plane stride to unrelated bindings too.
		for ( auto &poolSize : poolSizes )
			poolSize.descriptorCount *= 2;
		vk_log.infof( "V3DV: reserving twice the descriptor pool capacity" );
	}

	VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = uint32_t(m_descriptorSets.size()),
		.poolSizeCount = sizeof(poolSizes) / sizeof(poolSizes[0]),
		.pPoolSizes = poolSizes,
	};
	
	res = vk.CreateDescriptorPool(device(), &descriptorPoolCreateInfo, nullptr, &m_descriptorPool);
	if ( res != VK_SUCCESS )
	{
		vk_errorf( res, "vkCreateDescriptorPool failed" );
		return false;
	}

	return true;
}

bool CVulkanDevice::createShaders()
{
	struct ShaderInfo_t
	{
		const uint32_t* spirv;
		uint32_t size;
	};

	std::array<ShaderInfo_t, SHADER_TYPE_COUNT> shaderInfos;
#define SHADER(type, array) shaderInfos[SHADER_TYPE_##type] = m_bSupportsStorageImageWriteWithoutFormat ? ShaderInfo_t{array##_formatless, sizeof(array##_formatless)} : ShaderInfo_t{array, sizeof(array)}
	SHADER(BLIT, cs_composite_blit);
	SHADER(BLIT_RGBA16F, cs_composite_blit_rgba16f);
	SHADER(BLIT_RGB10A2, cs_composite_blit_rgb10a2);
	SHADER(BLUR, cs_composite_blur);
	SHADER(BLUR_RGB10A2, cs_composite_blur_rgb10a2);
	SHADER(BLUR_RGBA16F, cs_composite_blur_rgba16f);
	SHADER(BLUR_COND, cs_composite_blur_cond);
	SHADER(BLUR_COND_RGB10A2, cs_composite_blur_cond_rgb10a2);
	SHADER(BLUR_COND_RGBA16F, cs_composite_blur_cond_rgba16f);
	SHADER(BLUR_FIRST_PASS, cs_gaussian_blur_horizontal);
	SHADER(BLUR_FIRST_PASS_RGB10A2, cs_gaussian_blur_horizontal_rgb10a2);
	SHADER(BLUR_FIRST_PASS_RGBA16F, cs_gaussian_blur_horizontal_rgba16f);
	if ( m_uVendorID == 0x5143 ) /* Qualcomm */
	{
		SHADER(RCAS, cs_composite_rcas_1px);
	}
	else
	{
		SHADER(RCAS, cs_composite_rcas);
	}
	if ( m_uVendorID == 0x5143 ) /* Qualcomm */
	{
		SHADER(RCAS_RGBA16F, cs_composite_rcas_rgba16f_1px);
	}
	else
	{
		SHADER(RCAS_RGBA16F, cs_composite_rcas_rgba16f);
	}
	if ( m_uVendorID == 0x5143 ) /* Qualcomm */
	{
		SHADER(RCAS_RGB10A2, cs_composite_rcas_rgb10a2_1px);
	}
	else
	{
		SHADER(RCAS_RGB10A2, cs_composite_rcas_rgb10a2);
	}
	if (m_bSupportsFp16)
	{
		if ( m_uVendorID == 0x5143 ) /* Qualcomm */
		{
			SHADER(RCAS_FP16, cs_composite_rcas_fp16_1px);
		}
		else
		{
			SHADER(RCAS_FP16, cs_composite_rcas_fp16);
		}
		SHADER(EASU, cs_easu_fp16);
		SHADER(EASU_RGBA16F, cs_easu_fp16_rgba16f);
		SHADER(NIS, cs_nis_fp16);
		SHADER(NIS_RGB10A2, cs_nis_fp16_rgb10a2);
		SHADER(NIS_HDR_LINEAR, cs_nis_hdr_linear_fp16);
	}
	else
	{
		SHADER(RCAS_FP16, cs_composite_rcas);
		SHADER(EASU, cs_easu);
		SHADER(EASU_RGBA16F, cs_easu_rgba16f);
		SHADER(NIS, cs_nis);
		SHADER(NIS_RGB10A2, cs_nis_rgb10a2);
		SHADER(NIS_HDR_LINEAR, cs_nis_hdr_linear);
	}
	SHADER(RGB_TO_NV12, cs_rgb_to_nv12);

	SHADER(SGSR, cs_sgsr);
#undef SHADER

	for (uint32_t i = 0; i < shaderInfos.size(); i++)
	{
		VkShaderModuleCreateInfo shaderCreateInfo = {
			.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
			.codeSize = shaderInfos[i].size,
			.pCode = shaderInfos[i].spirv,
		};

		VkResult res = vk.CreateShaderModule(device(), &shaderCreateInfo, nullptr, &m_shaderModules[i]);
		if ( res != VK_SUCCESS )
		{
			vk_errorf( res, "vkCreateShaderModule failed" );
			return false;
		}
	}

	for ( auto [words, size, module] : { std::tuple{ vs_composite, sizeof(vs_composite), &m_blitVertex },
	                                  std::tuple{ fs_composite, sizeof(fs_composite), &m_blitFragment } } )
	{
		VkShaderModuleCreateInfo info = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = size, .pCode = words };
		if ( vk.CreateShaderModule(device(), &info, nullptr, module) != VK_SUCCESS )
			return false;
	}

	return true;
}

bool CVulkanDevice::createScratchResources()
{
	std::vector<VkDescriptorSetLayout> descriptorSetLayouts(m_descriptorSets.size(), m_descriptorSetLayout);
	
	VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = m_descriptorPool,
		.descriptorSetCount = (uint32_t)descriptorSetLayouts.size(),
		.pSetLayouts = descriptorSetLayouts.data(),
	};
	
	VkResult res = vk.AllocateDescriptorSets(device(), &descriptorSetAllocateInfo, m_descriptorSets.data());
	if ( res != VK_SUCCESS )
	{
		vk_log.errorf( "vkAllocateDescriptorSets failed" );
		return false;
	}
	m_freeDescriptorSets.assign( m_descriptorSets.begin(), m_descriptorSets.end() );

	// Make and map upload buffer
	
	VkBufferCreateInfo bufferCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = upload_buffer_size,
		.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
	};

	res = vk.CreateBuffer( device(), &bufferCreateInfo, nullptr, &m_uploadBuffer );
	if ( res != VK_SUCCESS )
	{
		vk_errorf( res, "vkCreateBuffer failed" );
		return false;
	}
	
	VkMemoryRequirements memRequirements;
	vk.GetBufferMemoryRequirements(device(), m_uploadBuffer, &memRequirements);
	
	uint32_t memTypeIndex =  findMemoryType(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT|VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, memRequirements.memoryTypeBits );
	if ( memTypeIndex == ~0u )
	{
		vk_log.errorf( "findMemoryType failed" );
		return false;
	}
	
	VkMemoryAllocateInfo allocInfo = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize = memRequirements.size,
		.memoryTypeIndex = memTypeIndex,
	};
	
	vk.AllocateMemory( device(), &allocInfo, nullptr, &m_uploadBufferMemory);
	
	vk.BindBufferMemory( device(), m_uploadBuffer, m_uploadBufferMemory, 0 );

	res = vk.MapMemory( device(), m_uploadBufferMemory, 0, VK_WHOLE_SIZE, 0, (void**)&m_uploadBufferData );
	if ( res != VK_SUCCESS )
	{
		vk_errorf( res, "vkMapMemory failed" );
		return false;
	}

	VkSemaphoreTypeCreateInfo timelineCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
		.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
	};

	VkSemaphoreCreateInfo semCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
		.pNext = &timelineCreateInfo,
	};

	res = vk.CreateSemaphore( device(), &semCreateInfo, NULL, &m_scratchTimelineSemaphore );
	if ( res != VK_SUCCESS )
	{
		vk_errorf( res, "vkCreateSemaphore failed" );
		return false;
	}

	return true;
}

VkSampler CVulkanDevice::sampler( SamplerState key )
{
	if ( m_samplerCache.count(key) != 0 )
		return m_samplerCache[key];

	VkSampler ret = VK_NULL_HANDLE;

	VkSamplerCreateInfo samplerCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.magFilter = key.bNearest ? VK_FILTER_NEAREST : VK_FILTER_LINEAR,
		.minFilter = key.bNearest ? VK_FILTER_NEAREST : VK_FILTER_LINEAR,
		.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
		.unnormalizedCoordinates = key.bUnnormalized,
	};

	vk.CreateSampler( device(), &samplerCreateInfo, nullptr, &ret );

	m_samplerCache[key] = ret;

	return ret;
}

VkPipeline CVulkanDevice::compilePipeline(uint32_t layerCount, uint32_t ycbcrMask, ShaderType type, uint32_t blur_layer_count, uint32_t composite_debug, uint32_t colorspace_mask, uint32_t output_eotf, bool itm_enable, bool fsr_simple_output, bool simple_blend, CompositeShaderConfig_t composite_config)
{
	const std::array<VkSpecializationMapEntry, 17> specializationEntries = {{
		{
			.constantID = 0,
			.offset     = sizeof(uint32_t) * 0,
			.size       = sizeof(uint32_t)
		},
		{
			.constantID = 1,
			.offset     = sizeof(uint32_t) * 1,
			.size       = sizeof(uint32_t)
		},
		{
			.constantID = 2,
			.offset     = sizeof(uint32_t) * 2,
			.size       = sizeof(uint32_t)
		},
		{
			.constantID = 3,
			.offset     = sizeof(uint32_t) * 3,
			.size       = sizeof(uint32_t)
		},
		{
			.constantID = 4,
			.offset     = sizeof(uint32_t) * 4,
			.size       = sizeof(uint32_t)
		},

		{
			.constantID = 5,
			.offset     = sizeof(uint32_t) * 5,
			.size       = sizeof(uint32_t)
		},

		{
			.constantID = 6,
			.offset     = sizeof(uint32_t) * 6,
			.size       = sizeof(uint32_t)
		},
		{
			.constantID = 7,
			.offset     = sizeof(uint32_t) * 7,
			.size       = sizeof(uint32_t)
		},
		{ .constantID = 8, .offset = sizeof(uint32_t) * 8, .size = sizeof(uint32_t) },
		{ .constantID = 9, .offset = sizeof(uint32_t) * 9, .size = sizeof(uint32_t) },
		{ .constantID = 10, .offset = sizeof(uint32_t) * 10, .size = sizeof(uint32_t) },
		{ .constantID = 11, .offset = sizeof(uint32_t) * 11, .size = sizeof(uint32_t) },
		{ .constantID = 12, .offset = sizeof(uint32_t) * 12, .size = sizeof(uint32_t) },
		{ .constantID = 13, .offset = sizeof(uint32_t) * 13, .size = sizeof(uint32_t) },
		{ .constantID = 14, .offset = sizeof(uint32_t) * 14, .size = sizeof(uint32_t) },
		{ .constantID = 15, .offset = sizeof(uint32_t) * 15, .size = sizeof(uint32_t) },
		{ .constantID = 16, .offset = sizeof(uint32_t) * 16, .size = sizeof(uint32_t) },
	}};

	struct {
		uint32_t layerCount;
		uint32_t ycbcrMask;
		uint32_t debug;
		uint32_t blur_layer_count;
		uint32_t colorspace_mask;
		uint32_t output_eotf;
		uint32_t fsr_simple_output;
		uint32_t itm_enable;
		uint32_t workgroup_x;
		uint32_t workgroup_y;
		uint32_t simple_blend;
		uint32_t filter_mask;
		uint32_t color_mgmt_mask;
		uint32_t sampling_mask;
		uint32_t opaque_output;
		uint32_t draw_layer;
		uint32_t linear_source;
	} specializationData = {
		.layerCount   = layerCount,
		.ycbcrMask    = ycbcrMask,
		.debug        = composite_debug,
		.blur_layer_count = blur_layer_count,
		.colorspace_mask = colorspace_mask,
		.output_eotf = output_eotf,
		.fsr_simple_output = fsr_simple_output,
		.itm_enable = itm_enable,
		.workgroup_x = fsr_simple_output && isV3dv() ? 256u : 8u,
		.workgroup_y = fsr_simple_output && isV3dv() ? 1u : 8u,
		.simple_blend = simple_blend,
		.filter_mask = composite_config.filterMask,
		.color_mgmt_mask = composite_config.colorMgmtMask,
		.sampling_mask = composite_config.samplingMask,
		.opaque_output = composite_config.opaqueOutput,
		.draw_layer = composite_config.drawLayer,
		.linear_source = composite_config.linearSource,
	};

	VkSpecializationInfo specializationInfo = {
		.mapEntryCount = uint32_t(specializationEntries.size()),
		.pMapEntries   = specializationEntries.data(),
		.dataSize      = sizeof(specializationData),
		.pData		   = &specializationData,
	};



	if ( composite_config.renderFormat != VK_FORMAT_UNDEFINED )
	{
		if (composite_config.drawLayer == 0)
			vk_log.debugf("Raster pipeline: %u layers, colorspace %#x, filters %#x, color management %#x, sampling %#x, target %u",
				layerCount, colorspace_mask, composite_config.filterMask, composite_config.colorMgmtMask, composite_config.samplingMask, composite_config.renderFormat);

		VkPipelineShaderStageCreateInfo stages[] = {
			{ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			  .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = m_blitVertex, .pName = "main" },
			{ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			  .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = m_blitFragment, .pName = "main",
			  .pSpecializationInfo = &specializationInfo },
		};
		VkPipelineVertexInputStateCreateInfo vertex = { .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
		VkPipelineInputAssemblyStateCreateInfo assembly = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST };
		VkPipelineViewportStateCreateInfo viewport = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1 };
		VkPipelineRasterizationStateCreateInfo raster = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			.polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE, .lineWidth = 1.0f };
		VkPipelineMultisampleStateCreateInfo samples = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT };
		VkPipelineColorBlendAttachmentState blend = {
			.blendEnable = composite_config.drawLayer != 0,
			.srcColorBlendFactor = VK_BLEND_FACTOR_ONE, .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
			.colorBlendOp = VK_BLEND_OP_ADD,
			.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE, .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
			.alphaBlendOp = VK_BLEND_OP_ADD,
			.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
		};
		VkPipelineColorBlendStateCreateInfo blending = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .attachmentCount = 1, .pAttachments = &blend };
		VkDynamicState dynamicStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
		VkPipelineDynamicStateCreateInfo dynamic = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, .dynamicStateCount = 2, .pDynamicStates = dynamicStates };
		VkPipelineRenderingCreateInfo rendering = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
			.colorAttachmentCount = 1, .pColorAttachmentFormats = &composite_config.renderFormat };
		VkGraphicsPipelineCreateInfo info = {
			.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &rendering,
			.stageCount = 2, .pStages = stages, .pVertexInputState = &vertex, .pInputAssemblyState = &assembly,
			.pViewportState = &viewport, .pRasterizationState = &raster, .pMultisampleState = &samples,
			.pColorBlendState = &blending, .pDynamicState = &dynamic, .layout = m_pipelineLayout,
		};
		VkPipeline result = VK_NULL_HANDLE;
		VkResult res = vk.CreateGraphicsPipelines(device(), VK_NULL_HANDLE, 1, &info, nullptr, &result);
		if (res != VK_SUCCESS)
			vk_errorf(res, "vkCreateGraphicsPipelines failed");
		return result;
	}

	VkComputePipelineCreateInfo computePipelineCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
		.stage = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_COMPUTE_BIT,
			.module = m_shaderModules[type],
			.pName = "main",
			.pSpecializationInfo = &specializationInfo
		},
		.layout = m_pipelineLayout,
	};

	VkPipeline result;

	VkResult res = vk.CreateComputePipelines(device(), VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &result);
	if (res != VK_SUCCESS) {
		vk_errorf( res, "vkCreateComputePipelines failed" );
		return VK_NULL_HANDLE;
	}

	return result;
}

void CVulkanDevice::compileAllPipelines(std::stop_token st)
{
	pthread_setname_np( pthread_self(), "gamescope-shdr" );

	std::array<PipelineInfo_t, SHADER_TYPE_COUNT> pipelineInfos;
#define SHADER(type, layer_count, max_ycbcr, blur_layers) pipelineInfos[SHADER_TYPE_##type] = {SHADER_TYPE_##type, layer_count, max_ycbcr, blur_layers}
	SHADER(BLIT, k_nMaxLayers, k_nMaxYcbcrMask_ToPreCompile, 1);
	SHADER(BLIT_RGBA16F, k_nMaxLayers, k_nMaxYcbcrMask_ToPreCompile, 1);
	SHADER(BLIT_RGB10A2, k_nMaxLayers, k_nMaxYcbcrMask_ToPreCompile, 1);
	SHADER(BLUR, k_nMaxLayers, k_nMaxYcbcrMask_ToPreCompile, k_nMaxBlurLayers);
	SHADER(BLUR_RGB10A2, k_nMaxLayers, k_nMaxYcbcrMask_ToPreCompile, k_nMaxBlurLayers);
	SHADER(BLUR_RGBA16F, k_nMaxLayers, k_nMaxYcbcrMask_ToPreCompile, k_nMaxBlurLayers);
	SHADER(BLUR_COND, k_nMaxLayers, k_nMaxYcbcrMask_ToPreCompile, k_nMaxBlurLayers);
	SHADER(BLUR_COND_RGB10A2, k_nMaxLayers, k_nMaxYcbcrMask_ToPreCompile, k_nMaxBlurLayers);
	SHADER(BLUR_COND_RGBA16F, k_nMaxLayers, k_nMaxYcbcrMask_ToPreCompile, k_nMaxBlurLayers);
	SHADER(BLUR_FIRST_PASS, 1, 2, 1);
	SHADER(BLUR_FIRST_PASS_RGB10A2, 1, 2, 1);
	SHADER(BLUR_FIRST_PASS_RGBA16F, 1, 2, 1);
	SHADER(RCAS, k_nMaxLayers, k_nMaxYcbcrMask_ToPreCompile, 1);
	SHADER(RCAS_FP16, k_nMaxLayers, k_nMaxYcbcrMask_ToPreCompile, 1);
	SHADER(RCAS_RGBA16F, k_nMaxLayers, k_nMaxYcbcrMask_ToPreCompile, 1);
	SHADER(RCAS_RGB10A2, k_nMaxLayers, k_nMaxYcbcrMask_ToPreCompile, 1);
	SHADER(EASU, 1, 1, 1);
	SHADER(EASU_RGBA16F, 1, 1, 1);
	SHADER(NIS, 1, 1, 1);
	SHADER(NIS_RGB10A2, 1, 1, 1);
	SHADER(NIS_HDR_LINEAR, 1, 1, 1);
	SHADER(RGB_TO_NV12, 1, 1, 1);

	SHADER(SGSR, 1, 1, 1);
#undef SHADER

	for (auto& info : pipelineInfos) {
		for (uint32_t layerCount = 1; layerCount <= info.layerCount; layerCount++) {
			for (uint32_t ycbcrMask = 0; ycbcrMask < info.ycbcrMask; ycbcrMask++) {
				for (uint32_t blur_layers = 1; blur_layers <= info.blurLayerCount; blur_layers++) {
					if (ycbcrMask >= (1u << (layerCount + 1)))
						continue;
					if (blur_layers > layerCount)
						continue;

					VkPipeline newPipeline = compilePipeline(layerCount, ycbcrMask, info.shaderType, blur_layers, info.compositeDebug, info.colorspaceMask, info.outputEOTF, info.itmEnable, info.fsrSimpleOutput);
					{
						std::lock_guard<std::mutex> lock(m_pipelineMutex);
						PipelineInfo_t key = {info.shaderType, layerCount, ycbcrMask, blur_layers, info.compositeDebug};
						if (st.stop_requested())
						{
							vk.DestroyPipeline(device(), newPipeline, nullptr);
							return;
						}
						auto result = m_pipelineMap.emplace(std::make_pair(key, newPipeline));
						if (!result.second)
							vk.DestroyPipeline(device(), newPipeline, nullptr);
					}
				}
			}
		}
	}
}

extern bool g_bSteamIsActiveWindow;

VkPipeline CVulkanDevice::pipeline(ShaderType type, uint32_t layerCount, uint32_t ycbcrMask, uint32_t blur_layers, uint32_t colorspace_mask, uint32_t output_eotf, bool itm_enable, bool fsr_simple_output, bool simple_blend, CompositeShaderConfig_t composite_config)
{
	uint32_t effective_debug = g_uCompositeDebug;
	if ( g_bSteamIsActiveWindow )
		effective_debug &= ~(CompositeDebugFlag::Heatmap | CompositeDebugFlag::Heatmap_MSWCG | CompositeDebugFlag::Heatmap_Hard);

	std::lock_guard<std::mutex> lock(m_pipelineMutex);
	PipelineInfo_t key = {type, layerCount, ycbcrMask, blur_layers, effective_debug, colorspace_mask, output_eotf, itm_enable, fsr_simple_output, simple_blend, composite_config};
	auto search = m_pipelineMap.find(key);
	if (search == m_pipelineMap.end())
	{
		VkPipeline result = compilePipeline(layerCount, ycbcrMask, type, blur_layers, effective_debug, colorspace_mask, output_eotf, itm_enable, fsr_simple_output, simple_blend, composite_config);
		m_pipelineMap[key] = result;
		return result;
	}
	else
	{
		return search->second;
	}
}


int32_t CVulkanDevice::findMemoryType( VkMemoryPropertyFlags properties, uint32_t requiredTypeBits )
{
	for ( uint32_t i = 0; i < m_memoryProperties.memoryTypeCount; i++ )
	{
		if ( ( ( 1 << i ) & requiredTypeBits ) == 0 )
			continue;
		
		if ( ( properties & m_memoryProperties.memoryTypes[ i ].propertyFlags ) != properties )
			continue;
		
		return i;
	}
	
	return -1;
}

std::unique_ptr<CVulkanCmdBuffer> CVulkanDevice::commandBuffer()
{
	std::unique_ptr<CVulkanCmdBuffer> cmdBuffer;
	if (m_unusedCmdBufs.empty())
	{
		VkCommandBuffer rawCmdBuffer;
		VkCommandBufferAllocateInfo commandBufferAllocateInfo = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.commandPool = m_commandPool,
			.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			.commandBufferCount = 1
		};

		VkResult res = vk.AllocateCommandBuffers( device(), &commandBufferAllocateInfo, &rawCmdBuffer );
		if ( res != VK_SUCCESS )
		{
			vk_errorf( res, "vkAllocateCommandBuffers failed" );
			return nullptr;
		}

		cmdBuffer = std::make_unique<CVulkanCmdBuffer>(this, rawCmdBuffer, queue(), queueFamily());
	}
	else
	{
		cmdBuffer = std::move(m_unusedCmdBufs.back());
		m_unusedCmdBufs.pop_back();
	}

	cmdBuffer->begin();
	return cmdBuffer;
}

uint64_t CVulkanDevice::submitInternal( CVulkanCmdBuffer* cmdBuffer )
{
	cmdBuffer->end();

	// The seq no of the last submission.
	const uint64_t lastSubmissionSeqNo = m_submissionSeqNo++;

	// This is the seq no of the command buffer we are going to submit.
	const uint64_t nextSeqNo = lastSubmissionSeqNo + 1;

	std::vector<VkSemaphore> pSignalSemaphores;
	std::vector<uint64_t> ulSignalPoints;

	std::vector<VkPipelineStageFlags> uWaitStageFlags;
	std::vector<VkSemaphore> pWaitSemaphores;
	std::vector<uint64_t> ulWaitPoints;

	pSignalSemaphores.push_back( m_scratchTimelineSemaphore );
	ulSignalPoints.push_back( nextSeqNo );
	if (cmdBuffer->m_signalCompletion)
	{
		pSignalSemaphores.push_back(cmdBuffer->m_completionSemaphore);
		ulSignalPoints.push_back(0); // Binary semaphore; timeline value is ignored.
	}


	for ( auto &dep : cmdBuffer->GetExternalSignals() )
	{
		pSignalSemaphores.push_back( dep.pTimelineSemaphore->pVkSemaphore );
		ulSignalPoints.push_back( dep.ulPoint );
	}

	for ( auto &dep : cmdBuffer->GetExternalDependencies() )
	{
		pWaitSemaphores.push_back( dep.pTimelineSemaphore->pVkSemaphore );
		ulWaitPoints.push_back( dep.ulPoint );
		// Client buffers can first be read by a transfer into a sampled shadow.
		uWaitStageFlags.push_back( VK_PIPELINE_STAGE_ALL_COMMANDS_BIT );
	}

	VkTimelineSemaphoreSubmitInfo timelineInfo = {
		.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
		// no need to ensure order of cmd buffer submission, we only have one queue
		.waitSemaphoreValueCount = static_cast<uint32_t>( ulWaitPoints.size() ),
		.pWaitSemaphoreValues = ulWaitPoints.data(),
		.signalSemaphoreValueCount = static_cast<uint32_t>( ulSignalPoints.size() ),
		.pSignalSemaphoreValues = ulSignalPoints.data(),
	};

	VkCommandBuffer rawCmdBuffer = cmdBuffer->rawBuffer();

	VkSubmitInfo submitInfo = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.pNext = &timelineInfo,
		.waitSemaphoreCount = static_cast<uint32_t>( pWaitSemaphores.size() ),
		.pWaitSemaphores = pWaitSemaphores.data(),
		.pWaitDstStageMask = uWaitStageFlags.data(),
		.commandBufferCount = 1,
		.pCommandBuffers = &rawCmdBuffer,
		.signalSemaphoreCount = static_cast<uint32_t>( pSignalSemaphores.size() ),
		.pSignalSemaphores = pSignalSemaphores.data(),
	};

	vk_check( vk.QueueSubmit( cmdBuffer->queue(), 1, &submitInfo, VK_NULL_HANDLE ) );

	return nextSeqNo;
}

uint64_t CVulkanDevice::submit(std::unique_ptr<CVulkanCmdBuffer> cmdBuffer, bool completionFence)
{
	if (completionFence && !cmdBuffer->m_completionSemaphore)
	{
		VkExportSemaphoreCreateInfo external = {
			.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO,
			.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT,
		};
		VkSemaphoreCreateInfo info = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &external };
		if (vk.CreateSemaphore(device(), &info, nullptr, &cmdBuffer->m_completionSemaphore) != VK_SUCCESS)
			vk_log.warnf("Completion sync-file semaphore unavailable; using CPU synchronization");
	}
	cmdBuffer->m_signalCompletion = completionFence && cmdBuffer->m_completionSemaphore;
	uint64_t nextSeqNo = submitInternal(cmdBuffer.get());
	if (cmdBuffer->m_signalCompletion)
	{
		VkSemaphoreGetFdInfoKHR info = {
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR,
			.semaphore = cmdBuffer->m_completionSemaphore,
			.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT,
		};
		int fd = -1;
		VkResult res = vk.GetSemaphoreFdKHR(device(), &info, &fd);
		if (res == VK_SUCCESS)
			cmdBuffer->m_completionFd = fd; // -1 is a valid, already-signaled sync file.
		else
		{
			vk_errorf(res, "Failed to export composition completion fence");
			cmdBuffer->m_completionFd.reset();
			cmdBuffer->m_completionExportFailed = true;
		}
	}
	m_pendingCmdBufs.emplace(nextSeqNo, std::move(cmdBuffer));
	return nextSeqNo;
}

VkDescriptorSet CVulkanDevice::descriptorSet()
{
	// Updating a descriptor set that an earlier submission still uses is invalid.
	// Reclaim completed command buffers first; only wait when all sets are busy.
	if ( m_freeDescriptorSets.empty() )
		garbageCollect();

	if ( m_freeDescriptorSets.empty() && !m_pendingCmdBufs.empty() )
	{
		vk_log.warnf( "Descriptor set pool exhausted; waiting for the oldest Vulkan submission" );
		wait( m_pendingCmdBufs.begin()->first );
	}

	assert( !m_freeDescriptorSets.empty() );
	VkDescriptorSet descriptorSet = m_freeDescriptorSets.back();
	m_freeDescriptorSets.pop_back();
	return descriptorSet;
}

void CVulkanDevice::recycleDescriptorSets( std::vector<VkDescriptorSet>& descriptorSets )
{
	m_freeDescriptorSets.insert(
		m_freeDescriptorSets.end(), descriptorSets.begin(), descriptorSets.end() );
	descriptorSets.clear();
}

void CVulkanDevice::garbageCollect( void )
{
	uint64_t currentSeqNo;
	vk_check( vk.GetSemaphoreCounterValue(device(), m_scratchTimelineSemaphore, &currentSeqNo) );

	// Reclaim uploads even when each connector waits on an older submission.
	if ( currentSeqNo == m_submissionSeqNo )
		m_uploadBufferOffset = 0;

	resetCmdBuffers(currentSeqNo);
}

std::optional<int> CVulkanDevice::takeCompletionFence(uint64_t sequence)
{
	auto it = m_pendingCmdBufs.find(sequence);
	return it == m_pendingCmdBufs.end() ? std::nullopt : std::exchange(it->second->m_completionFd, std::nullopt);
}

VulkanTimelineSemaphore_t::~VulkanTimelineSemaphore_t()
{
	if ( pVkSemaphore != VK_NULL_HANDLE )
	{
		pDevice->vk.DestroySemaphore( pDevice->device(), pVkSemaphore, nullptr );
		pVkSemaphore = VK_NULL_HANDLE;
	}
}

int VulkanTimelineSemaphore_t::GetFd() const
{
	const VkSemaphoreGetFdInfoKHR semaphoreGetInfo =
	{
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR,
		.semaphore = pVkSemaphore,
		.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT,
	};

	int32_t nFd = -1;
	VkResult res = VK_SUCCESS;
	if ( ( res = pDevice->vk.GetSemaphoreFdKHR( pDevice->device(), &semaphoreGetInfo, &nFd ) ) != VK_SUCCESS )
	{
		vk_errorf( res, "vkGetSemaphoreFdKHR failed" );
		return -1;
	}

	return nFd;
}

std::shared_ptr<VulkanTimelineSemaphore_t> CVulkanDevice::CreateTimelineSemaphore( uint64_t ulStartPoint, bool bShared )
{
	std::shared_ptr<VulkanTimelineSemaphore_t> pSemaphore = std::make_unique<VulkanTimelineSemaphore_t>();
	pSemaphore->pDevice = this;

	VkSemaphoreCreateInfo createInfo =
	{
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
	};

	VkSemaphoreTypeCreateInfo typeInfo =
	{
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
		.pNext = std::exchange( createInfo.pNext, &typeInfo ),
		.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
		.initialValue = ulStartPoint,
	};

	VkExportSemaphoreCreateInfo exportInfo =
	{
		.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO,
		.pNext = bShared ? std::exchange( createInfo.pNext, &exportInfo ) : nullptr,
		// This is a syncobj fd for any drivers using syncobj.
		.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT,
	};

	VkResult res;
	if ( ( res = vk.CreateSemaphore( m_device, &createInfo, nullptr, &pSemaphore->pVkSemaphore ) ) != VK_SUCCESS )
	{
		vk_errorf( res, "vkCreateSemaphore failed" );
		return nullptr;
	}

	return pSemaphore;
}

std::shared_ptr<VulkanTimelineSemaphore_t> CVulkanDevice::ImportTimelineSemaphore( gamescope::CTimeline *pTimeline )
{
	std::shared_ptr<VulkanTimelineSemaphore_t> pSemaphore = std::make_unique<VulkanTimelineSemaphore_t>();
	pSemaphore->pDevice = this;

	const VkSemaphoreTypeCreateInfo typeInfo =
	{
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
		.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
	};

	const VkSemaphoreCreateInfo createInfo =
	{
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
		.pNext = &typeInfo,
	};

	VkResult res;
	if ( ( res = vk.CreateSemaphore( m_device, &createInfo, nullptr, &pSemaphore->pVkSemaphore ) ) != VK_SUCCESS )
	{
		vk_errorf( res, "vkCreateSemaphore failed" );
		return nullptr;
	}

    // "Importing a semaphore payload from a file descriptor transfers
    // ownership of the file descriptor from the application to the Vulkan
    // implementation. The application must not perform any operations on
    // the file descriptor after a successful import."
	//
	// Thus, we must dup.

	VkImportSemaphoreFdInfoKHR importFdInfo =
	{
		.sType = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR,
		.pNext = nullptr,
		.semaphore = pSemaphore->pVkSemaphore,
		.flags = 0, // not temporary
		.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT,
		.fd = dup( pTimeline->GetSyncobjFd() ),
	};
	if ( ( res = vk.ImportSemaphoreFdKHR( m_device, &importFdInfo ) ) != VK_SUCCESS )
	{
		vk_errorf( res, "vkImportSemaphoreFdKHR failed" );
		return nullptr;
	}

	return pSemaphore;
}

bool CVulkanCmdBuffer::AddDependency( std::shared_ptr<VulkanTimelineSemaphore_t> pTimelineSemaphore, uint64_t ulPoint )
{
	if ( !pTimelineSemaphore )
		return false;

	for ( auto &dependency : m_ExternalDependencies )
	{
		if ( dependency.pTimelineSemaphore == pTimelineSemaphore )
		{
			dependency.ulPoint = std::max( dependency.ulPoint, ulPoint );
			return true;
		}
	}
	m_ExternalDependencies.emplace_back( std::move( pTimelineSemaphore ), ulPoint );
	return true;
}

bool CVulkanCmdBuffer::AddSignal( std::shared_ptr<VulkanTimelineSemaphore_t> pTimelineSemaphore, uint64_t ulPoint )
{
	if ( !pTimelineSemaphore )
		return false;

	m_ExternalSignals.emplace_back( std::move( pTimelineSemaphore ), ulPoint );
	return true;
}

void CVulkanDevice::wait(uint64_t sequence, bool reset)
{
	if (m_submissionSeqNo == sequence)
		m_uploadBufferOffset = 0;

	VkSemaphoreWaitInfo waitInfo = {
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
		.semaphoreCount = 1,
		.pSemaphores = &m_scratchTimelineSemaphore,
		.pValues = &sequence,
	} ;

	vk_check( vk.WaitSemaphores( device(), &waitInfo, ~0ull ) );

	if (reset)
		resetCmdBuffers(sequence);
}

bool CVulkanDevice::isComplete(uint64_t sequence)
{
	if ( sequence == 0 )
		return true;

	uint64_t completed = 0;
	return vk.GetSemaphoreCounterValue( device(), m_scratchTimelineSemaphore, &completed ) == VK_SUCCESS && completed >= sequence;
}

void CVulkanDevice::waitIdle(bool reset)
{
	wait(m_submissionSeqNo, reset);
}

void CVulkanDevice::resetCmdBuffers(uint64_t sequence)
{
	auto end = m_pendingCmdBufs.upper_bound(sequence);
	for (auto it = m_pendingCmdBufs.begin(); it != end; ++it)
	{
		it->second->reset();
		m_unusedCmdBufs.push_back(std::move(it->second));
	}
	m_pendingCmdBufs.erase(m_pendingCmdBufs.begin(), end);
}

CVulkanCmdBuffer::CVulkanCmdBuffer(CVulkanDevice *parent, VkCommandBuffer cmdBuffer, VkQueue queue, uint32_t queueFamily)
	: m_cmdBuffer(cmdBuffer), m_device(parent), m_queue(queue), m_queueFamily(queueFamily)
{
}

CVulkanCmdBuffer::~CVulkanCmdBuffer()
{
	if (m_completionFd && *m_completionFd >= 0)
		close(*m_completionFd);
	if (m_completionSemaphore)
		m_device->vk.DestroySemaphore(m_device->device(), m_completionSemaphore, nullptr);
	m_device->recycleDescriptorSets( m_descriptorSets );
	m_device->vk.FreeCommandBuffers(m_device->device(), m_device->commandPool(), 1, &m_cmdBuffer);
}

void CVulkanCmdBuffer::reset()
{
	if (m_completionFd && *m_completionFd >= 0)
		close(*m_completionFd);
	if (m_completionExportFailed)
	{
		m_device->vk.DestroySemaphore(m_device->device(), m_completionSemaphore, nullptr);
		m_completionSemaphore = VK_NULL_HANDLE;
		m_completionExportFailed = false;
	}
	m_completionFd.reset();
	m_signalCompletion = false;
	vk_check( m_device->vk.ResetCommandBuffer(m_cmdBuffer, 0) );
	m_device->recycleDescriptorSets( m_descriptorSets );
	m_textureRefs.clear();
	m_releasePointRefs.clear();
	m_textureState.clear();

	m_ExternalDependencies.clear();
	m_ExternalSignals.clear();
}

void CVulkanCmdBuffer::begin()
{
	VkCommandBufferBeginInfo commandBufferBeginInfo = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
	};

	vk_check( m_device->vk.BeginCommandBuffer(m_cmdBuffer, &commandBufferBeginInfo) );

	clearState();
}

void CVulkanCmdBuffer::end()
{
	insertBarrier(true);
	vk_check( m_device->vk.EndCommandBuffer(m_cmdBuffer) );
}

void CVulkanCmdBuffer::bindTexture(uint32_t slot, gamescope::Rc<CVulkanTexture> texture)
{
	m_boundTextures[slot] = texture.get();
	if (texture)
		m_textureRefs.emplace_back(std::move(texture));
}

void CVulkanCmdBuffer::bindColorMgmtLuts(uint32_t slot, gamescope::Rc<CVulkanTexture> lut1d, gamescope::Rc<CVulkanTexture> lut3d)
{
	m_shaperLut[slot] = lut1d.get();
	m_lut3D[slot] = lut3d.get();

	if (lut1d != nullptr)
		m_textureRefs.emplace_back(std::move(lut1d));
	if (lut3d != nullptr)
		m_textureRefs.emplace_back(std::move(lut3d));
}

void CVulkanCmdBuffer::setTextureSrgb(uint32_t slot, bool srgb)
{
	m_useSrgb[slot] = srgb;
}

void CVulkanCmdBuffer::setSamplerNearest(uint32_t slot, bool nearest)
{
	m_samplerState[slot].bNearest = nearest;
}

void CVulkanCmdBuffer::setSamplerUnnormalized(uint32_t slot, bool unnormalized)
{
	m_samplerState[slot].bUnnormalized = unnormalized;
}

void CVulkanCmdBuffer::bindTarget(gamescope::Rc<CVulkanTexture> target)
{
	m_target = target.get();
	if (target)
		m_textureRefs.emplace_back(std::move(target));
}

void CVulkanCmdBuffer::clearState()
{
	for (auto& texture : m_boundTextures)
		texture = nullptr;

	for (auto& sampler : m_samplerState)
		sampler = {};

	m_target = nullptr;
	m_useSrgb.reset();
}

template<class PushData, class... Args>
void CVulkanCmdBuffer::uploadConstants(Args&&... args)
{
	PushData data(std::forward<Args>(args)...);

	auto [ptr, offset] = m_device->uploadBufferData(sizeof(data), m_device->m_uUniformBufferAlignment);
	m_renderBufferOffset = offset;
	m_renderBufferSize = sizeof(data);
	memcpy(ptr, &data, sizeof(data));
}

void CVulkanCmdBuffer::bindPipeline(VkPipeline pipeline)
{
	m_device->vk.CmdBindPipeline(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
}

void CVulkanCmdBuffer::dispatch(uint32_t x, uint32_t y, uint32_t z)
{
	for (auto src : m_boundTextures)
	{
		if (src)
			prepareSrcImage(src);
	}
	assert(m_target != nullptr);
	prepareDestImage(m_target);
	insertBarrier();

	bindDescriptors(VK_PIPELINE_BIND_POINT_COMPUTE);
	const uint32_t storeBgra = m_target->storageViewSwapsRB();
	m_device->vk.CmdPushConstants(m_cmdBuffer, m_device->pipelineLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(storeBgra), &storeBgra);
	m_device->vk.CmdDispatch(m_cmdBuffer, x, y, z);
	markDirty(m_target);
}

void CVulkanCmdBuffer::bindDescriptors(VkPipelineBindPoint bindPoint, CVulkanTexture *linearSource)
{
	VkDescriptorSet descriptorSet = m_device->descriptorSet();
	m_descriptorSets.push_back( descriptorSet );

	std::array<VkWriteDescriptorSet, 8> writeDescriptorSets;
	std::array<VkDescriptorImageInfo, VKR_SAMPLER_SLOTS> imageDescriptors = {};
	std::array<VkDescriptorImageInfo, VKR_SAMPLER_SLOTS> ycbcrImageDescriptors = {};
	std::array<VkDescriptorImageInfo, VKR_TARGET_SLOTS> targetDescriptors = {};
	std::array<VkDescriptorImageInfo, VKR_LUT3D_COUNT> shaperLutDescriptor = {};
	std::array<VkDescriptorImageInfo, VKR_LUT3D_COUNT> lut3DDescriptor = {};
	VkDescriptorBufferInfo scratchDescriptor = {};
	VkDescriptorBufferInfo linearDescriptor = { linearSource ? linearSource->linearBuffer() : VK_NULL_HANDLE, 0, VK_WHOLE_SIZE };

	writeDescriptorSets[0] = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = descriptorSet,
		.dstBinding = 0,
		.dstArrayElement = 0,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.pBufferInfo = &scratchDescriptor,
	};

	writeDescriptorSets[1] = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = descriptorSet,
		.dstBinding = 1,
		.dstArrayElement = 0,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.pImageInfo = &targetDescriptors[0],
	};

	writeDescriptorSets[2] = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = descriptorSet,
		.dstBinding = 2,
		.dstArrayElement = 0,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.pImageInfo = &targetDescriptors[1],
	};

	writeDescriptorSets[3] = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = descriptorSet,
		.dstBinding = 3,
		.dstArrayElement = 0,
		.descriptorCount = imageDescriptors.size(),
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.pImageInfo = imageDescriptors.data(),
	};

	writeDescriptorSets[4] = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = descriptorSet,
		.dstBinding = 4,
		.dstArrayElement = 0,
		.descriptorCount = ycbcrImageDescriptors.size(),
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.pImageInfo = ycbcrImageDescriptors.data(),
	};

	writeDescriptorSets[5] = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = descriptorSet,
		.dstBinding = 5,
		.dstArrayElement = 0,
		.descriptorCount = shaperLutDescriptor.size(),
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.pImageInfo = shaperLutDescriptor.data(),
	};

	writeDescriptorSets[6] = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = descriptorSet,
		.dstBinding = 6,
		.dstArrayElement = 0,
		.descriptorCount = lut3DDescriptor.size(),
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.pImageInfo = lut3DDescriptor.data(),
	};

	writeDescriptorSets[7] = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = descriptorSet,
		.dstBinding = 7, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.pBufferInfo = &linearDescriptor,
	};
	scratchDescriptor.buffer = m_device->m_uploadBuffer;
	scratchDescriptor.offset = m_renderBufferOffset;
	scratchDescriptor.range = m_renderBufferSize;

	for (uint32_t i = 0; i < VKR_SAMPLER_SLOTS; i++)
	{
		imageDescriptors[i].sampler = m_device->sampler(m_samplerState[i]);
		imageDescriptors[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		ycbcrImageDescriptors[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		if (m_boundTextures[i] == nullptr)
			continue;

		VkImageView view = m_useSrgb[i] ? m_boundTextures[i]->srgbView() : m_boundTextures[i]->linearView();

		if (m_boundTextures[i]->format() == VK_FORMAT_G8_B8R8_2PLANE_420_UNORM)
			ycbcrImageDescriptors[i].imageView = view;
		else
			imageDescriptors[i].imageView = view;
	}

	for (uint32_t i = 0; i < VKR_LUT3D_COUNT; i++)
	{
		SamplerState linearState;
		linearState.bNearest = false;
		linearState.bUnnormalized = false;
		SamplerState nearestState; // TODO(Josh): Probably want to do this when I bring in tetrahedral interpolation.
		nearestState.bNearest = true;
		nearestState.bUnnormalized = false;

		SamplerState shaperState = linearState;
		if (m_shaperLut[i] && m_shaperLut[i]->format() == VK_FORMAT_R16_SFLOAT)
		{
			shaperState.bNearest = true;
			shaperState.bUnnormalized = true;
		}
		shaperLutDescriptor[i].sampler = m_device->sampler(shaperState);
		shaperLutDescriptor[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		// TODO(Josh): I hate the fact that srgbView = view *as* raw srgb and treat as linear.
		// I need to change this, it's so utterly stupid and confusing.
		shaperLutDescriptor[i].imageView = m_shaperLut[i] ? m_shaperLut[i]->srgbView() : VK_NULL_HANDLE;

		lut3DDescriptor[i].sampler = m_device->sampler(nearestState);
		lut3DDescriptor[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		lut3DDescriptor[i].imageView = m_lut3D[i] ? m_lut3D[i]->srgbView() : VK_NULL_HANDLE;
	}

	if (bindPoint == VK_PIPELINE_BIND_POINT_GRAPHICS)
	{
		// Fragment shaders do not use the storage-image bindings.
	}
	else if (!m_target->isYcbcr())
	{
		targetDescriptors[0].imageView = m_target->storageView();
		targetDescriptors[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
	}
	else
	{
		targetDescriptors[0].imageView = m_target->lumaView();
		targetDescriptors[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

		targetDescriptors[1].imageView = m_target->chromaView();
		targetDescriptors[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
	}

	m_device->vk.UpdateDescriptorSets(m_device->device(), writeDescriptorSets.size(), writeDescriptorSets.data(), 0, nullptr);

	m_device->vk.CmdBindDescriptorSets(m_cmdBuffer, bindPoint, m_device->pipelineLayout(), 0, 1, &descriptorSet, 0, nullptr);

}

void CVulkanCmdBuffer::drawComposite(const FrameInfo_t *frame, EOTF outputTF, CompositeShaderConfig_t config, bool copyBase)
{
	const bool cacheable = cv_raster_damage && kDisablePartialComposition && m_target->outputImage() &&
		!GetBackend()->UsesVulkanSwapchain() && !frame->frameGenerationActive && !g_pLastReshadeEffect;
	auto state = composite_state(frame, outputTF, config);
	const auto *previous = cacheable ? m_target->compositeState.get() : nullptr;
	VkRect2D damage = {{0, 0}, {m_target->width(), m_target->height()}};
	if (previous && previous->layerCount == state.layerCount && previous->outputTF == state.outputTF &&
		previous->config == state.config && previous->luts == state.luts)
	{
		int left = m_target->width(), top = m_target->height(), right = 0, bottom = 0;
		for (int i = 0; i < state.layerCount; ++i)
		{
			if (previous->layers[i] == state.layers[i])
				continue;
			const auto &old = previous->layers[i], &current = state.layers[i];
			if (old.damage.history && old.damage.history == current.damage.history &&
				old.geometry == current.geometry && old.size == current.size && old.flags == current.flags && old.ctm == current.ctm &&
				frame->layers[i].shaderFilter() != GamescopeUpscaleFilter::PIXEL)
			{
				if (auto changed = current.damage.history->Between(old.damage.sequence, current.damage.sequence))
				{
					auto r = composite_damage_bounds(current, *changed, m_target->width(), m_target->height());
					if (r.extent.width && r.extent.height)
					{
						left = std::min(left, r.offset.x); top = std::min(top, r.offset.y);
						right = std::max(right, r.offset.x + int(r.extent.width));
						bottom = std::max(bottom, r.offset.y + int(r.extent.height));
					}
					continue;
				}
			}
			for (const auto *layer : std::array<const CompositeState_t::Layer *, 2>{&previous->layers[i], &state.layers[i]})
			{
				const auto r = composite_layer_bounds(*layer, m_target->width(), m_target->height());
				left = std::min(left, r.offset.x); top = std::min(top, r.offset.y);
				right = std::max(right, r.offset.x + int(r.extent.width));
				bottom = std::max(bottom, r.offset.y + int(r.extent.height));
			}
		}
		if (right <= left || bottom <= top)
		{
			m_target->compositeState = std::make_unique<CompositeState_t>(std::move(state));
			return;
		}
		damage = {{left, top}, {uint32_t(right - left), uint32_t(bottom - top)}};
	}

	const bool preserve = previous && (damage.offset.x != 0 || damage.offset.y != 0 ||
		damage.extent.width != m_target->width() || damage.extent.height != m_target->height());
	VkRect2D renderDamage = damage;
	if (copyBase)
	{
		renderDamage = {};
		for (int i = 1; i < state.layerCount; ++i)
			renderDamage = union_rect(renderDamage, composite_layer_bounds(state.layers[i], m_target->width(), m_target->height()));
		renderDamage = intersect_rect(renderDamage, damage);
	}
	std::array<CVulkanTexture *, k_nMaxLayers> linearSources{};
	for (int i = copyBase ? 1 : 0; i < frame->layers.count(); ++i)
	{
		auto *tex = m_boundTextures[i];
		const auto colorspace = frame->layers[i].colorspace;
		if (cv_raster_linear_read && tex && tex->linearSource && tex->linearSourceDirty &&
			(config.samplingMask & (1u << (k_nMaxLayers + i))) &&
			(colorspace == GAMESCOPE_APP_TEXTURE_COLORSPACE_LINEAR || colorspace == GAMESCOPE_APP_TEXTURE_COLORSPACE_SRGB))
		{
			const auto &current = tex->linearSourceDamage;
			bool smallDamage = false;
			if (current.history && current.history == tex->sampledDamage.history)
				if (auto box = current.history->Between(tex->sampledDamage.sequence, current.sequence))
					smallDamage = uint64_t(box->width) * box->height < uint64_t(tex->width()) * tex->height() / 4;
			if (!smallDamage && tex->linearSource->linearBuffer())
			{
				linearSources[i] = tex->linearSource.get();
			}
		}
	}
	if (copyBase)
		copyImage(frame->layers[0].tex, m_target, damage);
	if (!renderDamage.extent.width || !renderDamage.extent.height)
	{
		if (cacheable)
			m_target->compositeState = std::make_unique<CompositeState_t>(std::move(state));
		return;
	}
	for (uint32_t i = copyBase ? 1 : 0; i < m_boundTextures.size(); ++i)
	{
		auto *src = m_boundTextures[i];
		if (!src)
			continue;
		if (i < linearSources.size() && linearSources[i])
		{
			src->ReferenceReleasePoints(m_releasePointRefs);
			prepareSrcImage(linearSources[i]);
		}
		else
			prepareSrcImage(src);
	}
	// Copies into sampled shadows advance the version during prepareSrcImage.
	for (int i = 0; i < frame->layers.count(); ++i)
		state.layers[i].version = frame->layers[i].tex->contentVersion.load(std::memory_order_relaxed);
	prepareDestImage(m_target, !preserve);
	insertBarrier();
	bindDescriptors(VK_PIPELINE_BIND_POINT_GRAPHICS);

	VkRenderingAttachmentInfo attachment = {
		.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
		.imageView = outputTF == EOTF_Gamma22 ? m_target->linearView() : m_target->srgbView(),
		.imageLayout = VK_IMAGE_LAYOUT_GENERAL,
		.loadOp = preserve || copyBase ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
	};
	VkRenderingInfo rendering = {
		.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
		.renderArea = renderDamage, .layerCount = 1,
		.colorAttachmentCount = 1, .pColorAttachments = &attachment,
	};
	m_device->vk.CmdBeginRendering(m_cmdBuffer, &rendering);
	if (preserve && !copyBase)
	{
		VkClearAttachment clear = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .colorAttachment = 0 };
		VkClearRect rect = { .rect = damage, .baseArrayLayer = 0, .layerCount = 1 };
		m_device->vk.CmdClearAttachments(m_cmdBuffer, 1, &clear, 1, &rect);
	}
	VkViewport viewport = { 0.0f, 0.0f, float(m_target->width()), float(m_target->height()), 0.0f, 1.0f };
	m_device->vk.CmdSetViewport(m_cmdBuffer, 0, 1, &viewport);
	config.renderFormat = outputTF == EOTF_Gamma22 ? ToSrgbVulkanFormat(m_target->format()) : m_target->format();
	for (int i = copyBase ? 1 : 0; i < frame->layers.count(); ++i)
	{
		VkRect2D scissor = intersect_rect(composite_layer_bounds(state.layers[i], m_target->width(), m_target->height()), renderDamage);
		if (!scissor.extent.width || !scissor.extent.height)
			continue;
		config.drawLayer = i;
		config.linearSource = linearSources[i] != nullptr;
		if (auto *source = linearSources[i])
		{
			bindDescriptors(VK_PIPELINE_BIND_POINT_GRAPHICS, source);
			const auto &dma = source->dmabuf();
			const uint32_t data[] = {dma.stride[0] / 4, dma.offset[0] / 4, source->width(), source->height(),
				uint32_t(source->format() == VK_FORMAT_B8G8R8A8_UNORM) | (uint32_t(!DRMFormatHasAlpha(dma.format)) << 1)};
			m_device->vk.CmdPushConstants(m_cmdBuffer, m_device->pipelineLayout(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(data), data);
		}
		VkPipeline pipeline = m_device->pipeline(SHADER_TYPE_BLIT, frame->layers.count(), frame->ycbcrMask(), 0,
			frame->colorspaceMask(), outputTF, false, false, false, config);
		m_device->vk.CmdBindPipeline(m_cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
		m_device->vk.CmdSetScissor(m_cmdBuffer, 0, 1, &scissor);
		m_device->vk.CmdDraw(m_cmdBuffer, 3, 1, 0, 0);
	}
	m_device->vk.CmdEndRendering(m_cmdBuffer);
	markDirty(m_target);
	if (cacheable)
		m_target->compositeState = std::make_unique<CompositeState_t>(std::move(state));
}

void CVulkanCmdBuffer::copyImage(gamescope::Rc<CVulkanTexture> src, gamescope::Rc<CVulkanTexture> dst, std::optional<VkRect2D> rect)
{
	assert(src->width() == dst->width());
	assert(src->height() == dst->height());
	if (src->linearSource && src->linearSourceDirty)
	{
		// A raw copy can read the producer directly, without first refreshing the
		// sampled shadow. Retain its explicit release point until this copy ends.
		src->ReferenceReleasePoints(m_releasePointRefs);
		m_textureRefs.emplace_back(src);
		src = src->linearSource;
	}
	prepareSrcImage(src.get());
	prepareDestImage(dst.get(), !rect || (rect->offset.x == 0 && rect->offset.y == 0 &&
		rect->extent.width == dst->width() && rect->extent.height == dst->height()));
	insertBarrier();

	VkImageCopy region = {
		.srcSubresource = {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.layerCount = 1
		},
		.srcOffset = {rect ? rect->offset.x : 0, rect ? rect->offset.y : 0, 0},
		.dstSubresource = {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.layerCount = 1
		},
		.dstOffset = {rect ? rect->offset.x : 0, rect ? rect->offset.y : 0, 0},
		.extent = {
			.width = rect ? rect->extent.width : src->width(),
			.height = rect ? rect->extent.height : src->height(),
			.depth = 1
		},
	};

	m_device->vk.CmdCopyImage(m_cmdBuffer, src->vkImage(), VK_IMAGE_LAYOUT_GENERAL, dst->vkImage(), VK_IMAGE_LAYOUT_GENERAL, 1, &region);

	markDirty(dst.get());
	m_textureRefs.emplace_back(std::move(src));
	m_textureRefs.emplace_back(std::move(dst));
}

void CVulkanCmdBuffer::copyImageToReadback(gamescope::Rc<CVulkanTexture> src)
{
	assert(src->isReadback());
	prepareSrcImage(src.get());
	insertBarrier();

	if (src->isYcbcr())
	{
		VkBufferImageCopy regions[2] = {
			{
				.bufferOffset = src->lumaOffset(),
				.bufferRowLength = src->lumaRowPitch(),
				.imageSubresource = {
					.aspectMask = VK_IMAGE_ASPECT_PLANE_0_BIT,
					.layerCount = 1
				},
				.imageExtent = {
					.width = src->width(),
					.height = src->height(),
					.depth = 1
				},
			},
			{
				.bufferOffset = src->chromaOffset(),
				// bufferRowLength is in texels and chroma texels are 2 bytes wide.
				.bufferRowLength = src->chromaRowPitch() / 2,
				.imageSubresource = {
					.aspectMask = VK_IMAGE_ASPECT_PLANE_1_BIT,
					.layerCount = 1
				},
				.imageExtent = {
					.width = (src->width() + 1) / 2,
					.height = (src->height() + 1) / 2,
					.depth = 1
				},
			},
		};

		m_device->vk.CmdCopyImageToBuffer(m_cmdBuffer, src->vkImage(), VK_IMAGE_LAYOUT_GENERAL, src->readbackBuffer(), 2, regions);
	}
	else
	{
		VkBufferImageCopy region = {
			.bufferOffset = 0,
			.bufferRowLength = src->rowPitch() / DRMFormatGetBPP(src->drmFormat()),
			.imageSubresource = {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.layerCount = 1
			},
			.imageExtent = {
				.width = src->width(),
				.height = src->height(),
				.depth = 1
			},
		};

		m_device->vk.CmdCopyImageToBuffer(m_cmdBuffer, src->vkImage(), VK_IMAGE_LAYOUT_GENERAL, src->readbackBuffer(), 1, &region);
	}

	m_textureRefs.emplace_back(std::move(src));
}

void CVulkanCmdBuffer::copyBufferToImage(VkBuffer buffer, VkDeviceSize offset, uint32_t stride, gamescope::Rc<CVulkanTexture> dst)
{
	prepareDestImage(dst.get());
	insertBarrier();
	VkBufferImageCopy region = {
		.bufferOffset = offset,
		.bufferRowLength = stride,
		.imageSubresource = {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.layerCount = 1,
		},
		.imageExtent = {
			.width = dst->width(),
			.height = dst->height(),
			.depth = dst->depth(),
		},
	};

	m_device->vk.CmdCopyBufferToImage(m_cmdBuffer, buffer, dst->vkImage(), VK_IMAGE_LAYOUT_GENERAL, 1, &region);

	markDirty(dst.get());

	m_textureRefs.emplace_back(std::move(dst));
}

static uint32_t external_queue_family( CVulkanDevice *device )
{
	return device->supportsModifiers()
		? VK_QUEUE_FAMILY_FOREIGN_EXT
		: VK_QUEUE_FAMILY_EXTERNAL_KHR;
}

void CVulkanCmdBuffer::prepareSrcImage(CVulkanTexture *image)
{
	if ( image->linearSource && image->linearSourceDirty && !m_textureState.contains( image ) )
	{
		std::optional<VkRect2D> region;
		const auto &current = image->linearSourceDamage;
		if (cv_raster_damage && current.history && current.history == image->sampledDamage.history)
		{
			if (auto damage = current.history->Between(image->sampledDamage.sequence, current.sequence))
			{
				const int left = std::clamp(damage->x, 0, int(image->width()));
				const int top = std::clamp(damage->y, 0, int(image->height()));
				const int right = std::clamp(damage->x + damage->width, left, int(image->width()));
				const int bottom = std::clamp(damage->y + damage->height, top, int(image->height()));
				region = VkRect2D{{left, top}, {uint32_t(right - left), uint32_t(bottom - top)}};
			}
		}
		// Each client buffer has its own shadow. Accumulate surface damage since
		// that shadow was sampled, including commits that used other buffers.
		if (!region || (region->extent.width && region->extent.height))
			copyImage(image->linearSource, image, region);
		image->sampledDamage = current;
		image->linearSourceDirty = false;
	}
	auto result = m_textureState.emplace(image, TextureState());
	// no need to reimport if the image didn't change
	if (!result.second)
		return;
	image->ReferenceReleasePoints( m_releasePointRefs );
	// using the swapchain image as a source without writing to it doesn't make any sense
	assert(image->outputImage() == false);
	result.first->second.needsImport =
		image->queueFamily == external_queue_family( m_device );
	result.first->second.needsExport = image->externalImage();
}

void CVulkanCmdBuffer::prepareDestImage(CVulkanTexture *image, bool discard)
{
	auto result = m_textureState.emplace(image, TextureState());
	// no need to discard if the image is already image/in the correct layout
	if (!result.second)
		return;
	image->ReferenceReleasePoints( m_releasePointRefs );
	result.first->second.discarded = discard;
	result.first->second.needsImport =
		image->queueFamily == external_queue_family( m_device );
	result.first->second.needsExport = image->externalImage();
	result.first->second.needsPresentLayout = image->outputImage();
}

void CVulkanCmdBuffer::discardImage(CVulkanTexture *image)
{
	auto result = m_textureState.emplace(image, TextureState());
	if (!result.second)
		return;
	result.first->second.discarded = true;
}

void CVulkanCmdBuffer::markDirty(CVulkanTexture *image)
{
	++image->contentVersion;
	image->compositeState.reset();
	auto result = m_textureState.find(image);
	// image should have been prepared already
	assert(result !=  m_textureState.end());
	result->second.dirty = true;
}

void CVulkanCmdBuffer::insertBarrier(bool flush)
{
	std::vector<VkImageMemoryBarrier> barriers;
	std::vector<VkBufferMemoryBarrier> bufferBarriers;

	uint32_t externalQueue = external_queue_family( m_device );

	VkImageSubresourceRange subResRange =
	{
		.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
		.levelCount = 1,
		.layerCount = 1
	};

	for (auto& pair : m_textureState)
	{
		CVulkanTexture *image = pair.first;
		TextureState& state = pair.second;
		assert(!flush || !state.needsImport);

		bool isExport = flush && state.needsExport;
		bool isPresent = flush && state.needsPresentLayout;

		if (!state.discarded && !state.dirty && !state.needsImport && !isExport && !isPresent)
			continue;

		const VkAccessFlags write_bits = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		const VkAccessFlags read_bits = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;

		if (image->queueFamily == VK_QUEUE_FAMILY_IGNORED)
			image->queueFamily = m_queueFamily;

		VkImageMemoryBarrier memoryBarrier =
		{
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.srcAccessMask = state.dirty ? write_bits : 0u,
			.dstAccessMask = (isExport || isPresent) ? 0u : read_bits | write_bits,
			.oldLayout = state.discarded ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL,
			.newLayout = isPresent ? GetBackend()->GetPresentLayout() : VK_IMAGE_LAYOUT_GENERAL,
			.srcQueueFamilyIndex = isExport ? image->queueFamily : state.needsImport ? externalQueue : image->queueFamily,
			.dstQueueFamilyIndex = isExport ? externalQueue : state.needsImport ? m_queueFamily : m_queueFamily,
			.image = image->vkImage(),
			.subresourceRange = subResRange
		};

		if (image->m_linearBuffer)
			bufferBarriers.push_back({
				.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
				.srcAccessMask = 0, .dstAccessMask = isExport ? 0u : VkAccessFlags(VK_ACCESS_SHADER_READ_BIT),
				.srcQueueFamilyIndex = memoryBarrier.srcQueueFamilyIndex,
				.dstQueueFamilyIndex = memoryBarrier.dstQueueFamilyIndex,
				.buffer = image->m_linearBuffer, .offset = 0, .size = VK_WHOLE_SIZE,
			});
		barriers.push_back(memoryBarrier);
		image->queueFamily = memoryBarrier.dstQueueFamilyIndex;

		state.discarded = false;
		state.dirty = false;
		state.needsImport = false;
	}

	// TODO replace VK_PIPELINE_STAGE_ALL_COMMANDS_BIT
	m_device->vk.CmdPipelineBarrier(m_cmdBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
									0, 0, nullptr, bufferBarriers.size(), bufferBarriers.data(), barriers.size(), barriers.data());
}

CVulkanDevice g_device;

static bool allDMABUFsEqual( wlr_dmabuf_attributes *pDMA )
{
	if ( pDMA->n_planes == 1 )
		return true;

	struct stat first_stat;
	if ( fstat( pDMA->fd[0], &first_stat ) != 0 )
	{
		vk_log.errorf_errno( "fstat failed" );
		return false;
	}

	for ( int i = 1; i < pDMA->n_planes; ++i )
	{
		struct stat plane_stat;
		if ( fstat( pDMA->fd[i], &plane_stat ) != 0 )
		{
			vk_log.errorf_errno( "fstat failed" );
			return false;
		}
		if ( plane_stat.st_ino != first_stat.st_ino )
			return false;
	}

	return true;
}

static VkResult getModifierProps( const VkImageCreateInfo *imageInfo, uint64_t modifier, VkExternalImageFormatProperties *externalFormatProps)
{
	VkPhysicalDeviceImageDrmFormatModifierInfoEXT modifierFormatInfo = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_DRM_FORMAT_MODIFIER_INFO_EXT,
		.drmFormatModifier = modifier,
		.sharingMode = imageInfo->sharingMode,
	};

	VkPhysicalDeviceExternalImageFormatInfo externalImageFormatInfo = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO,
		.pNext = &modifierFormatInfo,
		.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
	};

	VkPhysicalDeviceImageFormatInfo2 imageFormatInfo = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
		.pNext = &externalImageFormatInfo,
		.format = imageInfo->format,
		.type = imageInfo->imageType,
		.tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
		.usage = imageInfo->usage,
		.flags = imageInfo->flags,
	};

	const VkImageFormatListCreateInfo *readonlyList = pNextFind<VkImageFormatListCreateInfo>(imageInfo, VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO);
	VkImageFormatListCreateInfo formatList = {};
	if ( readonlyList != nullptr )
	{
		formatList = *readonlyList;
		formatList.pNext = std::exchange(imageFormatInfo.pNext, &formatList);
	}

	VkImageFormatProperties2 imageProps = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
		.pNext = externalFormatProps,
	};

	return g_device.vk.GetPhysicalDeviceImageFormatProperties2(g_device.physDev(), &imageFormatInfo, &imageProps);
}

static VkImageViewType VulkanImageTypeToViewType(VkImageType type)
{
	switch (type)
	{
		case VK_IMAGE_TYPE_1D: return VK_IMAGE_VIEW_TYPE_1D;
		case VK_IMAGE_TYPE_2D: return VK_IMAGE_VIEW_TYPE_2D;
		case VK_IMAGE_TYPE_3D: return VK_IMAGE_VIEW_TYPE_3D;
		default: abort();
	}
}

static VkFormat typed_storage_view_format( VkFormat format )
{
	format = ToLinearVulkanFormat( format );
	if ( g_device.supportsStorageImageWriteWithoutFormat() )
		return format;
	if ( format == VK_FORMAT_B8G8R8A8_UNORM )
		return VK_FORMAT_R8G8B8A8_UNORM;
	if ( format == VK_FORMAT_A2R10G10B10_UNORM_PACK32 )
		return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
	return format;
}

bool CVulkanTexture::BInitStorageView()
{
	const VkFormat format = typed_storage_view_format( m_format );
	if ( format == ToLinearVulkanFormat( m_format ) )
		return true;
	VkImageViewUsageCreateInfo usage = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
		.usage = VK_IMAGE_USAGE_STORAGE_BIT,
	};
	VkImageViewCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.pNext = &usage,
		.image = m_vkImage,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = format,
		.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
	};
	VkResult res = g_device.vk.CreateImageView(g_device.device(), &info, nullptr, &m_storageView);
	if ( res != VK_SUCCESS )
		vk_errorf(res, "vkCreateImageView for typed storage failed");
	return res == VK_SUCCESS;
}

bool CVulkanTexture::BInit( uint32_t width, uint32_t height, uint32_t depth, uint32_t drmFormat, createFlags flags, wlr_dmabuf_attributes *pDMA /* = nullptr */,  uint32_t contentWidth /* = 0 */, uint32_t contentHeight /* =  0 */, CVulkanTexture *pExistingImageToReuseMemory, gamescope::OwningRc<gamescope::IBackendFb> pBackendFb )
{
	m_pBackendFb = std::move( pBackendFb );
	m_drmFormat = drmFormat;
	m_bColorAttachment = flags.bColorAttachment;
	m_bStorage = flags.bStorage;
	VkResult res = VK_ERROR_INITIALIZATION_FAILED;
	// Forced composition only scans out backend-allocated outputs. Sampled
	// intermediates and client uploads need no separate KMS allocation.
	if ( GetBackend()->UsesBackendAllocatedScanout() && flags.bSampled && !flags.bOutputImage )
		flags.bFlippable = false;

	VkImageTiling tiling = (flags.bMappable || flags.bLinear) ? VK_IMAGE_TILING_LINEAR : VK_IMAGE_TILING_OPTIMAL;
	VkImageUsageFlags usage = 0;
	VkMemoryPropertyFlags properties;

	if ( flags.bSampled == true )
	{
		usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
	}

	if ( flags.bStorage == true )
	{
		usage |= VK_IMAGE_USAGE_STORAGE_BIT;
	}

	if ( flags.bColorAttachment == true )
	{
		usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
	}

	if ( flags.bFlippable == true && pDMA == nullptr )
	{
		flags.bExportable = true;
	}

	if ( flags.bReadback == true )
	{
		flags.bTransferSrc = true;
	}

	if ( flags.bTransferSrc == true )
	{
		usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
	}

	if ( flags.bTransferDst == true )
	{
		usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	}

	m_usage = usage;

	if ( flags.bMappable == true )
	{
		properties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
	}
	else
	{
		properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	}

	if ( flags.bOutputImage == true )
	{
		m_bOutputImage = true;	
	}

	m_bExternal = pDMA || flags.bExportable == true;
	// Imported DMA-BUFs begin in the external producer's ownership. Locally
	// allocated exportable images remain unowned until their first Vulkan use.
	queueFamily = pDMA
		? external_queue_family( &g_device )
		: VK_QUEUE_FAMILY_IGNORED;

	// Possible extensions for below
	wsi_image_create_info wsiImageCreateInfo = {};
	VkExternalMemoryImageCreateInfo externalImageCreateInfo = {};
	VkImageDrmFormatModifierExplicitCreateInfoEXT modifierInfo = {};
	VkSubresourceLayout modifierPlaneLayouts[4] = {};
	VkImageDrmFormatModifierListCreateInfoEXT modifierListInfo = {};
	
	VkImageCreateInfo imageInfo = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType = flags.imageType,
		.format = DRMFormatToVulkan(drmFormat, false),
		.extent = {
			.width = width,
			.height = height,
			.depth = depth,
		},
		.mipLevels = 1,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = tiling,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	assert( imageInfo.format != VK_FORMAT_UNDEFINED );

	std::vector<VkFormat> formats = {
		DRMFormatToVulkan(drmFormat, false),
		DRMFormatToVulkan(drmFormat, true),
	};

	if ( flags.bStorage && typed_storage_view_format(imageInfo.format) != imageInfo.format )
		formats.push_back(typed_storage_view_format(imageInfo.format));

	VkImageFormatListCreateInfo formatList = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO,
		.viewFormatCount = (uint32_t)formats.size(),
		.pViewFormats = formats.data(),
	};

	if ( formats[0] != formats[1] || formats.size() > 2 )
	{
		formatList.pNext = std::exchange(imageInfo.pNext, &formatList);
		imageInfo.flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
	}

	if ( pDMA != nullptr )
	{
		assert( drmFormat == pDMA->format );
	}

	if ( g_device.supportsModifiers() && pDMA && pDMA->modifier != DRM_FORMAT_MOD_INVALID )
	{
		VkExternalImageFormatProperties externalImageProperties = {
			.sType = VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES,
		};

		res = getModifierProps( &imageInfo, pDMA->modifier, &externalImageProperties );
		if ( res != VK_SUCCESS && res != VK_ERROR_FORMAT_NOT_SUPPORTED ) {
			vk_errorf( res, "getModifierProps failed" );
			return false;
		}

		if ( res == VK_SUCCESS &&
		     ( externalImageProperties.externalMemoryProperties.externalMemoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT ) )
		{
			modifierInfo = {
				.sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT,
				.pNext = std::exchange(imageInfo.pNext, &modifierInfo),
				.drmFormatModifier = pDMA->modifier,
				.drmFormatModifierPlaneCount = uint32_t(pDMA->n_planes),
				.pPlaneLayouts = modifierPlaneLayouts,
			};

			for ( int i = 0; i < pDMA->n_planes; ++i )
			{
				modifierPlaneLayouts[i].offset = pDMA->offset[i];
				modifierPlaneLayouts[i].rowPitch = pDMA->stride[i];
			}

			imageInfo.tiling = tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
		}
	}

	std::vector<uint64_t> modifiers = {};
	// TODO(JoshA): Move this code to backend for making flippable image.
	if ( GetBackend()->UsesModifiers() && flags.bFlippable && g_device.supportsModifiers() && !pDMA )
	{
		assert( drmFormat != DRM_FORMAT_INVALID );

		uint64_t linear = DRM_FORMAT_MOD_LINEAR;

		const uint64_t *possibleModifiers;
		size_t numPossibleModifiers;
		if ( flags.bLinear )
		{
			possibleModifiers = &linear;
			numPossibleModifiers = 1;
		}
		else
		{
			std::span<const uint64_t> modifiers = GetBackend()->GetSupportedModifiers( drmFormat );
			assert( !modifiers.empty() );
			possibleModifiers = modifiers.data();
			numPossibleModifiers = modifiers.size();
		}

		for ( size_t i = 0; i < numPossibleModifiers; i++ )
		{
			uint64_t modifier = possibleModifiers[i];

			VkExternalImageFormatProperties externalFormatProps = {
				.sType = VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES,
			};
			res = getModifierProps( &imageInfo, modifier, &externalFormatProps );
			if ( res == VK_ERROR_FORMAT_NOT_SUPPORTED )
				continue;
			else if ( res != VK_SUCCESS ) {
				vk_errorf( res, "getModifierProps failed" );
				return false;
			}

			if ( !( externalFormatProps.externalMemoryProperties.externalMemoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT ) )
				continue;

			modifiers.push_back( modifier );
		}

		assert( modifiers.size() > 0 );

		modifierListInfo = {
			.sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT,
			.pNext = std::exchange(imageInfo.pNext, &modifierListInfo),
			.drmFormatModifierCount = uint32_t(modifiers.size()),
			.pDrmFormatModifiers = modifiers.data(),
		};

		externalImageCreateInfo = {
			.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
			.pNext = std::exchange(imageInfo.pNext, &externalImageCreateInfo),
			.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
		};

		imageInfo.tiling = tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
	}

	if ( flags.bFlippable == true && tiling != VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT )
	{
		// We want to scan-out the image
		wsiImageCreateInfo = {
			.sType = VK_STRUCTURE_TYPE_WSI_IMAGE_CREATE_INFO_MESA,
			.pNext = std::exchange(imageInfo.pNext, &wsiImageCreateInfo),
			.scanout = VK_TRUE,
		};
	}
	
	if ( pDMA != nullptr )
	{
		externalImageCreateInfo = {
			.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
			.pNext = std::exchange(imageInfo.pNext, &externalImageCreateInfo),
			.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
		};
	}

	m_width = width;
	m_height = height;
	m_depth = depth;

	if (contentWidth && contentHeight)
	{
		m_contentWidth = contentWidth;
		m_contentHeight = contentHeight;
	}
	else
	{
		m_contentWidth = width;
		m_contentHeight = height;
	}

	m_format = imageInfo.format;

	res = g_device.vk.CreateImage(g_device.device(), &imageInfo, nullptr, &m_vkImage);
	if (res != VK_SUCCESS) {
		vk_errorf( res, "vkCreateImage failed" );
		return false;
	}
	m_bOwnsImage = true;
	
	VkMemoryRequirements memRequirements;
	g_device.vk.GetImageMemoryRequirements(g_device.device(), m_vkImage, &memRequirements);

	VkMemoryAllocateInfo allocInfo = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize = memRequirements.size,
		.memoryTypeIndex = uint32_t(g_device.findMemoryType(properties, memRequirements.memoryTypeBits)),
	};

	m_size = allocInfo.allocationSize;

	VkDeviceMemory memoryHandle = VK_NULL_HANDLE;

	if ( pExistingImageToReuseMemory == nullptr )
	{
		// Possible pNexts
		VkImportMemoryFdInfoKHR importMemoryInfo = {};
		VkExportMemoryAllocateInfo memory_export_info = {};
		VkMemoryDedicatedAllocateInfo memory_dedicated_info = {};
		struct wsi_memory_allocate_info memory_wsi_info = {};

		if ( flags.bFlippable == true && !( pDMA && g_device.isV3dv() ) )
		{
			memory_wsi_info = {
				.sType = VK_STRUCTURE_TYPE_WSI_MEMORY_ALLOCATE_INFO_MESA,
				.pNext = std::exchange(allocInfo.pNext, &memory_wsi_info),
			};
		}

		if ( flags.bExportable == true || pDMA != nullptr )
		{
			memory_dedicated_info = {
				.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
				.pNext = std::exchange(allocInfo.pNext, &memory_dedicated_info),
				.image = m_vkImage,
			};
		}
		
		if ( flags.bExportable == true && pDMA == nullptr )
		{
			// We'll export it to DRM
			memory_export_info = {
				.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
				.pNext = std::exchange(allocInfo.pNext, &memory_export_info),
				.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
			};
		}
		
		if ( pDMA != nullptr )
		{
			// TODO: multi-planar DISTINCT DMA-BUFs support (see vkBindImageMemory2
			// and VkBindImagePlaneMemoryInfo)
			assert( allDMABUFsEqual( pDMA ) );

			// Importing memory from a FD transfers ownership of the FD
			int fd = dup( pDMA->fd[0] );
			if ( fd < 0 )
			{
				vk_log.errorf_errno( "dup failed" );
				return false;
			}

			VkMemoryFdPropertiesKHR fdProperties = {
				.sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR,
			};
			res = g_device.vk.GetMemoryFdPropertiesKHR(
				g_device.device(), VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
				fd, &fdProperties );
			if ( res != VK_SUCCESS )
			{
				vk_errorf( res, "vkGetMemoryFdPropertiesKHR failed" );
				close( fd );
				return false;
			}

			const uint32_t uMemoryTypeBits = memRequirements.memoryTypeBits & fdProperties.memoryTypeBits;
			int32_t nMemoryType = g_device.findMemoryType( properties, uMemoryTypeBits );
			if ( nMemoryType < 0 )
				nMemoryType = g_device.findMemoryType( 0, uMemoryTypeBits );
			if ( nMemoryType < 0 )
			{
				vk_log.errorf( "No compatible Vulkan memory type for scanout DMA-BUF import" );
				close( fd );
				return false;
			}
			allocInfo.memoryTypeIndex = uint32_t( nMemoryType );

			// Memory already provided by pDMA
			importMemoryInfo = {
					.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
					.pNext = std::exchange(allocInfo.pNext, &importMemoryInfo),
					.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
					.fd = fd,
			};
		}
		
		res = g_device.vk.AllocateMemory( g_device.device(), &allocInfo, nullptr, &memoryHandle );
		if ( res != VK_SUCCESS )
		{
			if ( pDMA != nullptr )
				close( importMemoryInfo.fd );
			vk_errorf( res, "vkAllocateMemory failed" );
			return false;
		}

		m_vkImageMemory = memoryHandle;
	}
	else
	{
		vk_log.infof("%d vs %d!", (int)pExistingImageToReuseMemory->m_size, (int)m_size);
		assert(pExistingImageToReuseMemory->m_size >= m_size);

		memoryHandle = pExistingImageToReuseMemory->m_vkImageMemory;
		m_vkImageMemory = VK_NULL_HANDLE;
	}
	
	res = g_device.vk.BindImageMemory( g_device.device(), m_vkImage, memoryHandle, 0 );
	if ( res != VK_SUCCESS )
	{
		vk_errorf( res, "vkBindImageMemory failed" );
		return false;
	}

	if ( flags.bMappable == true )
	{
		assert( tiling != VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT );
		const VkImageSubresource image_subresource = {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
		};
		VkSubresourceLayout image_layout;
		g_device.vk.GetImageSubresourceLayout(g_device.device(), m_vkImage, &image_subresource, &image_layout);

		m_unRowPitch = image_layout.rowPitch;

		if (isYcbcr())
		{
			const VkImageSubresource lumaSubresource = {
				.aspectMask = VK_IMAGE_ASPECT_PLANE_0_BIT,
			};
			VkSubresourceLayout lumaLayout;
			g_device.vk.GetImageSubresourceLayout(g_device.device(), m_vkImage, &lumaSubresource, &lumaLayout);

			m_lumaOffset = lumaLayout.offset;
			m_lumaPitch = lumaLayout.rowPitch;

			const VkImageSubresource chromaSubresource = {
				.aspectMask = VK_IMAGE_ASPECT_PLANE_1_BIT,
			};
			VkSubresourceLayout chromaLayout;
			g_device.vk.GetImageSubresourceLayout(g_device.device(), m_vkImage, &chromaSubresource, &chromaLayout);

			m_chromaOffset = chromaLayout.offset;
			m_chromaPitch = chromaLayout.rowPitch;
		}
	}
	
	if ( flags.bExportable == true )
	{
		// We assume we own the memory when doing this right now.
		// We could support the import scenario as well if needed (but we
		// already have a DMA-BUF in that case).
		assert( pDMA == nullptr );

		struct wlr_dmabuf_attributes dmabuf = {
			.width = int(width),
			.height = int(height),
			.format = drmFormat,
		};
		assert( dmabuf.format != DRM_FORMAT_INVALID );

		// TODO: disjoint planes support
		const VkMemoryGetFdInfoKHR memory_get_fd_info = {
			.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
			.memory = memoryHandle,
			.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
		};
		res = g_device.vk.GetMemoryFdKHR(g_device.device(), &memory_get_fd_info, &dmabuf.fd[0]);
		if ( res != VK_SUCCESS ) {
			vk_errorf( res, "vkGetMemoryFdKHR failed" );
			return false;
		}

		if ( tiling == VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT )
		{
			assert( g_device.vk.GetImageDrmFormatModifierPropertiesEXT != nullptr );

			VkImageDrmFormatModifierPropertiesEXT imgModifierProps = {
				.sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_PROPERTIES_EXT,
			};

			res = g_device.vk.GetImageDrmFormatModifierPropertiesEXT( g_device.device(), m_vkImage, &imgModifierProps );
			if ( res != VK_SUCCESS ) {
				vk_errorf( res, "vkGetImageDrmFormatModifierPropertiesEXT failed" );
				return false;
			}
			dmabuf.modifier = imgModifierProps.drmFormatModifier;

			assert( DRMModifierProps.count( m_format ) > 0);
			assert( DRMModifierProps[ m_format ].count( dmabuf.modifier ) > 0);

			dmabuf.n_planes = DRMModifierProps[ m_format ][ dmabuf.modifier ].drmFormatModifierPlaneCount;

			const VkImageAspectFlagBits planeAspects[] = {
				VK_IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT,
				VK_IMAGE_ASPECT_MEMORY_PLANE_1_BIT_EXT,
				VK_IMAGE_ASPECT_MEMORY_PLANE_2_BIT_EXT,
				VK_IMAGE_ASPECT_MEMORY_PLANE_3_BIT_EXT,
			};
			assert( dmabuf.n_planes <= 4 );

			for ( int i = 0; i < dmabuf.n_planes; i++ )
			{
				const VkImageSubresource subresource = {
					.aspectMask = planeAspects[i],
				};
				VkSubresourceLayout subresourceLayout = {};
				g_device.vk.GetImageSubresourceLayout( g_device.device(), m_vkImage, &subresource, &subresourceLayout );
				dmabuf.offset[i] = subresourceLayout.offset;
				dmabuf.stride[i] = subresourceLayout.rowPitch;
			}

			// Copy the first FD to all other planes
			for ( int i = 1; i < dmabuf.n_planes; i++ )
			{
				dmabuf.fd[i] = dup( dmabuf.fd[0] );
				if ( dmabuf.fd[i] < 0 ) {
					vk_log.errorf_errno( "dup failed" );
					return false;
				}
			}
		}
		else
		{
			const VkImageSubresource subresource = {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			};
			VkSubresourceLayout subresourceLayout = {};
			g_device.vk.GetImageSubresourceLayout( g_device.device(), m_vkImage, &subresource, &subresourceLayout );

			dmabuf.n_planes = 1;
			dmabuf.modifier = DRM_FORMAT_MOD_INVALID;
			dmabuf.offset[0] = 0;
			dmabuf.stride[0] = subresourceLayout.rowPitch;
		}

		m_dmabuf = dmabuf;
	}
	else if ( pDMA != nullptr && (flags.bFlippable || (flags.bTransferSrc && !flags.bSampled)) )
	{
		m_dmabuf = *pDMA;
		m_dmabuf.n_planes = 0;
		for ( int i = 0; i < pDMA->n_planes; i++ )
		{
			m_dmabuf.fd[i] = dup( pDMA->fd[i] );
			if ( m_dmabuf.fd[i] < 0 )
			{
				vk_log.errorf_errno( "dup failed" );
				return false;
			}
			m_dmabuf.n_planes = i + 1;
		}
	}

	if ( flags.bFlippable == true )
	{
		m_pBackendFb = GetBackend()->ImportDmabufToBackend( &m_dmabuf );
		if ( pDMA != nullptr && m_pBackendFb == nullptr )
			return false;
	}

	bool bHasAlpha = !flags.bOutputImage && !flags.bStorage ? DRMFormatHasAlpha(drmFormat) : true;

	if (!bHasAlpha )
	{
		// not compatible with with swizzles
		assert ( flags.bStorage == false );
	}

	if ( flags.bStorage || flags.bSampled || flags.bColorAttachment )
	{
		VkImageViewCreateInfo createInfo = {
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.image = m_vkImage,
			.viewType = VulkanImageTypeToViewType(flags.imageType),
			.format = DRMFormatToVulkan(drmFormat, false),
			.components = {
				.r = VK_COMPONENT_SWIZZLE_IDENTITY,
				.g = VK_COMPONENT_SWIZZLE_IDENTITY,
				.b = VK_COMPONENT_SWIZZLE_IDENTITY,
				.a = bHasAlpha ? VK_COMPONENT_SWIZZLE_IDENTITY : VK_COMPONENT_SWIZZLE_ONE,
			},
			.subresourceRange = {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.levelCount = 1,
				.layerCount = 1,
			},
		};

		res = g_device.vk.CreateImageView(g_device.device(), &createInfo, nullptr, &m_srgbView);
		if ( res != VK_SUCCESS ) {
			vk_errorf( res, "vkCreateImageView failed" );
			return false;
		}

		if ( flags.bStorage && !BInitStorageView() )
			return false;

		if ( flags.bSampled || flags.bColorAttachment )
		{
			VkImageViewUsageCreateInfo viewUsageInfo = {
				.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
				.usage = usage & ~VK_IMAGE_USAGE_STORAGE_BIT,
			};
			createInfo.pNext = &viewUsageInfo;
			createInfo.format = DRMFormatToVulkan(drmFormat, true);
			res = g_device.vk.CreateImageView(g_device.device(), &createInfo, nullptr, &m_linearView);
			if ( res != VK_SUCCESS ) {
				vk_errorf( res, "vkCreateImageView failed" );
				return false;
			}
		}


		if ( isYcbcr() )
		{
			createInfo.pNext = NULL;
			createInfo.format = VK_FORMAT_R8_UNORM;

			createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_PLANE_0_BIT;
			res = g_device.vk.CreateImageView(g_device.device(), &createInfo, nullptr, &m_lumaView);
			if ( res != VK_SUCCESS ) {
				vk_errorf( res, "vkCreateImageView failed" );
				return false;
			}

			createInfo.pNext = NULL;
			createInfo.format = VK_FORMAT_R8G8_UNORM;
			createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_PLANE_1_BIT;
			res = g_device.vk.CreateImageView(g_device.device(), &createInfo, nullptr, &m_chromaView);
			if ( res != VK_SUCCESS ) {
				vk_errorf( res, "vkCreateImageView failed" );
				return false;
			}

			createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		}
	}

	if ( flags.bMappable )
	{
		if (pExistingImageToReuseMemory)
		{
			m_pMappedData = pExistingImageToReuseMemory->m_pMappedData;
		}
		else
		{
			void *pData = nullptr;
			res = g_device.vk.MapMemory( g_device.device(), memoryHandle, 0, VK_WHOLE_SIZE, 0, &pData );
			if ( res != VK_SUCCESS )
			{
				vk_errorf( res, "vkMapMemory failed" );
				return false;
			}
			m_pMappedData = (uint8_t*)pData;
		}
	}

	if ( flags.bReadback )
	{
		VkDeviceSize readbackSize;
		if ( isYcbcr() )
		{
			m_lumaOffset = 0;
			m_lumaPitch = width;
			m_chromaOffset = width * height;
			m_chromaPitch = ((width + 1) / 2) * 2;
			readbackSize = VkDeviceSize(width) * height + VkDeviceSize(m_chromaPitch) * ((height + 1) / 2);
		}
		else
		{
			m_unRowPitch = width * DRMFormatGetBPP( drmFormat );
			readbackSize = VkDeviceSize(m_unRowPitch) * height;
		}

		VkBufferCreateInfo bufferCreateInfo = {
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = readbackSize,
			.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		};
		res = g_device.vk.CreateBuffer( g_device.device(), &bufferCreateInfo, nullptr, &m_vkReadbackBuffer );
		if ( res != VK_SUCCESS )
		{
			vk_errorf( res, "vkCreateBuffer failed for readback buffer" );
			return false;
		}

		VkMemoryRequirements bufMemRequirements;
		g_device.vk.GetBufferMemoryRequirements( g_device.device(), m_vkReadbackBuffer, &bufMemRequirements );

		int32_t memTypeIndex = g_device.findMemoryType( VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, bufMemRequirements.memoryTypeBits );
		if ( memTypeIndex == -1 )
			memTypeIndex = g_device.findMemoryType( VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, bufMemRequirements.memoryTypeBits );
		if ( memTypeIndex == -1 )
		{
			vk_log.errorf( "findMemoryType failed for readback buffer" );
			return false;
		}

		VkMemoryAllocateInfo readbackAllocInfo = {
			.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
			.allocationSize = bufMemRequirements.size,
			.memoryTypeIndex = uint32_t(memTypeIndex),
		};
		res = g_device.vk.AllocateMemory( g_device.device(), &readbackAllocInfo, nullptr, &m_vkReadbackMemory );
		if ( res != VK_SUCCESS )
		{
			vk_errorf( res, "vkAllocateMemory failed for readback buffer" );
			return false;
		}

		res = g_device.vk.BindBufferMemory( g_device.device(), m_vkReadbackBuffer, m_vkReadbackMemory, 0 );
		if ( res != VK_SUCCESS )
		{
			vk_errorf( res, "vkBindBufferMemory failed for readback buffer" );
			return false;
		}

		void *pData = nullptr;
		res = g_device.vk.MapMemory( g_device.device(), m_vkReadbackMemory, 0, VK_WHOLE_SIZE, 0, &pData );
		if ( res != VK_SUCCESS )
		{
			vk_errorf( res, "vkMapMemory failed for readback buffer" );
			return false;
		}
		m_pMappedData = (uint8_t*)pData;
	}
	
	m_bInitialized = true;
	
	return true;
}

bool CVulkanTexture::BInitFromSwapchain( VkImage image, uint32_t width, uint32_t height, VkFormat format )
{
	m_drmFormat = VulkanFormatToDRM( format );
	m_vkImage = image;
	m_vkImageMemory = VK_NULL_HANDLE;
	m_width = width;
	m_height = height;
	m_depth = 1;
	m_format = format;
	m_contentWidth = width;
	m_contentHeight = height;
	m_bOutputImage = true;

	VkImageViewCreateInfo createInfo = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = image,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = ToLinearVulkanFormat( format ),
		.components = {
			.r = VK_COMPONENT_SWIZZLE_IDENTITY,
			.g = VK_COMPONENT_SWIZZLE_IDENTITY,
			.b = VK_COMPONENT_SWIZZLE_IDENTITY,
			.a = VK_COMPONENT_SWIZZLE_IDENTITY,
		},
		.subresourceRange = {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.levelCount = 1,
			.layerCount = 1,
		},
	};

	VkResult res = g_device.vk.CreateImageView(g_device.device(), &createInfo, nullptr, &m_srgbView);
	if ( res != VK_SUCCESS ) {
		vk_errorf( res, "vkCreateImageView failed" );
		return false;
	}

	if ( !BInitStorageView() )
		return false;

	VkImageViewUsageCreateInfo viewUsageInfo = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
		.usage = VK_IMAGE_USAGE_SAMPLED_BIT,
	};

	createInfo.pNext = &viewUsageInfo;
	createInfo.format = ToSrgbVulkanFormat( format );

	res = g_device.vk.CreateImageView(g_device.device(), &createInfo, nullptr, &m_linearView);
	if ( res != VK_SUCCESS ) {
		vk_errorf( res, "vkCreateImageView failed" );
		return false;
	}

	m_bInitialized = true;

	return true;
}

uint32_t CVulkanTexture::IncRef()
{
	uint32_t uRefCount = gamescope::RcObject::IncRef();
	if ( m_pBackendFb && !uRefCount )
	{
		m_pBackendFb->IncRef();
	}
	return uRefCount;
}
uint32_t CVulkanTexture::DecRef()
{
	// Need to pull it out as we could be destroyed in DecRef.
	gamescope::IBackendFb *pBackendFb = m_pBackendFb.get();

	uint32_t uRefCount = gamescope::RcObject::DecRef();
	if ( pBackendFb && !uRefCount )
	{
		pBackendFb->DecRef();
	}
	return uRefCount;
}

bool CVulkanTexture::IsInUse()
{
	if ( m_pBackendFb && m_pBackendFb->GetRefCount() != 0 )
		return true;

	return GetRefCount() != 0;
}

CVulkanTexture::CVulkanTexture( void )
{
	static std::atomic<uint64_t> nextID = {0};
	textureID = ++nextID;
}

VkBuffer CVulkanTexture::linearBuffer()
{
	if (m_linearBufferTried)
		return m_linearBuffer;
	m_linearBufferTried = true;
	if (m_dmabuf.modifier != DRM_FORMAT_MOD_LINEAR || m_dmabuf.n_planes != 1 ||
		(m_dmabuf.stride[0] % 4) || (m_dmabuf.offset[0] % 4) ||
		(m_format != VK_FORMAT_R8G8B8A8_UNORM && m_format != VK_FORMAT_B8G8R8A8_UNORM))
		return VK_NULL_HANDLE;
	const VkDeviceSize size = VkDeviceSize(m_dmabuf.offset[0]) + VkDeviceSize(m_dmabuf.stride[0]) * (m_height - 1) + VkDeviceSize(m_width) * 4;
	VkPhysicalDeviceProperties properties;
	g_device.vk.GetPhysicalDeviceProperties(g_device.physDev(), &properties);
	if (size > properties.limits.maxStorageBufferRange)
		return VK_NULL_HANDLE;
	VkPhysicalDeviceExternalBufferInfo externalInfo = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO,
		.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
	};
	VkExternalBufferProperties externalProperties = {.sType = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES};
	g_device.vk.GetPhysicalDeviceExternalBufferProperties(g_device.physDev(), &externalInfo, &externalProperties);
	if (!(externalProperties.externalMemoryProperties.externalMemoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT))
		return VK_NULL_HANDLE;
	VkExternalMemoryBufferCreateInfo external = {
		.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
		.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
	};
	VkBufferCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .pNext = &external,
		.size = size, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	VkBuffer buffer = VK_NULL_HANDLE;
	if (g_device.vk.CreateBuffer(g_device.device(), &info, nullptr, &buffer) != VK_SUCCESS)
		return VK_NULL_HANDLE;
	VkMemoryRequirements requirements;
	g_device.vk.GetBufferMemoryRequirements(g_device.device(), buffer, &requirements);
	VkMemoryFdPropertiesKHR fdProperties = {.sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR};
	VkDeviceMemory memory = VK_NULL_HANDLE;
	struct stat backing{};
	int fd = dup(m_dmabuf.fd[0]);
	if (fd >= 0 && fstat(fd, &backing) == 0 && backing.st_size > 0 &&
		requirements.size <= VkDeviceSize(backing.st_size) &&
		g_device.vk.GetMemoryFdPropertiesKHR(g_device.device(), VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, fd, &fdProperties) == VK_SUCCESS)
	{
		int type = g_device.findMemoryType(0, requirements.memoryTypeBits & fdProperties.memoryTypeBits);
		VkMemoryDedicatedAllocateInfo dedicated = {.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, .buffer = buffer};
		VkImportMemoryFdInfoKHR import = {
			.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR, .pNext = &dedicated,
			.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, .fd = fd,
		};
		VkMemoryAllocateInfo allocation = {
			.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &import,
			.allocationSize = requirements.size, .memoryTypeIndex = uint32_t(type),
		};
		if (type >= 0 && g_device.vk.AllocateMemory(g_device.device(), &allocation, nullptr, &memory) == VK_SUCCESS)
			fd = -1; // Vulkan consumed the duplicate.
	}
	if (fd >= 0) close(fd);
	if (memory && g_device.vk.BindBufferMemory(g_device.device(), buffer, memory, 0) == VK_SUCCESS)
	{
		m_linearBuffer = buffer;
		m_linearBufferMemory = memory;
		return buffer;
	}
	g_device.vk.DestroyBuffer(g_device.device(), buffer, nullptr);
	if (memory) g_device.vk.FreeMemory(g_device.device(), memory, nullptr);
	return VK_NULL_HANDLE;
}

CVulkanTexture::~CVulkanTexture( void )
{
	if (m_linearBuffer) g_device.vk.DestroyBuffer(g_device.device(), m_linearBuffer, nullptr);
	if (m_linearBufferMemory) g_device.vk.FreeMemory(g_device.device(), m_linearBufferMemory, nullptr);
	wlr_dmabuf_attributes_finish( &m_dmabuf );

	if ( m_vkReadbackBuffer != VK_NULL_HANDLE )
	{
		g_device.vk.DestroyBuffer( g_device.device(), m_vkReadbackBuffer, nullptr );
		m_vkReadbackBuffer = VK_NULL_HANDLE;

		if ( m_pMappedData != nullptr )
		{
			g_device.vk.UnmapMemory( g_device.device(), m_vkReadbackMemory );
			m_pMappedData = nullptr;
		}

		g_device.vk.FreeMemory( g_device.device(), m_vkReadbackMemory, nullptr );
		m_vkReadbackMemory = VK_NULL_HANDLE;
	}

	if ( m_pMappedData != nullptr && m_vkImageMemory )
	{
		g_device.vk.UnmapMemory( g_device.device(), m_vkImageMemory );
		m_pMappedData = nullptr;
	}

	if ( m_storageView != VK_NULL_HANDLE )
	{
		g_device.vk.DestroyImageView( g_device.device(), m_storageView, nullptr );
		m_storageView = VK_NULL_HANDLE;
	}
	if ( m_srgbView != VK_NULL_HANDLE )
	{
		g_device.vk.DestroyImageView( g_device.device(), m_srgbView, nullptr );
		m_srgbView = VK_NULL_HANDLE;
	}

	if ( m_linearView != VK_NULL_HANDLE )
	{
		g_device.vk.DestroyImageView( g_device.device(), m_linearView, nullptr );
		m_linearView = VK_NULL_HANDLE;
	}

	if ( m_pBackendFb != nullptr )
		m_pBackendFb = nullptr;

	if ( m_vkImage != VK_NULL_HANDLE && m_bOwnsImage )
	{
		g_device.vk.DestroyImage( g_device.device(), m_vkImage, nullptr );
		m_vkImage = VK_NULL_HANDLE;
		m_bOwnsImage = false;
	}

	if ( m_vkImageMemory != VK_NULL_HANDLE )
	{
		g_device.vk.FreeMemory( g_device.device(), m_vkImageMemory, nullptr );
		m_vkImageMemory = VK_NULL_HANDLE;
	}

	m_bInitialized = false;
}

int CVulkanTexture::memoryFence()
{
	const VkMemoryGetFdInfoKHR memory_get_fd_info = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
		.memory = m_vkImageMemory,
		.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
	};
	int fence = -1;
	VkResult res = g_device.vk.GetMemoryFdKHR(g_device.device(), &memory_get_fd_info, &fence);
	if ( res != VK_SUCCESS ) {
		fprintf( stderr, "vkGetMemoryFdKHR failed\n" );
	}

	return fence;
}

void CVulkanTexture::AddReleasePoint( const std::shared_ptr<gamescope::CReleaseTimelinePoint> &pReleasePoint )
{
	if ( !pReleasePoint )
		return;

	std::lock_guard lock( m_ReleasePointMutex );
	m_ReleasePoints.emplace_back( pReleasePoint );
	m_bHasReleasePoints.store( true, std::memory_order_release );
}

void CVulkanTexture::ReferenceReleasePoints( std::vector<std::shared_ptr<gamescope::CReleaseTimelinePoint>> &references )
{
	if ( !m_bHasReleasePoints.load( std::memory_order_acquire ) )
		return;

	std::lock_guard lock( m_ReleasePointMutex );
	for ( auto it = m_ReleasePoints.begin(); it != m_ReleasePoints.end(); )
	{
		if ( std::shared_ptr<gamescope::CReleaseTimelinePoint> pReleasePoint = it->lock() )
		{
			references.emplace_back( std::move( pReleasePoint ) );
			++it;
		}
		else
		{
			it = m_ReleasePoints.erase( it );
		}
	}
	if ( m_ReleasePoints.empty() )
		m_bHasReleasePoints.store( false, std::memory_order_release );
}

static bool is_image_format_modifier_supported(VkFormat format, uint32_t drmFormat, uint64_t modifier)
{
  VkPhysicalDeviceImageFormatInfo2 imageFormatInfo = {
    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
    .format = format,
    .type = VK_IMAGE_TYPE_2D,
    .tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
    .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
  };

  std::array<VkFormat, 2> formats = {
    DRMFormatToVulkan(drmFormat, false),
    DRMFormatToVulkan(drmFormat, true),
  };

  VkImageFormatListCreateInfo formatList = {
    .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO,
    .viewFormatCount = (uint32_t)formats.size(),
    .pViewFormats = formats.data(),
  };

  if ( formats[0] != formats[1] )
    {
      formatList.pNext = std::exchange(imageFormatInfo.pNext,
				       &formatList);
      imageFormatInfo.flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
    }

  VkPhysicalDeviceImageDrmFormatModifierInfoEXT modifierInfo = {
    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_DRM_FORMAT_MODIFIER_INFO_EXT,
    .pNext = nullptr,
    .drmFormatModifier = modifier,
  };

  modifierInfo.pNext = std::exchange(imageFormatInfo.pNext, &modifierInfo);

  VkImageFormatProperties2 imageFormatProps = {
    .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
  };

  VkResult res = g_device.vk.GetPhysicalDeviceImageFormatProperties2( g_device.physDev(), &imageFormatInfo, &imageFormatProps );
  if (res == VK_ERROR_FORMAT_NOT_SUPPORTED && drmFormat != DRM_FORMAT_NV12)
  {
    // Import transfer-only RGB modifiers through the sampled shadow on any GPU.
    imageFormatInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    res = g_device.vk.GetPhysicalDeviceImageFormatProperties2( g_device.physDev(), &imageFormatInfo, &imageFormatProps );
  }
  return res == VK_SUCCESS;
}

bool vulkan_init_format(VkFormat format, uint32_t drmFormat)
{
	// First, check whether the Vulkan format is supported
	VkPhysicalDeviceImageFormatInfo2 imageFormatInfo = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
		.format = format,
		.type = VK_IMAGE_TYPE_2D,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = VK_IMAGE_USAGE_SAMPLED_BIT,
	};

	std::vector<VkFormat> formats = {
		DRMFormatToVulkan(drmFormat, false),
		DRMFormatToVulkan(drmFormat, true),
	};

	VkImageFormatListCreateInfo formatList = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO,
		.viewFormatCount = (uint32_t)formats.size(),
		.pViewFormats = formats.data(),
	};

	if ( formats[0] != formats[1] || formats.size() > 2 )
	{
		formatList.pNext = std::exchange(imageFormatInfo.pNext,
						 &formatList);
		imageFormatInfo.flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
	}


	VkImageFormatProperties2 imageFormatProps = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
	};

	VkResult res = g_device.vk.GetPhysicalDeviceImageFormatProperties2( g_device.physDev(), &imageFormatInfo, &imageFormatProps );
	if ( res == VK_ERROR_FORMAT_NOT_SUPPORTED )
	{
		return false;
	}
	else if ( res != VK_SUCCESS )
	{
		vk_errorf( res, "vkGetPhysicalDeviceImageFormatProperties2 failed for DRM format 0x%" PRIX32, drmFormat );
		return false;
	}

	wlr_drm_format_set_add( &sampledShmFormats, drmFormat, DRM_FORMAT_MOD_LINEAR );

	if ( g_device.supportsModifiers() )
	{
		// Then, collect the list of modifiers supported for sampled usage
		VkDrmFormatModifierPropertiesListEXT modifierPropList = {
			.sType = VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_EXT,
		};
		VkFormatProperties2 formatProps = {
			.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2,
			.pNext = &modifierPropList,
		};

		g_device.vk.GetPhysicalDeviceFormatProperties2( g_device.physDev(), format, &formatProps );

		if ( modifierPropList.drmFormatModifierCount == 0 )
		{
			vk_errorf( res, "vkGetPhysicalDeviceFormatProperties2 returned zero modifiers for DRM format 0x%" PRIX32, drmFormat );
			return false;
		}

		std::vector<VkDrmFormatModifierPropertiesEXT> modifierProps(modifierPropList.drmFormatModifierCount);
		modifierPropList.pDrmFormatModifierProperties = modifierProps.data();
		g_device.vk.GetPhysicalDeviceFormatProperties2( g_device.physDev(), format, &formatProps );

		std::map< uint64_t, VkDrmFormatModifierPropertiesEXT > map = {};

		for ( size_t j = 0; j < modifierProps.size(); j++ )
		{
			map[ modifierProps[j].drmFormatModifier ] = modifierProps[j];

			uint64_t modifier = modifierProps[j].drmFormatModifier;

			if ( !is_image_format_modifier_supported( format, drmFormat, modifier ) )
				continue;

			const VkFormatFeatureFlags requiredFeature = drmFormat != DRM_FORMAT_NV12 &&
				!(modifierProps[j].drmFormatModifierTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)
				? VK_FORMAT_FEATURE_TRANSFER_SRC_BIT : VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
			if ( ( modifierProps[j].drmFormatModifierTilingFeatures & requiredFeature ) == 0 )
			{
				continue;
			}

			// Composed clients need sampler-compatible modifiers, not KMS-compatible
			// ones. Backend-allocated scanout already forces their composition.
			if ( !g_bAllowDeferredBackend && !GetBackend()->UsesBackendAllocatedScanout() )
			{
				if ( GetBackend()->UsesModifiers() && !gamescope::Algorithm::Contains( GetBackend()->GetSupportedModifiers( drmFormat ), modifier ) )
					continue;
			}

			wlr_drm_format_set_add( &sampledDRMFormats, drmFormat, modifier );
			s_SampledModifierFormats[ drmFormat ].emplace_back( modifier );
		}

		DRMModifierProps[ format ] = map;
		return true;
	}
	else
	{
		if ( GetBackend()->UsesModifiers() && !GetBackend()->SupportsInvalidModifier( drmFormat ) )
			return false;

		wlr_drm_format_set_add( &sampledDRMFormats, drmFormat, DRM_FORMAT_MOD_INVALID );
		return false;
	}
}

bool vulkan_init_formats()
{
	for ( size_t i = 0; s_DRMVKFormatTable[i].DRMFormat != DRM_FORMAT_INVALID; i++ )
	{
		if (s_DRMVKFormatTable[i].internal)
			continue;

		VkFormat format = s_DRMVKFormatTable[i].vkFormat;
		VkFormat srgbFormat = s_DRMVKFormatTable[i].vkFormatSrgb;
		uint32_t drmFormat = s_DRMVKFormatTable[i].DRMFormat;

		vulkan_init_format(format, drmFormat);
		if (format != srgbFormat)
			vulkan_init_format(srgbFormat, drmFormat);
	}

	vk_log.infof( "supported DRM formats for sampling usage:" );
	for ( size_t i = 0; i < sampledDRMFormats.len; i++ )
	{
		uint32_t fmt = sampledDRMFormats.formats[ i ].format;
#if HAVE_DRM
		char *name = drmGetFormatName(fmt);
		vk_log.infof( "  %s (0x%" PRIX32 ")", name, fmt );
		free(name);
#endif
	}

	return true;
}

bool acquire_next_image( void )
{
	VkResult res = g_device.vk.AcquireNextImageKHR( g_device.device(), g_output.swapChain, UINT64_MAX, VK_NULL_HANDLE, g_output.acquireFence, &g_output.nOutImage );
	if ( res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR )
		return false;
	if ( g_device.vk.WaitForFences( g_device.device(), 1, &g_output.acquireFence, false, UINT64_MAX ) != VK_SUCCESS )
		return false;
	return g_device.vk.ResetFences( g_device.device(), 1, &g_output.acquireFence ) == VK_SUCCESS;
}


static std::atomic<uint64_t> g_currentPresentWaitId = {0u};
static std::atomic<uint64_t> g_currentPresentFrameGenerationOutputId = {0u};
static std::mutex present_wait_lock;
static std::jthread g_presentWaitThread;
static constexpr uint64_t k_uPresentWaitShutdownId = ~0ull;

extern void mangoapp_output_update( uint64_t vblanktime, uint64_t frameGenerationOutputId );
static void present_wait_thread_func( std::stop_token stopToken )
{
	pthread_setname_np( pthread_self(), "gamescope-prsnt" );
	uint64_t present_wait_id = 0;

	std::stop_callback stopCallback( stopToken, []
	{
		std::unique_lock lock( present_wait_lock );
		g_currentPresentWaitId = k_uPresentWaitShutdownId;
		g_currentPresentWaitId.notify_all();
	} );

	while ( !stopToken.stop_requested() )
	{
		g_currentPresentWaitId.wait( present_wait_id );

		// Lock to make sure swapchain destruction is waited on and that
		// it's for this swapchain.
		{
			std::unique_lock lock( present_wait_lock );
			present_wait_id = g_currentPresentWaitId.load();

			if ( stopToken.stop_requested() )
				break;

			if ( present_wait_id != 0 )
			{
				g_device.vk.WaitForPresentKHR( g_device.device(), g_output.swapChain, present_wait_id, 1'000'000'000lu );
				uint64_t vblanktime = get_time_in_nanos();
				GetVBlankTimer().MarkVBlank( vblanktime, true );
				mangoapp_output_update( vblanktime,
					g_currentPresentFrameGenerationOutputId.load() );
			}
		}
	}
}

void vulkan_present_wait_shutdown( void )
{
	if ( !g_presentWaitThread.joinable() )
		return;

	g_presentWaitThread.request_stop();
	g_presentWaitThread.join();
}

void vulkan_update_swapchain_hdr_metadata( VulkanOutput_t *pOutput )
{
	if (!g_output.swapchainHDRMetadata)
		return;

	if ( !g_device.vk.SetHdrMetadataEXT )
	{
		static bool s_bWarned = false;
		if (!s_bWarned)
		{
			vk_log.errorf("Unable to forward HDR metadata with Vulkan as vkSetMetadataEXT is not supported.");
			s_bWarned = true;
		}
		return;
	}

	const hdr_metadata_infoframe &infoframe = g_output.swapchainHDRMetadata->View<hdr_output_metadata>().hdmi_metadata_type1;
	VkHdrMetadataEXT metadata =
	{
		.sType = VK_STRUCTURE_TYPE_HDR_METADATA_EXT,
		.displayPrimaryRed = VkXYColorEXT { color_xy_from_u16(infoframe.display_primaries[0].x), color_xy_from_u16(infoframe.display_primaries[0].y) },
		.displayPrimaryGreen = VkXYColorEXT { color_xy_from_u16(infoframe.display_primaries[1].x), color_xy_from_u16(infoframe.display_primaries[1].y), },
		.displayPrimaryBlue = VkXYColorEXT { color_xy_from_u16(infoframe.display_primaries[2].x), color_xy_from_u16(infoframe.display_primaries[2].y), },
		.whitePoint = VkXYColorEXT { color_xy_from_u16(infoframe.white_point.x), color_xy_from_u16(infoframe.white_point.y), },
		.maxLuminance = nits_from_u16(infoframe.max_display_mastering_luminance),
		.minLuminance = nits_from_u16_dark(infoframe.min_display_mastering_luminance),
		.maxContentLightLevel = nits_from_u16(infoframe.max_cll),
		.maxFrameAverageLightLevel = nits_from_u16(infoframe.max_fall),
	};
	g_device.vk.SetHdrMetadataEXT(g_device.device(), 1, &g_output.swapChain, &metadata);
}

void vulkan_present_to_window( uint64_t frameGenerationOutputId )
{
	static uint64_t s_lastPresentId = 0;

	uint64_t presentId = ++s_lastPresentId;
	
	auto feedback = steamcompmgr_get_base_layer_swapchain_feedback();
	if (feedback && feedback->hdr_metadata_blob)
	{
		if ( feedback->hdr_metadata_blob != g_output.swapchainHDRMetadata )
		{
			g_output.swapchainHDRMetadata = feedback->hdr_metadata_blob;
			vulkan_update_swapchain_hdr_metadata( &g_output );
		}
	}
	else if ( g_output.swapchainHDRMetadata != nullptr )
	{
		// Only way to clear hdr metadata for a swapchain in Vulkan
		// is to recreate the swapchain.
		g_output.swapchainHDRMetadata = nullptr;
		vulkan_remake_swapchain();
	}


	VkPresentIdKHR presentIdInfo = {
		.sType = VK_STRUCTURE_TYPE_PRESENT_ID_KHR,
		.swapchainCount = 1,
		.pPresentIds = &presentId,
	};

	VkPresentInfoKHR presentInfo = {
		.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
		.pNext = &presentIdInfo,
		.swapchainCount = 1,
		.pSwapchains = &g_output.swapChain,
		.pImageIndices = &g_output.nOutImage,
	};

	if ( g_device.vk.QueuePresentKHR( g_device.queue(), &presentInfo ) == VK_SUCCESS )
	{
		g_currentPresentFrameGenerationOutputId = frameGenerationOutputId;
		g_currentPresentWaitId = presentId;
		g_currentPresentWaitId.notify_all();
	}
	else
		vulkan_remake_swapchain();

	while ( !acquire_next_image() )
		vulkan_remake_swapchain();
}

gamescope::Rc<CVulkanTexture> vulkan_create_1d_lut(uint32_t size, VkFormat format)
{
	CVulkanTexture::createFlags flags;
	flags.bSampled = true;
	flags.bTransferDst = true;
	flags.imageType = VK_IMAGE_TYPE_1D;

	auto texture = new CVulkanTexture();
	auto drmFormat = VulkanFormatToDRM( format );
	bool bRes = texture->BInit( size, 1u, 1u, drmFormat, flags );
	assert( bRes );

	return texture;
}

gamescope::Rc<CVulkanTexture> vulkan_create_3d_lut(uint32_t width, uint32_t height, uint32_t depth)
{
	CVulkanTexture::createFlags flags;
	flags.bSampled = true;
	flags.bTransferDst = true;
	flags.imageType = VK_IMAGE_TYPE_3D;

	auto texture = new CVulkanTexture();
	auto drmFormat = VulkanFormatToDRM( VK_FORMAT_R16G16B16A16_UNORM );
	bool bRes = texture->BInit( width, height, depth, drmFormat, flags );
	assert( bRes );

	return texture;
}

void vulkan_update_luts(const gamescope::Rc<CVulkanTexture>& lut1d, const gamescope::Rc<CVulkanTexture>& lut3d, void* lut1d_data, void* lut3d_data)
{
	const size_t lut1d_size = lut1d->width() * sizeof(uint16_t) * 4;
	const size_t lut3d_size = size_t(lut3d->width()) * lut3d->height() * lut3d->depth() * sizeof(uint16_t) * 4;
	const auto unchanged = [](const auto &texture, const void *data, size_t size) {
		return texture->uploadedLut.size() * sizeof(uint16_t) == size && !memcmp(texture->uploadedLut.data(), data, size);
	};
	const bool changed1d = !unchanged(lut1d, lut1d_data, lut1d_size);
	const bool changed3d = !unchanged(lut3d, lut3d_data, lut3d_size);
	if (!changed1d && !changed3d)
		return;
	lut1d->uploadedLut.assign(static_cast<uint16_t *>(lut1d_data), static_cast<uint16_t *>(lut1d_data) + lut1d_size / sizeof(uint16_t));
	lut3d->uploadedLut.assign(static_cast<uint16_t *>(lut3d_data), static_cast<uint16_t *>(lut3d_data) + lut3d_size / sizeof(uint16_t));

	// Recognize identity tables once, at upload, including explicitly supplied LUTs.
	// One 16-bit quantization step accounts for roundoff in generated matrices.
	const auto identityValue = [](uint16_t value, uint32_t coordinate, uint32_t size) {
		return size > 1 && std::abs(int(value) - int(std::lround(65535.0 * coordinate / (size - 1)))) <= 1;
	};
	const auto *one = static_cast<const uint16_t *>(lut1d_data);
	lut1d->identityLut = lut1d->width() > 1;
	for (uint32_t x = 0; x < lut1d->width() && lut1d->identityLut; ++x)
		for (uint32_t c = 0; c < 3; ++c)
			lut1d->identityLut &= identityValue(one[4 * x + c], x, lut1d->width());
	const auto *three = static_cast<const uint16_t *>(lut3d_data);
	lut1d->blackPreservingLut = one[0] == 0 && one[1] == 0 && one[2] == 0;
	lut3d->blackPreservingLut = three[0] == 0 && three[1] == 0 && three[2] == 0;
	lut3d->identityLut = lut3d->width() > 1 && lut3d->height() > 1 && lut3d->depth() > 1;
	for (uint32_t z = 0; z < lut3d->depth() && lut3d->identityLut; ++z)
		for (uint32_t y = 0; y < lut3d->height() && lut3d->identityLut; ++y)
			for (uint32_t x = 0; x < lut3d->width() && lut3d->identityLut; ++x)
			{
				const size_t p = 4 * (x + lut3d->width() * (y + lut3d->height() * z));
				lut3d->identityLut &= identityValue(three[p], x, lut3d->width()) &&
					identityValue(three[p + 1], y, lut3d->height()) && identityValue(three[p + 2], z, lut3d->depth());
			}

	// A separable 3D LUT is three independent curves. For aligned 8-bit SDR
	// samples, tabulate every possible input value, including blend linearization.
	// This preserves the LUT instead of dropping night mode or changing filtering.
	std::optional<std::array<uint16_t, 256 * 3>> predecoded;
	const std::array<uint32_t, 3> sizes{lut3d->width(), lut3d->height(), lut3d->depth()};
	const std::array<size_t, 3> strides{1, sizes[0], size_t(sizes[0]) * sizes[1]};
	bool separable = lut1d->identityLut && !lut3d->identityLut &&
		std::all_of(sizes.begin(), sizes.end(), [](auto size) { return size > 1; });
	for (uint32_t z = 0; z < sizes[2] && separable; ++z)
		for (uint32_t y = 0; y < sizes[1] && separable; ++y)
			for (uint32_t x = 0; x < sizes[0] && separable; ++x)
			{
				const size_t p = 4 * (x + sizes[0] * (y + sizes[1] * z));
				const std::array<uint32_t, 3> coordinates{x, y, z};
				for (uint32_t c = 0; c < 3; ++c)
					separable &= std::abs(int(three[p + c]) - int(three[4 * strides[c] * coordinates[c] + c])) <= 1;
			}
	if (separable)
	{
		predecoded.emplace();
		for (uint32_t x = 0; x < 256; ++x)
			for (uint32_t c = 0; c < 3; ++c)
			{
				const double p = double(x) * (sizes[c] - 1) / 255.0;
				const uint32_t lo = uint32_t(p), hi = std::min(lo + 1, sizes[c] - 1);
				const double a = three[4 * strides[c] * lo + c] / 65535.0;
				const double b = three[4 * strides[c] * hi + c] / 65535.0;
				const double encoded = a + (b - a) * (p - lo);
				const double linear = encoded <= 0.04045 ? encoded / 12.92 : std::pow((encoded + 0.055) / 1.055, 2.4);
				(*predecoded)[256 * c + x] = glm::packHalf1x16(float(linear));
			}
		if (!lut3d->predecodedLut)
			lut3d->predecodedLut = vulkan_create_1d_lut(256 * 3, VK_FORMAT_R16_SFLOAT);
		lut3d->predecodedLut->blackPreservingLut = lut3d->blackPreservingLut;
	}
	else
		lut3d->predecodedLut = nullptr;
	const size_t predecodedSize = predecoded ? sizeof(*predecoded) : 0;
	auto [base_dst, base_offset] = g_device.uploadBufferData(lut1d_size + lut3d_size + predecodedSize);

	void* lut1d_dst = base_dst;
	void *lut3d_dst = ((uint8_t*)base_dst) + lut1d_size;
	memcpy(lut1d_dst, lut1d_data, lut1d_size);
	memcpy(lut3d_dst, lut3d_data, lut3d_size);
	if (predecoded)
		memcpy(static_cast<uint8_t *>(base_dst) + lut1d_size + lut3d_size, predecoded->data(), predecodedSize);

	auto cmdBuffer = g_device.commandBuffer();
	if (changed1d)
		cmdBuffer->copyBufferToImage(g_device.uploadBuffer(), base_offset, 0, lut1d);
	if (changed3d)
		cmdBuffer->copyBufferToImage(g_device.uploadBuffer(), base_offset + lut1d_size, 0, lut3d);
	if (predecoded)
		cmdBuffer->copyBufferToImage(g_device.uploadBuffer(), base_offset + lut1d_size + lut3d_size, 0, lut3d->predecodedLut);
	// Subsequent draws use the same queue and the command-buffer barriers make
	// these writes visible. The upload arena waits before reusing its storage.
	g_device.submit(std::move(cmdBuffer));
}

gamescope::Rc<CVulkanTexture> vulkan_get_hacky_blank_texture()
{
	return g_output.temporaryHackyBlankImage.get();
}

gamescope::OwningRc<CVulkanTexture> vulkan_create_flat_texture( uint32_t width, uint32_t height, uint8_t r, uint8_t g, uint8_t b, uint8_t a )
{
	CVulkanTexture::createFlags flags;
	flags.bFlippable = true;
	flags.bSampled = true;
	flags.bTransferDst = true;

	gamescope::OwningRc<CVulkanTexture> texture = new CVulkanTexture();
	bool bRes = texture->BInit( width, height, 1u, VulkanFormatToDRM( VK_FORMAT_B8G8R8A8_UNORM ), flags );
	if ( !bRes )
		return nullptr;

	auto [_dst, offset] = g_device.uploadBufferData( width * height * 4 );
	uint8_t *dst = (uint8_t *)_dst;
	for ( uint32_t i = 0; i < width * height * 4; i += 4 )
	{
		dst[i + 0] = b;
		dst[i + 1] = g;
		dst[i + 2] = r;
		dst[i + 3] = a;
	}

	auto cmdBuffer = g_device.commandBuffer();
	cmdBuffer->copyBufferToImage(g_device.uploadBuffer(), offset, 0, texture.get());
	g_device.submit(std::move(cmdBuffer));
	g_device.waitIdle();

	return texture;
}

gamescope::OwningRc<CVulkanTexture> vulkan_create_debug_blank_texture()
{
	// To match Steam's scaling, which is capped at 1080p
	int width = std::min<int>( g_nOutputWidth, 1920 );
	int height = std::min<int>( g_nOutputHeight, 1080 );

	return vulkan_create_flat_texture( width, height, 0, 0, 0, 0 );
}

bool vulkan_supports_hdr10()
{
	for ( auto& format : g_output.surfaceFormats )
	{
		if ( format.colorSpace == VK_COLOR_SPACE_HDR10_ST2084_EXT )
			return true;
	}

	return false;
}

extern bool g_bOutputHDREnabled;

bool vulkan_make_swapchain( VulkanOutput_t *pOutput )
{
	uint32_t imageCount = pOutput->surfaceCaps.minImageCount + 1;
	uint32_t formatCount = pOutput->surfaceFormats.size();
	uint32_t surfaceFormat = formatCount;
	VkColorSpaceKHR preferredColorSpace = g_bOutputHDREnabled ? VK_COLOR_SPACE_HDR10_ST2084_EXT : VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;

	if ( surfaceFormat == formatCount )
	{
		for ( surfaceFormat = 0; surfaceFormat < formatCount; surfaceFormat++ )
		{
			if ( pOutput->surfaceFormats[ surfaceFormat ].format == VK_FORMAT_A2B10G10R10_UNORM_PACK32 &&
				 pOutput->surfaceFormats[ surfaceFormat ].colorSpace == preferredColorSpace )
				break;
		}
	}

	if ( surfaceFormat == formatCount )
	{
		for ( surfaceFormat = 0; surfaceFormat < formatCount; surfaceFormat++ )
		{
			if ( pOutput->surfaceFormats[ surfaceFormat ].format == VK_FORMAT_A2R10G10B10_UNORM_PACK32 &&
				 pOutput->surfaceFormats[ surfaceFormat ].colorSpace == preferredColorSpace )
				break;
		}
	}

	if ( surfaceFormat == formatCount )
	{
		for ( surfaceFormat = 0; surfaceFormat < formatCount; surfaceFormat++ )
		{
			if ( pOutput->surfaceFormats[ surfaceFormat ].format == VK_FORMAT_B8G8R8A8_UNORM &&
				 pOutput->surfaceFormats[ surfaceFormat ].colorSpace == preferredColorSpace )
				break;
		}
	}
	
	if ( surfaceFormat == formatCount )
		return false;

	VkFormat eVkFormat = pOutput->surfaceFormats[ surfaceFormat ].format;
	pOutput->uOutputFormat = VulkanFormatToDRM( pOutput->surfaceFormats[ surfaceFormat ].format );
	
	std::vector<VkFormat> formats =
	{
		ToSrgbVulkanFormat( eVkFormat ),
		ToLinearVulkanFormat( eVkFormat ),
	};

	if ( typed_storage_view_format(eVkFormat) != ToLinearVulkanFormat(eVkFormat) )
		formats.push_back(typed_storage_view_format(eVkFormat));

	VkImageFormatListCreateInfo usageListInfo = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO,
		.viewFormatCount = uint32_t(formats.size()),
		.pViewFormats = formats.data(),
	};

	vk_log.infof("Creating Gamescope nested swapchain with format %u and colorspace %u", eVkFormat, pOutput->surfaceFormats[surfaceFormat].colorSpace);

	VkSwapchainCreateInfoKHR createInfo = {
		.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
		.pNext = (formats[0] != formats[1] || formats.size() > 2) ? &usageListInfo : nullptr,
		.flags = (formats[0] != formats[1] || formats.size() > 2) ? VK_SWAPCHAIN_CREATE_MUTABLE_FORMAT_BIT_KHR : (VkSwapchainCreateFlagBitsKHR )0,
		.surface = pOutput->surface,
		.minImageCount = imageCount,
		.imageFormat = eVkFormat,
		.imageColorSpace = pOutput->surfaceFormats[surfaceFormat].colorSpace,
		.imageExtent = {
			.width = g_nOutputWidth,
			.height = g_nOutputHeight,
		},
		.imageArrayLayers = 1,
		.imageUsage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
		.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.preTransform = pOutput->surfaceCaps.currentTransform,
		.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
		.presentMode = VK_PRESENT_MODE_FIFO_KHR,
		.clipped = VK_TRUE,
	};

	if (g_device.vk.CreateSwapchainKHR( g_device.device(), &createInfo, nullptr, &pOutput->swapChain) != VK_SUCCESS ) {
		return false;
	}

	g_device.vk.GetSwapchainImagesKHR( g_device.device(), pOutput->swapChain, &imageCount, nullptr );
	std::vector<VkImage> swapchainImages( imageCount );
	g_device.vk.GetSwapchainImagesKHR( g_device.device(), pOutput->swapChain, &imageCount, swapchainImages.data() );

	pOutput->outputImages.resize(imageCount);

	for ( uint32_t i = 0; i < pOutput->outputImages.size(); i++ )
	{
		pOutput->outputImages[i] = new CVulkanTexture();

		if ( !pOutput->outputImages[i]->BInitFromSwapchain(swapchainImages[i], g_nOutputWidth, g_nOutputHeight, eVkFormat))
			return false;
	}

	VkFenceCreateInfo fenceInfo = {
		.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
	};

	g_device.vk.CreateFence( g_device.device(), &fenceInfo, nullptr, &pOutput->acquireFence );

	vulkan_update_swapchain_hdr_metadata(pOutput);

	return true;
}

bool vulkan_remake_swapchain( void )
{
	std::unique_lock lock(present_wait_lock);
	g_currentPresentWaitId = 0;
	g_currentPresentFrameGenerationOutputId = 0;
	g_currentPresentWaitId.notify_all();

	VulkanOutput_t *pOutput = &g_output;
	g_device.waitIdle();
	g_device.vk.QueueWaitIdle( g_device.queue() );

	pOutput->outputImages.clear();

	g_device.vk.DestroySwapchainKHR( g_device.device(), pOutput->swapChain, nullptr );

	// Delete screenshot/capture textures to be remade if needed
	for (auto& pScreenshotTexture : pOutput->pScreenshotTextures)
		pScreenshotTexture = nullptr;
	for (auto& pCaptureTexture : pOutput->pCaptureTextures)
		pCaptureTexture = nullptr;

	bool bRet = vulkan_make_swapchain( pOutput );
	assert( bRet ); // Something has gone horribly wrong!
	return bRet;
}

static std::vector<uint64_t> vulkan_get_backend_scanout_modifiers( VulkanOutput_t *pOutput )
{
	std::vector<uint64_t> modifiers;
	const bool bHasOverlay = pOutput->uOutputFormatOverlay != VK_FORMAT_UNDEFINED && !kDisablePartialComposition;

	for ( uint64_t ulModifier : GetBackend()->GetSupportedModifiers( pOutput->uOutputFormat ) )
	{
		if ( g_device.isV3dv() && ulModifier != DRM_FORMAT_MOD_LINEAR )
			continue;
		if ( ulModifier == DRM_FORMAT_MOD_INVALID )
			continue;
		if ( bHasOverlay &&
		     !gamescope::Algorithm::Contains( GetBackend()->GetSupportedModifiers( pOutput->uOutputFormatOverlay ), ulModifier ) )
			continue;
		modifiers.push_back( ulModifier );
	}

	return modifiers;
}

// GBM placement and Vulkan image usage are separate capabilities. In particular,
// linear scanout can be a color attachment without supporting storage or sampling.
static bool dmabuf_supports_usage(uint32_t format, uint64_t modifier, VkImageUsageFlags usage)
{
	VkImageCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType = VK_IMAGE_TYPE_2D, .format = DRMFormatToVulkan(format, false),
		.extent = {1, 1, 1}, .mipLevels = 1, .arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT, .usage = usage, .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	VkExternalImageFormatProperties external = { .sType = VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES };
	return getModifierProps(&info, modifier, &external) == VK_SUCCESS &&
		(external.externalMemoryProperties.externalMemoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT);
}

static bool vulkan_make_backend_output_images( VulkanOutput_t *pOutput,
	const CVulkanTexture::createFlags &outputImageflags,
	uint32_t uWidth, uint32_t uHeight )
{
	const std::vector<uint64_t> modifiers = vulkan_get_backend_scanout_modifiers( pOutput );
	if ( modifiers.empty() )
		return false;

	const bool bHasOverlay = pOutput->uOutputFormatOverlay != VK_FORMAT_UNDEFINED && !kDisablePartialComposition;
	for ( size_t i = 0; i < pOutput->outputImages.size(); i++ )
	{
		wlr_dmabuf_attributes dmabuf = {};
		if ( !GetBackend()->CreateScanoutDmabuf(
			uWidth, uHeight, pOutput->uOutputFormat, modifiers, &dmabuf ) )
			return false;

		auto flags = outputImageflags;
		const VkImageUsageFlags attachmentUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
		if (!dmabuf_supports_usage(pOutput->uOutputFormat, dmabuf.modifier,
			attachmentUsage | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT))
		{
			flags.bStorage = false;
			flags.bSampled = false;
			if (!dmabuf_supports_usage(pOutput->uOutputFormat, dmabuf.modifier, attachmentUsage))
			{
				wlr_dmabuf_attributes_finish(&dmabuf);
				return false;
			}
		}

		pOutput->outputImages[i] = new CVulkanTexture();
		bool bSuccess = pOutput->outputImages[i]->BInit(
			uWidth, uHeight, 1u, pOutput->uOutputFormat,
			flags, &dmabuf );

		if ( bSuccess && bHasOverlay )
		{
			wlr_dmabuf_attributes overlayDmabuf = dmabuf;
			overlayDmabuf.format = pOutput->uOutputFormatOverlay;
			pOutput->outputImagesPartialOverlay[i] = new CVulkanTexture();
			bSuccess = pOutput->outputImagesPartialOverlay[i]->BInit(
				uWidth, uHeight, 1u, pOutput->uOutputFormatOverlay,
				flags, &overlayDmabuf );
		}

		wlr_dmabuf_attributes_finish( &dmabuf );
		if ( !bSuccess )
			return false;
	}

	return true;
}

static bool vulkan_make_output_images( VulkanOutput_t *pOutput )
{
	pOutput->compositionScratch = nullptr;
	CVulkanTexture::createFlags outputImageflags;
	outputImageflags.bFlippable = true;
	outputImageflags.bStorage = true;
	outputImageflags.bColorAttachment = true;
	outputImageflags.bTransferDst = true;
	outputImageflags.bTransferSrc = true; // for screenshots
	outputImageflags.bSampled = true; // for pipewire blits
	outputImageflags.bOutputImage = true;

	pOutput->outputImages.resize(3); // extra image for partial composition.
	pOutput->outputImagesPartialOverlay.resize(3);

	pOutput->outputImages[0] = nullptr;
	pOutput->outputImages[1] = nullptr;
	pOutput->outputImages[2] = nullptr;
	pOutput->outputImagesPartialOverlay[0] = nullptr;
	pOutput->outputImagesPartialOverlay[1] = nullptr;
	pOutput->outputImagesPartialOverlay[2] = nullptr;

	uint32_t uDRMFormat = pOutput->uOutputFormat;

	uint32_t l_nOutputWidth = g_nOutputWidth;
	uint32_t l_nOutputHeight = g_nOutputHeight;

	if ( g_uOutputRotation & 1u ) {
		l_nOutputWidth = g_nOutputHeight;
		l_nOutputHeight = g_nOutputWidth;
	}

	if ( GetBackend()->UsesBackendAllocatedScanout() )
	{
		if ( vulkan_make_backend_output_images( pOutput, outputImageflags, l_nOutputWidth, l_nOutputHeight ) )
		{
			// Oh no.
			pOutput->temporaryHackyBlankImage = vulkan_create_debug_blank_texture();
			return true;
		}

		vk_log.errorf( "Failed to create backend GBM scanout buffers; falling back to Vulkan scanout allocation." );
		for ( auto &pImage : pOutput->outputImages )
			pImage = nullptr;
		for ( auto &pImage : pOutput->outputImagesPartialOverlay )
			pImage = nullptr;
	}

	pOutput->outputImages[0] = new CVulkanTexture();
	bool bSuccess = pOutput->outputImages[0]->BInit( l_nOutputWidth, l_nOutputHeight, 1u, uDRMFormat, outputImageflags );
	if ( bSuccess != true )
	{
		vk_log.errorf( "failed to allocate buffer for KMS" );
		return false;
	}

	pOutput->outputImages[1] = new CVulkanTexture();
	bSuccess = pOutput->outputImages[1]->BInit( l_nOutputWidth, l_nOutputHeight, 1u, uDRMFormat, outputImageflags );
	if ( bSuccess != true )
	{
		vk_log.errorf( "failed to allocate buffer for KMS" );
		return false;
	}

	pOutput->outputImages[2] = new CVulkanTexture();
	bSuccess = pOutput->outputImages[2]->BInit( l_nOutputWidth, l_nOutputHeight, 1u, uDRMFormat, outputImageflags );
	if ( bSuccess != true )
	{
		vk_log.errorf( "failed to allocate buffer for KMS" );
		return false;
	}

	// Oh no.
	pOutput->temporaryHackyBlankImage = vulkan_create_debug_blank_texture();

	if ( pOutput->uOutputFormatOverlay != VK_FORMAT_UNDEFINED && !kDisablePartialComposition )
	{
		uint32_t uPartialDRMFormat = pOutput->uOutputFormatOverlay;

		pOutput->outputImagesPartialOverlay[0] = new CVulkanTexture();
		bool bSuccess = pOutput->outputImagesPartialOverlay[0]->BInit( l_nOutputWidth, l_nOutputHeight, 1u, uPartialDRMFormat, outputImageflags, nullptr, 0, 0, pOutput->outputImages[0].get() );
		if ( bSuccess != true )
		{
			vk_log.errorf( "failed to allocate buffer for KMS" );
			return false;
		}

		pOutput->outputImagesPartialOverlay[1] = new CVulkanTexture();
		bSuccess = pOutput->outputImagesPartialOverlay[1]->BInit( l_nOutputWidth, l_nOutputHeight, 1u, uPartialDRMFormat, outputImageflags, nullptr, 0, 0, pOutput->outputImages[1].get() );
		if ( bSuccess != true )
		{
			vk_log.errorf( "failed to allocate buffer for KMS" );
			return false;
		}

		pOutput->outputImagesPartialOverlay[2] = new CVulkanTexture();
		bSuccess = pOutput->outputImagesPartialOverlay[2]->BInit( l_nOutputWidth, l_nOutputHeight, 1u, uPartialDRMFormat, outputImageflags, nullptr, 0, 0, pOutput->outputImages[2].get() );
		if ( bSuccess != true )
		{
			vk_log.errorf( "failed to allocate buffer for KMS" );
			return false;
		}
	}

	return true;
}

// Without a swapchain the shared images are ours to make, so they wait for their first use.
static bool vulkan_defer_output_images()
{
	return !GetBackend()->UsesVulkanSwapchain();
}

static bool vulkan_ensure_output_images()
{
	if ( !vulkan_defer_output_images() )
		return true;

	VulkanOutput_t *pOutput = &g_output;
	if ( !pOutput->outputImages.empty() && pOutput->outputImages[0] )
		return true;

	vk_log.debugf( "Making the shared output images on first use" );
	if ( vulkan_make_output_images( pOutput ) )
		return true;

	// A half-made set would pass the check above, so the next use tries again from nothing.
	pOutput->outputImages.clear();
	pOutput->outputImagesPartialOverlay.clear();
	return false;
}

bool vulkan_remake_output_images()
{
	VulkanOutput_t *pOutput = &g_output;
	g_device.waitIdle();

	pOutput->nOutImage = 0;

	// Delete screenshot/capture textures to be remade if needed
	for (auto& pScreenshotTexture : pOutput->pScreenshotTextures)
		pScreenshotTexture = nullptr;
	for (auto& pCaptureTexture : pOutput->pCaptureTextures)
		pCaptureTexture = nullptr;

	if ( vulkan_defer_output_images() )
	{
		pOutput->compositionScratch = nullptr;
		pOutput->outputImages.clear();
		pOutput->outputImagesPartialOverlay.clear();
		pOutput->temporaryHackyBlankImage = vulkan_create_debug_blank_texture();
		return true;
	}

	bool bRet = vulkan_make_output_images( pOutput );
	assert( bRet );
	return bRet;
}

bool vulkan_make_output()
{
	VulkanOutput_t *pOutput = &g_output;

	VkResult result;
	
	if ( GetBackend()->UsesVulkanSwapchain() )
	{
		result = g_device.vk.GetPhysicalDeviceSurfaceCapabilitiesKHR( g_device.physDev(), pOutput->surface, &pOutput->surfaceCaps );
		if ( result != VK_SUCCESS )
		{
			vk_errorf( result, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR failed" );
			return false;
		}
		
		uint32_t formatCount = 0;
		result = g_device.vk.GetPhysicalDeviceSurfaceFormatsKHR( g_device.physDev(), pOutput->surface, &formatCount, nullptr );
		if ( result != VK_SUCCESS )
		{
			vk_errorf( result, "vkGetPhysicalDeviceSurfaceFormatsKHR failed" );
			return false;
		}
		
		if ( formatCount != 0 ) {
			pOutput->surfaceFormats.resize( formatCount );
			g_device.vk.GetPhysicalDeviceSurfaceFormatsKHR( g_device.physDev(), pOutput->surface, &formatCount, pOutput->surfaceFormats.data() );
			if ( result != VK_SUCCESS )
			{
				vk_errorf( result, "vkGetPhysicalDeviceSurfaceFormatsKHR failed" );
				return false;
			}
		}
		
		uint32_t presentModeCount = false;
		result = g_device.vk.GetPhysicalDeviceSurfacePresentModesKHR(g_device.physDev(), pOutput->surface, &presentModeCount, nullptr );
		if ( result != VK_SUCCESS )
		{
			vk_errorf( result, "vkGetPhysicalDeviceSurfacePresentModesKHR failed" );
			return false;
		}
		
		if ( presentModeCount != 0 ) {
			pOutput->presentModes.resize(presentModeCount);
			result = g_device.vk.GetPhysicalDeviceSurfacePresentModesKHR( g_device.physDev(), pOutput->surface, &presentModeCount, pOutput->presentModes.data() );
			if ( result != VK_SUCCESS )
			{
				vk_errorf( result, "vkGetPhysicalDeviceSurfacePresentModesKHR failed" );
				return false;
			}
		}
		
		if ( !vulkan_make_swapchain( pOutput ) )
			return false;

		while ( !acquire_next_image() )
			vulkan_remake_swapchain();
	}
	else
	{
		GetBackend()->GetPreferredOutputFormat( &pOutput->uOutputFormat, &pOutput->uOutputFormatOverlay );

		if ( pOutput->uOutputFormat == DRM_FORMAT_INVALID )
		{
			vk_log.errorf( "failed to find Vulkan format suitable for KMS" );
			return false;
		}

		if ( pOutput->uOutputFormatOverlay == DRM_FORMAT_INVALID )
		{
			vk_log.errorf( "failed to find Vulkan format suitable for KMS partial overlays" );
			return false;
		}

		if ( vulkan_defer_output_images() )
			pOutput->temporaryHackyBlankImage = vulkan_create_debug_blank_texture();
		else if ( !vulkan_make_output_images( pOutput ) )
			return false;
	}

	return true;
}

static void update_tmp_images( uint32_t width, uint32_t height, uint32_t drmFormat = DRM_FORMAT_ARGB8888 )
{
	if ( g_output.tmpOutput != nullptr
			&& width == g_output.tmpOutput->width()
			&& height == g_output.tmpOutput->height()
			&& drmFormat == g_output.tmpOutput->drmFormat() )
	{
		return;
	}

	CVulkanTexture::createFlags createFlags;
	createFlags.bSampled = true;
	createFlags.bStorage = true;

	g_output.tmpOutput = new CVulkanTexture();
	bool bSuccess = g_output.tmpOutput->BInit( width, height, 1u, drmFormat, createFlags, nullptr );

	if ( !bSuccess )
	{
		vk_log.errorf( "failed to create fsr output" );
		return;
	}
}



static bool init_nis_data()
{
	// Create the NIS images
	// Select between the FP16 or FP32 coefficients

	void* coefScaleData = g_device.supportsFp16() ? (void*) coef_scale_fp16 : (void*) coef_scale;

	void* coefUsmData = g_device.supportsFp16() ? (void*) coef_usm_fp16 : (void*) coef_usm;

	uint32_t nisFormat = g_device.supportsFp16() ? DRM_FORMAT_ABGR16161616F : DRM_FORMAT_ABGR32323232F;

	uint32_t width = kFilterSize / 4;
	uint32_t height = kPhaseCount;

	g_output.nisScalerImage = vulkan_create_texture_from_bits( width, height, width, height, nisFormat, {}, coefScaleData );
	g_output.nisUsmImage = vulkan_create_texture_from_bits( width, height, width, height, nisFormat, {}, coefUsmData );

	return true;
}

VkInstance vulkan_get_instance( void )
{
	static VkInstance s_pVkInstance = []() -> VkInstance
	{
		VkResult result = VK_ERROR_INITIALIZATION_FAILED;

		if ( ( result = vulkan_load_module() ) != VK_SUCCESS )
		{
			vk_errorf( result, "Failed to load vulkan module." );
			return nullptr;
		}

		auto instanceExtensions = GetBackend()->GetInstanceExtensions();

		const VkApplicationInfo appInfo = {
			.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO,
			.pApplicationName   = "gamescope",
			.applicationVersion = VK_MAKE_VERSION(1, 0, 0),
			.pEngineName        = "hopefully not just some code",
			.engineVersion      = VK_MAKE_VERSION(1, 0, 0),
			.apiVersion         = VK_API_VERSION_1_3,
		};

		const VkInstanceCreateInfo createInfo = {
			.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
			.pApplicationInfo        = &appInfo,
			.enabledExtensionCount   = (uint32_t)instanceExtensions.size(),
			.ppEnabledExtensionNames = instanceExtensions.data(),
		};

		VkInstance instance = nullptr;
		result = g_pfn_vkCreateInstance(&createInfo, 0, &instance);
		if ( result != VK_SUCCESS )
		{
			vk_errorf( result, "vkCreateInstance failed" );
		}

		return instance;
	}();

	return s_pVkInstance;
}

bool vulkan_init( VkInstance instance, VkSurfaceKHR surface )
{
	static bool s_bInitted = false;
	if ( s_bInitted )
	{
		g_output.surface = surface;
		return true;
	}

	if (!g_device.BInit(instance, surface))
		return false;

	if (!init_nis_data())
		return false;

	if ( GetBackend()->UsesVulkanSwapchain() )
	{
		g_presentWaitThread = std::jthread( present_wait_thread_func );
	}

	s_bInitted = true;

	return true;
}

gamescope::OwningRc<CVulkanTexture> vulkan_create_texture_from_dmabuf( struct wlr_dmabuf_attributes *pDMA, gamescope::OwningRc<gamescope::IBackendFb> pBackendFb )
{
	gamescope::OwningRc<CVulkanTexture> pTex = new CVulkanTexture();

	CVulkanTexture::createFlags texCreateFlags;
	texCreateFlags.bSampled = true;
	if (pDMA->modifier != DRM_FORMAT_MOD_INVALID && pDMA->format != DRM_FORMAT_NV12 &&
		dmabuf_supports_usage(pDMA->format, pDMA->modifier, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT))
		texCreateFlags.bTransferSrc = true;
	if ( pDMA->modifier != DRM_FORMAT_MOD_INVALID && pDMA->format != DRM_FORMAT_NV12 &&
		!dmabuf_supports_usage(pDMA->format, pDMA->modifier, VK_IMAGE_USAGE_SAMPLED_BIT) &&
		dmabuf_supports_usage(pDMA->format, pDMA->modifier, VK_IMAGE_USAGE_TRANSFER_SRC_BIT) )
	{
		CVulkanTexture::createFlags sourceFlags;
		sourceFlags.bTransferSrc = true;
		gamescope::OwningRc<CVulkanTexture> source = new CVulkanTexture();
		if ( !source->BInit( pDMA->width, pDMA->height, 1u, pDMA->format, sourceFlags, pDMA ) )
			return nullptr;
		texCreateFlags.bTransferDst = true;
		if ( !pTex->BInit( pDMA->width, pDMA->height, 1u, pDMA->format, texCreateFlags ) )
			return nullptr;
		pTex->linearSource = std::move( source );
		return pTex;
	}

	//fprintf(stderr, "pDMA->width: %d pDMA->height: %d pDMA->format: 0x%x pDMA->modifier: 0x%lx pDMA->n_planes: %d\n",
	//	pDMA->width, pDMA->height, pDMA->format, pDMA->modifier, pDMA->n_planes);
	
	if ( pTex->BInit( pDMA->width, pDMA->height, 1u, pDMA->format, texCreateFlags, pDMA, 0, 0, nullptr, pBackendFb ) == false )
		return nullptr;
	
	return pTex;
}

gamescope::OwningRc<CVulkanTexture> vulkan_create_texture_from_bits( uint32_t width, uint32_t height, uint32_t contentWidth, uint32_t contentHeight, uint32_t drmFormat, CVulkanTexture::createFlags texCreateFlags, void *bits )
{
	gamescope::OwningRc<CVulkanTexture> pTex = new CVulkanTexture();

	texCreateFlags.bSampled = true;
	texCreateFlags.bTransferDst = true;

	if ( pTex->BInit( width, height, 1u, drmFormat, texCreateFlags, nullptr,  contentWidth, contentHeight) == false )
		return nullptr;

	size_t size = width * height * DRMFormatGetBPP(drmFormat);
	auto [ dst, offset ] = g_device.uploadBufferData(size);
	memcpy( dst, bits, size );

	auto cmdBuffer = g_device.commandBuffer();

	cmdBuffer->copyBufferToImage(g_device.uploadBuffer(), offset, 0, pTex.get());
	// TODO: Sync this copyBufferToImage.

	g_device.submit(std::move(cmdBuffer));
	g_device.waitIdle();

	return pTex;
}

static uint32_t s_frameId = 0;

void vulkan_garbage_collect( void )
{
	g_device.garbageCollect();
}

static gamescope::Rc<CVulkanTexture> acquire_pooled_texture( auto& pool, uint32_t width, uint32_t height, bool exportable, uint32_t drmFormat, EStreamColorspace colorspace, bool bDeviceLocal = false )
{
	for (auto& pTexture : pool)
	{
		// Evict a stale texture and reuse the slot
		if (pTexture && pTexture->GetRefCount() == 0 &&
			(width != pTexture->width() ||
			 height != pTexture->height() ||
			 drmFormat != pTexture->drmFormat() ||
			 bDeviceLocal != (pTexture->mappedData() == nullptr)))
		{
			pTexture = nullptr;
		}

		if (pTexture == nullptr)
		{
			pTexture = new CVulkanTexture();

			CVulkanTexture::createFlags textureFlags;
			textureFlags.bTransferDst = true;
			textureFlags.bStorage = true;
			textureFlags.bSampled = true; // required for RGB-to-NV12 shader to sample this texture
			if ( !bDeviceLocal )
			{
				textureFlags.bReadback = g_device.isV3dv() && !exportable;
				textureFlags.bMappable = !textureFlags.bReadback;
				if (!textureFlags.bReadback && (exportable || drmFormat == DRM_FORMAT_NV12)) {
					textureFlags.bExportable = true;
					textureFlags.bLinear = true; // TODO: support multi-planar DMA-BUF export via PipeWire
				}
			}

			bool bSuccess = pTexture->BInit( width, height, 1u, drmFormat, textureFlags );
			pTexture->setStreamColorspace(colorspace);

			assert( bSuccess );
		}

		if (pTexture->GetRefCount() != 0 ||
			width != pTexture->width() ||
			height != pTexture->height() ||
			drmFormat != pTexture->drmFormat() ||
			bDeviceLocal != (pTexture->mappedData() == nullptr))
			continue;

		return pTexture.get();
	}

	return nullptr;
}

gamescope::Rc<CVulkanTexture> vulkan_acquire_screenshot_texture(uint32_t width, uint32_t height, bool exportable, uint32_t drmFormat, EStreamColorspace colorspace)
{
	auto texture = acquire_pooled_texture(g_output.pScreenshotTextures, width, height, exportable, drmFormat, colorspace);
	if (!texture)
		vk_log.errorf("Unable to acquire screenshot texture. Out of textures.");
	return texture;
}

gamescope::Rc<CVulkanTexture> vulkan_acquire_capture_texture(uint32_t width, uint32_t height, bool exportable, uint32_t drmFormat, EStreamColorspace colorspace, bool bDeviceLocal)
{
	auto texture = acquire_pooled_texture(g_output.pCaptureTextures, width, height, exportable, drmFormat, colorspace, bDeviceLocal);
	if (!texture)
		vk_log.errorf("Unable to acquire capture texture. Out of textures.");
	return texture;
}

// XRGB2101010 storage images are an optional Vulkan feature that NVIDIA hardware
// doesn't support, and imageStore lands in XBGR order there, swapping R/B.
uint32_t vulkan_get_rgb10_capture_format( void )
{
	static uint32_t s_uFormat = []()
	{
		// Check both tilings, direct capture textures are linear and readback ones optimal
		VkFormatProperties props;
		g_device.vk.GetPhysicalDeviceFormatProperties( g_device.physDev(), VK_FORMAT_A2R10G10B10_UNORM_PACK32, &props );
		constexpr VkFormatFeatureFlags uNeeded = VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
		if ( ( props.linearTilingFeatures & props.optimalTilingFeatures & uNeeded ) == uNeeded )
			return DRM_FORMAT_XRGB2101010;

		vk_log.infof( "XRGB2101010 lacks storage/sampled support, capturing as XBGR2101010" );
		return DRM_FORMAT_XBGR2101010;
	}();
	return s_uFormat;
}

// Internal display's native brightness.
float g_flInternalDisplayBrightnessNits = 500.0f;

float g_flHDRItmSdrNits = 100.f;
float g_flHDRItmTargetNits = 1000.f;

#pragma pack(push, 1)
struct BlitPushData_t
{
	vec2_t scale[k_nMaxLayers];
	vec2_t offset[k_nMaxLayers];
	float opacity[k_nMaxLayers];
	glm::mat3x4 ctm[k_nMaxLayers];
	uint32_t borderMask;
	uint32_t frameId;
	uint32_t blurRadius;

	uint32_t u_shaderFilter;
	uint32_t u_alphaMode;

    float u_linearToNits; // unset
    float u_nitsToLinear; // unset
    float u_itmSdrNits; // unset
    float u_itmTargetNits; // unset

	uint32_t u_rotation = 0;

	explicit BlitPushData_t(const struct FrameInfo_t *frameInfo, uint32_t rotation = 0)
	{
		u_shaderFilter = 0;
		u_alphaMode = 0;
		u_rotation = rotation;

		for (int i = 0; i < frameInfo->layers.count(); i++) {
			const FrameInfo_t::Layer_t *layer = &frameInfo->layers.get( i );
			scale[i] = layer->scale;
			offset[i] = layer->offsetPixelCenter();
			opacity[i] = layer->opacity;
			u_shaderFilter |= uint32_t(layer->shaderFilter()) << (i * 4);

			u_alphaMode |= ((uint32_t)layer->eAlphaBlendingMode) << ( i * 4 );

			if (layer->ctm)
			{
				ctm[i] = layer->ctm->View<glm::mat3x4>();
			}
			else
			{
				ctm[i] = glm::mat3x4
				{
					1, 0, 0, 0,
					0, 1, 0, 0,
					0, 0, 1, 0
				};
			}
		}

		borderMask = frameInfo->borderMask();
		frameId = s_frameId++;
		blurRadius = frameInfo->blurRadius ? ( frameInfo->blurRadius * 2 ) - 1 : 0;

		u_linearToNits = g_flInternalDisplayBrightnessNits;
		u_nitsToLinear = 1.0f / g_flInternalDisplayBrightnessNits;
		u_itmSdrNits = g_flHDRItmSdrNits;
		u_itmTargetNits = g_flHDRItmTargetNits;
	}

	explicit BlitPushData_t(float blit_scale) {
		scale[0] = { blit_scale, blit_scale };
		offset[0] = { 0.5f, 0.5f };
		opacity[0] = 1.0f;
        u_shaderFilter = (uint32_t)GamescopeUpscaleFilter::LINEAR;
		u_alphaMode = 0;
		ctm[0] = glm::mat3x4
		{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0
		};
		borderMask = 0;
		frameId = s_frameId;

		u_linearToNits = g_flInternalDisplayBrightnessNits;
		u_nitsToLinear = 1.0f / g_flInternalDisplayBrightnessNits;
		u_itmSdrNits = g_flHDRItmSdrNits;
		u_itmTargetNits = g_flHDRItmTargetNits;
	}
};

struct CaptureConvertBlitData_t
{
	vec2_t scale[1];
	vec2_t offset[1];
	float opacity[1];
	glm::mat3x4 ctm[1];
	mat3x4 outputCTM;
	uint32_t borderMask;
	uint32_t halfExtent[2];

	explicit CaptureConvertBlitData_t(float blit_scale, const mat3x4 &color_matrix) {
		scale[0] = { blit_scale, blit_scale };
		offset[0] = { 0.0f, 0.0f };
		opacity[0] = 1.0f;
		borderMask = 0;
		ctm[0] = glm::mat3x4
		{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0
		};
		outputCTM = color_matrix;
	}
};

struct uvec4_t
{
	uint32_t  x;
	uint32_t  y;
	uint32_t  z;
	uint32_t  w;
};
struct uvec2_t
{
	uint32_t x;
	uint32_t y;
};

struct EasuPushData_t
{
	uvec4_t Const0;
	uvec4_t Const1;
	uvec4_t Const2;
	uvec4_t Const3;
	glm::mat3x4 ctm;

	EasuPushData_t(const FrameInfo_t::Layer_t *layer, uint32_t inputX, uint32_t inputY, uint32_t tempX, uint32_t tempY)
	{
		FsrEasuCon(&Const0.x, &Const1.x, &Const2.x, &Const3.x, inputX, inputY, inputX, inputY, tempX, tempY);

		if (layer->ctm)
		{
			ctm = layer->ctm->View<glm::mat3x4>();
		}
		else
		{
			ctm = glm::mat3x4
			{
				1, 0, 0, 0,
				0, 1, 0, 0,
				0, 0, 1, 0
			};
		}
	}
};

struct SgsrPushData_t
{
	uint32_t u_width;
	uint32_t u_height;

	SgsrPushData_t(uint32_t tempX, uint32_t tempY)
		: u_width(tempX), u_height(tempY)
	{
	}
};

struct RcasPushData_t
{
	uvec2_t u_layer0Offset;
	vec2_t u_scale[k_nMaxLayers - 1];
	vec2_t u_offset[k_nMaxLayers - 1];
	float u_opacity[k_nMaxLayers];
	glm::mat3x4 ctm[k_nMaxLayers];
	uint32_t u_borderMask;
	uint32_t u_frameId;
	uint32_t u_c1;

	uint32_t u_shaderFilter;
	uint32_t u_alphaMode;

    float u_linearToNits; // unset
    float u_nitsToLinear; // unset
    float u_itmSdrNits; // unset
    float u_itmTargetNits; // unset

	uint32_t u_rotation = 0;

	RcasPushData_t(const struct FrameInfo_t *frameInfo, float sharpness, uint32_t rotation = 0)
	{
		u_rotation = rotation;
		uvec4_t tmp;
		FsrRcasCon(&tmp.x, sharpness);
		u_layer0Offset.x = uint32_t(int32_t(frameInfo->layers[0].offset.x));
		u_layer0Offset.y = uint32_t(int32_t(frameInfo->layers[0].offset.y));
		u_borderMask = frameInfo->borderMask() >> 1u;
		u_frameId = s_frameId++;
		u_c1 = tmp.x;
		u_shaderFilter = 0;
		u_alphaMode = 0;

		for (int i = 0; i < frameInfo->layers.count(); i++)
		{
			const FrameInfo_t::Layer_t *layer = &frameInfo->layers.get( i );

            GamescopeUpscaleFilter eFilter = layer->filter == GamescopeUpscaleFilter::SGSR ? GamescopeUpscaleFilter::LINEAR : layer->filter;
            if (i == 0 || layer->isScreenSize() || (eFilter == GamescopeUpscaleFilter::LINEAR && layer->viewConvertsToLinearAutomatically()))
                u_shaderFilter |= ((uint32_t)GamescopeUpscaleFilter::FROM_VIEW) << (i * 4);
            else
                u_shaderFilter |= ((uint32_t)eFilter) << (i * 4);

			u_alphaMode |= ((uint32_t)layer->eAlphaBlendingMode) << ( i * 4 );

			if (layer->ctm)
			{
				ctm[i] = layer->ctm->View<glm::mat3x4>();
			}
			else
			{
				ctm[i] = glm::mat3x4
				{
					1, 0, 0, 0,
					0, 1, 0, 0,
					0, 0, 1, 0
				};
			}

			u_opacity[i] = frameInfo->layers.get( i ).opacity;
		}

		u_linearToNits = g_flInternalDisplayBrightnessNits;
		u_nitsToLinear = 1.0f / g_flInternalDisplayBrightnessNits;
		u_itmSdrNits = g_flHDRItmSdrNits;
		u_itmTargetNits = g_flHDRItmTargetNits;

		for (int i = 1; i < frameInfo->layers.count(); i++)
		{
			u_scale[i - 1] = frameInfo->layers.get( i ).scale;
			u_offset[i - 1] = frameInfo->layers.get( i ).offsetPixelCenter();
		}
	}
};

struct NisPushData_t
{
	NISConfig nisConfig;

	NisPushData_t(uint32_t inputX, uint32_t inputY, uint32_t tempX, uint32_t tempY, float sharpness, NISHDRMode hdrMode)
	{
		NVScalerUpdateConfig(
			nisConfig, sharpness,
			0, 0,
			inputX, inputY,
			inputX, inputY,
			0, 0,
			tempX, tempY,
			tempX, tempY,
			hdrMode);
	}
};
#pragma pack(pop)

void bind_all_layers(CVulkanCmdBuffer* cmdBuffer, const struct FrameInfo_t *frameInfo, uint32_t encodedInputMask = 0)
{
	for ( int i = 0; i < frameInfo->layers.count(); i++ )
	{
		const FrameInfo_t::Layer_t *layer = &frameInfo->layers.get( i );

		bool nearest = layer->isScreenSize()
                    || layer->filter == GamescopeUpscaleFilter::NEAREST
                    || (layer->filter == GamescopeUpscaleFilter::LINEAR && !layer->viewConvertsToLinearAutomatically());

		cmdBuffer->bindTexture(i, layer->tex);
		cmdBuffer->setTextureSrgb(i, (encodedInputMask & (1u << i)) || layer->colorspace != GAMESCOPE_APP_TEXTURE_COLORSPACE_LINEAR);
		cmdBuffer->setSamplerNearest(i, nearest);
		cmdBuffer->setSamplerUnnormalized(i, true);
	}
	for (uint32_t i = frameInfo->layers.count(); i < VKR_SAMPLER_SLOTS; i++)
	{
		cmdBuffer->bindTexture(i, nullptr);
	}
}

std::optional<uint64_t> vulkan_screenshot( const struct FrameInfo_t *frameInfo, gamescope::Rc<CVulkanTexture> pScreenshotTexture, gamescope::Rc<CVulkanTexture> pYUVOutTexture )
{
	EOTF outputTF = frameInfo->outputEncodingEOTF;
	if (frameInfo->blendInDisplayEncoding)
		outputTF = EOTF_Count; // Mura correction blends encoded values.

	auto cmdBuffer = g_device.commandBuffer();

	for (uint32_t i = 0; i < EOTF_Count; i++)
		cmdBuffer->bindColorMgmtLuts(i, frameInfo->shaperLut[i], frameInfo->lut3D[i]);

	cmdBuffer->bindPipeline( g_device.pipeline(SHADER_TYPE_BLIT, frameInfo->layers.count(), frameInfo->ycbcrMask(), 0u, frameInfo->colorspaceMask(), outputTF ));
	bind_all_layers(cmdBuffer.get(), frameInfo);
	cmdBuffer->bindTarget(pScreenshotTexture);
	cmdBuffer->uploadConstants<BlitPushData_t>(frameInfo);

	const int pixelsPerGroup = 8;

	cmdBuffer->dispatch(div_roundup(currentOutputWidth, pixelsPerGroup), div_roundup(currentOutputHeight, pixelsPerGroup));

	if ( pScreenshotTexture->isReadback() )
		cmdBuffer->copyImageToReadback(pScreenshotTexture);

	if ( pYUVOutTexture != nullptr )
	{
		float scale = (float)pScreenshotTexture->width() / pYUVOutTexture->width();

		CaptureConvertBlitData_t constants( scale, colorspace_to_conversion_from_srgb_matrix( pYUVOutTexture->streamColorspace() ) );
		constants.halfExtent[0] = pYUVOutTexture->width() / 2.0f;
		constants.halfExtent[1] = pYUVOutTexture->height() / 2.0f;
		cmdBuffer->uploadConstants<CaptureConvertBlitData_t>(constants);

		for (uint32_t i = 0; i < EOTF_Count; i++)
			cmdBuffer->bindColorMgmtLuts(i, nullptr, nullptr);

		cmdBuffer->bindPipeline(g_device.pipeline( SHADER_TYPE_RGB_TO_NV12, 1, 0, 0, GAMESCOPE_APP_TEXTURE_COLORSPACE_SRGB, EOTF_Count ));
		cmdBuffer->bindTexture(0, pScreenshotTexture);
		cmdBuffer->setTextureSrgb(0, true);
		cmdBuffer->setSamplerNearest(0, false);
		cmdBuffer->setSamplerUnnormalized(0, true);
		for (uint32_t i = 1; i < VKR_SAMPLER_SLOTS; i++)
		{
			cmdBuffer->bindTexture(i, nullptr);
		}
		cmdBuffer->bindTarget(pYUVOutTexture);

		const int pixelsPerGroup = 8;

		// For ycbcr, we operate on 2 pixels at a time, so use the half-extent.
		const int dispatchSize = pixelsPerGroup * 2;

		cmdBuffer->dispatch(div_roundup(pYUVOutTexture->width(), dispatchSize), div_roundup(pYUVOutTexture->height(), dispatchSize));

		if ( pYUVOutTexture->isReadback() )
			cmdBuffer->copyImageToReadback(pYUVOutTexture);
	}

	uint64_t sequence = g_device.submit(std::move(cmdBuffer));
	return sequence;
}

gamescope::ConVar<int> cv_capture_readback{ "capture_readback", -1,
	"Keep PipeWire capture images device-local and read back through a buffer. "
	"-1 = auto (NVIDIA), 0 = never, 1 = always" };

bool vulkan_capture_prefers_readback( void )
{
	if ( cv_capture_readback >= 0 )
		return cv_capture_readback != 0;
	// NVIDIA maps host-visible memory GPU-uncached on both the proprietary
	// driver and NVK, so shader stores to it become uncoalesced PCIe writes
	return g_device.vendorID() == 0x10de; /* NVIDIA */
}

extern std::string g_reshade_effect;
extern uint32_t g_reshade_technique_idx;



static bool is_8bit_sdr_target( const CVulkanTexture *texture )
{
	return texture->format() == VK_FORMAT_B8G8R8A8_UNORM ||
	       texture->format() == VK_FORMAT_R8G8B8A8_UNORM;
}

static bool is_fp16_fsr_target( const CVulkanTexture *texture )
{
	return texture->format() == VK_FORMAT_R16G16B16A16_SFLOAT;
}

static bool is_rgb10a2_target( const CVulkanTexture *texture )
{
	// BGR targets use a compatible RGB view for typed stores.
	return texture->format() == VK_FORMAT_A2B10G10R10_UNORM_PACK32 ||
	       texture->format() == VK_FORMAT_A2R10G10B10_UNORM_PACK32;
}

static ShaderType blit_shader_for_target( const CVulkanTexture *texture )
{
	return is_fp16_fsr_target(texture) ? SHADER_TYPE_BLIT_RGBA16F :
	       is_rgb10a2_target(texture) ? SHADER_TYPE_BLIT_RGB10A2 : SHADER_TYPE_BLIT;
}

static uint32_t tmp_image_format( const FrameInfo_t *frame )
{
	switch ( frame->layers[0].colorspace )
	{
	case GAMESCOPE_APP_TEXTURE_COLORSPACE_SCRGB: return DRM_FORMAT_ABGR16161616F;
	case GAMESCOPE_APP_TEXTURE_COLORSPACE_HDR10_PQ: return DRM_FORMAT_ABGR2101010;
	default: return DRM_FORMAT_ARGB8888;
	}
}

static ShaderType blur_shader_for_target( ShaderType type, const CVulkanTexture *target )
{
	if ( type == SHADER_TYPE_BLUR_FIRST_PASS )
		return is_fp16_fsr_target(target) ? SHADER_TYPE_BLUR_FIRST_PASS_RGBA16F : is_rgb10a2_target(target) ? SHADER_TYPE_BLUR_FIRST_PASS_RGB10A2 : type;
	if ( type == SHADER_TYPE_BLUR_COND )
		return is_fp16_fsr_target(target) ? SHADER_TYPE_BLUR_COND_RGBA16F : is_rgb10a2_target(target) ? SHADER_TYPE_BLUR_COND_RGB10A2 : type;
	return is_fp16_fsr_target(target) ? SHADER_TYPE_BLUR_RGBA16F : is_rgb10a2_target(target) ? SHADER_TYPE_BLUR_RGB10A2 : type;
}

static bool can_use_simple_fsr_output( const FrameInfo_t *frameInfo, CVulkanTexture *output, EOTF outputTF )
{
	if ( frameInfo->layers.count() != 1 ||
	     frameInfo->blendInDisplayEncoding ||
	     outputTF != EOTF_Gamma22 ||
	     g_bHDRItmEnable ||
	     g_uCompositeDebug != 0 ||
	     !is_8bit_sdr_target( output ) )
		return false;

	const FrameInfo_t::Layer_t &layer = frameInfo->layers[0];
	if ( !close_enough( layer.opacity, 1.0f ) ||
	     !close_enough( layer.offset.x, 0.0f ) ||
	     !close_enough( layer.offset.y, 0.0f ) ||
	     layer.blackBorder ||
	     layer.ctm ||
	     layer.integerWidth() != currentOutputWidth ||
	     layer.integerHeight() != currentOutputHeight ||
	     output->width() != currentOutputWidth ||
	     output->height() != currentOutputHeight )
		return false;

	if ( layer.colorspace != GAMESCOPE_APP_TEXTURE_COLORSPACE_LINEAR &&
	     layer.colorspace != GAMESCOPE_APP_TEXTURE_COLORSPACE_SRGB )
		return false;

	// RCAS writes encoded SDR into a bounded UNORM target. With one opaque
	// layer, an identity SDR LUT adds no clamping or blending that changes it.
	return (!frameInfo->shaperLut[EOTF_Gamma22] && !frameInfo->lut3D[EOTF_Gamma22]) ||
		(frameInfo->shaperLut[EOTF_Gamma22] && frameInfo->lut3D[EOTF_Gamma22] &&
		 frameInfo->shaperLut[EOTF_Gamma22]->identityLut && frameInfo->lut3D[EOTF_Gamma22]->identityLut);
}

static bool can_use_fp16_fsr_rcas( const FrameInfo_t *frameInfo, const CVulkanTexture *output, EOTF outputTF )
{
	if ( cv_fsr_rcas_fp16 == 0 ||
	     !g_device.supportsFp16() ||
	     frameInfo->blendInDisplayEncoding ||
	     outputTF != EOTF_Gamma22 ||
	     g_bHDRItmEnable ||
	     !is_8bit_sdr_target( output ) )
		return false;

	// Automatic mode is intentionally conservative until other vendors have
	// equivalent image-quality and driver validation. Positive values override
	// the vendor policy, but never the Vulkan feature or SDR format checks.
	if ( cv_fsr_rcas_fp16 < 0 && g_device.vendorID() != 0x1002 )
		return false;

	for ( int i = 0; i < frameInfo->layers.count(); i++ )
	{
		const GamescopeAppTextureColorspace colorspace = frameInfo->layers[i].colorspace;
		if ( colorspace != GAMESCOPE_APP_TEXTURE_COLORSPACE_LINEAR &&
		     colorspace != GAMESCOPE_APP_TEXTURE_COLORSPACE_SRGB )
			return false;
	}

	return true;
}

std::optional<uint64_t> vulkan_composite( struct FrameInfo_t *frameInfo, gamescope::Rc<CVulkanTexture> pPipewireTexture, bool partial, gamescope::Rc<CVulkanTexture> pOutputOverride, bool increment, std::unique_ptr<CVulkanCmdBuffer> pInCommandBuffer, gamescope::Rc<CVulkanTexture> pFsrIntermediateOverride )
{
	if ( !pOutputOverride && !vulkan_ensure_output_images() )
		return std::nullopt;

	const uint32_t uOutputRotation = pOutputOverride ? 0u : g_uOutputRotation;
	FrameInfo_t sampledFrame = *frameInfo;
	uint32_t predecodedLutMask = 0;
	gamescope::Rc<CVulkanTexture> scanout = pOutputOverride ? pOutputOverride.get()
		: partial ? g_output.outputImagesPartialOverlay[g_output.nOutImage].get() : g_output.outputImages[g_output.nOutImage].get();
	if (!frameInfo->useFSRLayer0 && !frameInfo->useSGSRLayer0 && !frameInfo->useNISLayer0 && !frameInfo->blurLayer0 && uOutputRotation == 0 && g_reshade_effect.empty())
	{


		if (g_uCompositeDebug == 0)
		{
			int retained = 0;
			for (int i = 0; i < sampledFrame.layers.count(); ++i)
			{
				auto &layer = sampledFrame.layers[i];
				if (i > 0 && layer.opacity == 0.0f && layer.eAlphaBlendingMode != ALPHA_BLENDING_MODE_NONE)
					continue;
				const auto offset = layer.offsetPixelCenter();
				const float x1 = offset.x * layer.scale.x, y1 = offset.y * layer.scale.y;
				const float x2 = (currentOutputWidth - 1 + offset.x) * layer.scale.x;
				const float y2 = (currentOutputHeight - 1 + offset.y) * layer.scale.y;
				const bool coversOutput = layer.blackBorder ||
					(std::min(x1, x2) >= 0 && std::min(y1, y2) >= 0 &&
					 std::max(x1, x2) < layer.tex->width() && std::max(y1, y2) < layer.tex->height());
				// NONE replaces every output pixel, including transparent borders.
				if (i > 0 && (layer.eAlphaBlendingMode == ALPHA_BLENDING_MODE_NONE ||
					(layer.opacity == 1.0f && !layer.hasAlpha() && coversOutput)))
					retained = 0;
				if (retained != i)
					sampledFrame.layers[retained] = std::move(layer);
				++retained;
			}
			sampledFrame.layers.truncate(retained);
		}
		for (int i = 0; i < sampledFrame.layers.count(); ++i)
		{
			auto &layer = sampledFrame.layers[i];
			if (layer.colorspace == GAMESCOPE_APP_TEXTURE_COLORSPACE_SRGB &&
				!(layer.outputColorMgmtApplied && sampledFrame.outputEncodingEOTF == EOTF_Gamma22) &&
				ToSrgbVulkanFormat(layer.tex->format()) != layer.tex->format())
				layer.colorspace = GAMESCOPE_APP_TEXTURE_COLORSPACE_LINEAR;
		}
		uint32_t usedEotfs = 0;
		bool boundedSdr = true;
		for (auto &layer : std::span(sampledFrame.layers.data(), sampledFrame.layers.count()))
		{
			if (layer.ctm && layer.ctm->View<glm::mat3x4>() == glm::mat3x4{
				1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0})
				layer.ctm = nullptr;
			if (layer.outputColorMgmtApplied && sampledFrame.outputEncodingEOTF == EOTF_Gamma22 &&
				layer.colorspace == GAMESCOPE_APP_TEXTURE_COLORSPACE_SRGB)
				continue;
			if (layer.colorspace == GAMESCOPE_APP_TEXTURE_COLORSPACE_PASSTHRU)
				continue;
			const bool sdr = layer.colorspace == GAMESCOPE_APP_TEXTURE_COLORSPACE_LINEAR ||
				layer.colorspace == GAMESCOPE_APP_TEXTURE_COLORSPACE_SRGB;
			usedEotfs |= 1u << (g_bHDRItmEnable || !sdr ? EOTF_PQ : EOTF_Gamma22);
			if (sdr)
				boundedSdr &= !layer.ctm && ToSrgbVulkanFormat(layer.tex->format()) != layer.tex->format();
		}
		for (uint32_t i = 0; i < EOTF_Count; ++i)
		{
			// No dependency, descriptor reference, or shader work for unused tables.
			const bool identity = sampledFrame.applyOutputColorMgmt && i == EOTF_Gamma22 && sampledFrame.outputEncodingEOTF == EOTF_Gamma22 &&
				!g_bHDRItmEnable && boundedSdr && sampledFrame.shaperLut[i] && sampledFrame.lut3D[i] &&
				sampledFrame.shaperLut[i]->identityLut && sampledFrame.lut3D[i]->identityLut;
			if (!(usedEotfs & (1u << i)) || identity)
			{
				sampledFrame.shaperLut[i] = nullptr;
				sampledFrame.lut3D[i] = nullptr;
			}
			else if (i == EOTF_Gamma22 && is_8bit_sdr_target(scanout.get()) && !sampledFrame.blendInDisplayEncoding &&
				sampledFrame.outputEncodingEOTF == EOTF_Gamma22 && sampledFrame.shaperLut[i] &&
				sampledFrame.shaperLut[i]->identityLut && sampledFrame.lut3D[i] && sampledFrame.lut3D[i]->predecodedLut)
			{
				sampledFrame.shaperLut[i] = sampledFrame.lut3D[i]->predecodedLut;
				predecodedLutMask |= 1u << i;
			}

		}
		frameInfo = &sampledFrame;
	}


	EOTF outputTF = frameInfo->outputEncodingEOTF;
	if (frameInfo->blendInDisplayEncoding)
		outputTF = EOTF_Count; // Mura correction blends encoded values.

	g_pLastReshadeEffect = nullptr;
	if (!g_reshade_effect.empty())
	{
		if (frameInfo->layers.count() > 0 && frameInfo->layers.get( 0 ).tex)
		{
			ReshadeEffectKey key
			{
				.path             = g_reshade_effect,
				.bufferWidth      = frameInfo->layers.get( 0 ).tex->width(),
				.bufferHeight     = frameInfo->layers.get( 0 ).tex->height(),
				.bufferColorSpace = frameInfo->layers.get( 0 ).colorspace,
				.bufferFormat     = frameInfo->layers.get( 0 ).tex->format(),
				.techniqueIdx     = g_reshade_technique_idx,
			};

			ReshadeEffectPipeline* pipeline = g_reshadeManager.pipeline(key);
			g_pLastReshadeEffect = pipeline;

			if (pipeline != nullptr)
			{
				uint64_t seq = pipeline->execute(frameInfo->layers.get( 0 ).tex, &frameInfo->layers.get( 0 ).tex);
				g_device.wait(seq);
			}
		}
	}
	else
	{
		g_reshadeManager.clear();
	}

	const FrameInfo_t::Layer_t emptyBase{};
	const auto &base = frameInfo->layers.count() > 0 ? frameInfo->layers[0] : emptyBase;
	const auto *copySource = base.tex && base.tex->linearSource && base.tex->linearSourceDirty ? base.tex->linearSource.get() : base.tex.get();
	const bool canCopyBase = frameInfo->layers.count() > 0 && copySource && !partial && !pOutputOverride && !pPipewireTexture &&
		!GetBackend()->UsesVulkanSwapchain() && uOutputRotation == 0 && !frameInfo->useFSRLayer0 && !frameInfo->useSGSRLayer0 && !frameInfo->useNISLayer0 &&
		!frameInfo->blurLayer0 && !frameInfo->frameGenerationActive && !g_bHDRItmEnable && !g_pLastReshadeEffect && g_uCompositeDebug == 0 &&
		outputTF == EOTF_Gamma22 && (base.colorspace == GAMESCOPE_APP_TEXTURE_COLORSPACE_LINEAR || base.colorspace == GAMESCOPE_APP_TEXTURE_COLORSPACE_SRGB) &&
		!base.ctm && !frameInfo->shaperLut[EOTF_Gamma22] && base.opacity == 1.0f &&
		base.scale.x == 1.0f && base.scale.y == 1.0f && base.offset.x == 0.0f && base.offset.y == 0.0f &&
		base.tex->width() == scanout->width() && base.tex->height() == scanout->height() &&
		copySource->format() == scanout->format() && (copySource->usage() & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) &&
		(scanout->usage() & VK_IMAGE_USAGE_TRANSFER_DST_BIT) &&
		(!DRMFormatHasAlpha(scanout->drmFormat()) || DRMFormatHasAlpha(copySource->drmFormat()));
	const bool copyOnly = canCopyBase && frameInfo->layers.count() == 1;
	const bool raster = frameInfo->layers.count() > 0 && scanout->colorAttachment() && g_device.queueFamily() == g_device.generalQueueFamily() &&
		uOutputRotation == 0 && !frameInfo->useFSRLayer0 && !frameInfo->useSGSRLayer0 && !frameInfo->useNISLayer0 && !frameInfo->blurLayer0 &&
		!pPipewireTexture && !g_bHDRItmEnable && g_uCompositeDebug == 0 &&
		(outputTF == EOTF_Count || (outputTF == EOTF_Gamma22 && ToSrgbVulkanFormat(scanout->format()) != scanout->format())) &&
		std::all_of(frameInfo->layers.data() + 1, frameInfo->layers.data() + frameInfo->layers.count(),
			[](const auto &layer) { return layer.eAlphaBlendingMode == ALPHA_BLENDING_MODE_PREMULTIPLIED; });
	if ( !copyOnly && !raster && !pOutputOverride && !scanout->storageImage() && !g_output.compositionScratch )
	{
		CVulkanTexture::createFlags flags;
		flags.bStorage = flags.bSampled = flags.bTransferSrc = flags.bTransferDst = true;
		g_output.compositionScratch = new CVulkanTexture();
		if ( !g_output.compositionScratch->BInit(scanout->width(), scanout->height(), 1, scanout->drmFormat(), flags) )
		{
			g_output.compositionScratch = nullptr;
			return std::nullopt;
		}
	}
	gamescope::Rc<CVulkanTexture> compositeImage = !copyOnly && !raster && !pOutputOverride && g_output.compositionScratch
		? g_output.compositionScratch.get() : scanout.get();

	auto cmdBuffer = pInCommandBuffer ? std::move( pInCommandBuffer ) : g_device.commandBuffer();

	if (!copyOnly)
		for (uint32_t i = 0; i < EOTF_Count; i++)
			cmdBuffer->bindColorMgmtLuts(i, frameInfo->shaperLut[i], frameInfo->lut3D[i]);
	// Subsurfaces and popups are fenced by the client's acquire point. Wait on it on
	// the GPU rather than on this thread to avoid stalling the compositor.
	for ( int i = 0; i < frameInfo->layers.count(); i++ )
	{
		const FrameInfo_t::Layer_t &layer = frameInfo->layers.get( i );
		if ( layer.acquirePoint && !cmdBuffer->AddDependency(
			layer.acquirePoint->GetTimeline()->ToVkSemaphore(), layer.acquirePoint->GetPoint() ) )
			return std::nullopt;
	}


	if (copyOnly)
		cmdBuffer->copyImage(base.tex, scanout);
	else if ( frameInfo->useFSRLayer0 || frameInfo->useSGSRLayer0 )
	{
		uint32_t inputX = frameInfo->layers.get( 0 ).tex->width();
		uint32_t inputY = frameInfo->layers.get( 0 ).tex->height();

		uint32_t tempX = frameInfo->layers.get( 0 ).integerWidth();
		uint32_t tempY = frameInfo->layers.get( 0 ).integerHeight();

		if ( !pFsrIntermediateOverride )
		{
			const uint32_t fsrIntermediateFormat = ColorspaceIsHDR( frameInfo->layers[0].colorspace )
				? DRM_FORMAT_ABGR16161616F
				: DRM_FORMAT_ARGB8888;
			update_tmp_images( tempX, tempY, fsrIntermediateFormat );
		}
		gamescope::Rc<CVulkanTexture> fsrIntermediate = pFsrIntermediateOverride;
		if ( !fsrIntermediate )
			fsrIntermediate = g_output.tmpOutput;

		const ShaderType easuShader = frameInfo->useSGSRLayer0 ? SHADER_TYPE_SGSR : is_fp16_fsr_target( fsrIntermediate.get() )
			? SHADER_TYPE_EASU_RGBA16F
			: SHADER_TYPE_EASU;
		cmdBuffer->bindPipeline(g_device.pipeline(easuShader, 1, 0, 0, frameInfo->layers[0].colorspace));
		cmdBuffer->bindTarget(fsrIntermediate);
		cmdBuffer->bindTexture(0, frameInfo->layers[0].tex);
		cmdBuffer->setTextureSrgb(0, true);
		cmdBuffer->setSamplerUnnormalized(0, false);
		cmdBuffer->setSamplerNearest(0, false);
		if ( frameInfo->useSGSRLayer0 )
			cmdBuffer->uploadConstants<SgsrPushData_t>(tempX, tempY);
		else
			cmdBuffer->uploadConstants<EasuPushData_t>(&frameInfo->layers[0], inputX, inputY, tempX, tempY);

		int pixelsPerGroup = 16;

		cmdBuffer->dispatch(div_roundup(tempX, pixelsPerGroup), div_roundup(tempY, pixelsPerGroup));

		const bool simpleFsrOutput = can_use_simple_fsr_output( frameInfo, compositeImage.get(), outputTF );
		const ShaderType rcasShader = is_fp16_fsr_target( compositeImage.get() )
			? SHADER_TYPE_RCAS_RGBA16F
			: is_rgb10a2_target( compositeImage.get() )
				? SHADER_TYPE_RCAS_RGB10A2
				: can_use_fp16_fsr_rcas( frameInfo, compositeImage.get(), outputTF )
					? SHADER_TYPE_RCAS_FP16
					: SHADER_TYPE_RCAS;
		cmdBuffer->bindPipeline(g_device.pipeline(rcasShader, frameInfo->layers.count(), frameInfo->ycbcrMask() & ~1, 0u, frameInfo->colorspaceMask(), outputTF, false, simpleFsrOutput ));
		bind_all_layers(cmdBuffer.get(), frameInfo);
		cmdBuffer->bindTexture(0, fsrIntermediate);
		cmdBuffer->setTextureSrgb(0, true);
		cmdBuffer->setSamplerUnnormalized(0, false);
		cmdBuffer->setSamplerNearest(0, false);

		cmdBuffer->bindTarget(compositeImage);

		cmdBuffer->uploadConstants<RcasPushData_t>(frameInfo, frameInfo->nUpscaleSharpness / 10.0f, uOutputRotation);

		cmdBuffer->dispatch(div_roundup(currentOutputWidth, pixelsPerGroup), div_roundup(currentOutputHeight, pixelsPerGroup));
	}
	else if ( frameInfo->useNISLayer0 )
	{
		uint32_t inputX = frameInfo->layers.get( 0 ).tex->width();
		uint32_t inputY = frameInfo->layers.get( 0 ).tex->height();

		uint32_t tempX = frameInfo->layers.get( 0 ).integerWidth();
		uint32_t tempY = frameInfo->layers.get( 0 ).integerHeight();

		update_tmp_images(tempX, tempY, tmp_image_format(frameInfo));

		float nisSharpness = (20 - frameInfo->nUpscaleSharpness) / 20.0f;

		// The SDR luma ramp bottoms out at 0.9, which is 72 nits in scRGB.
		const bool bNisLinearHdr = frameInfo->layers.get( 0 ).colorspace == GAMESCOPE_APP_TEXTURE_COLORSPACE_SCRGB;

		cmdBuffer->bindPipeline(g_device.pipeline(bNisLinearHdr ? SHADER_TYPE_NIS_HDR_LINEAR : is_rgb10a2_target(g_output.tmpOutput.get()) ? SHADER_TYPE_NIS_RGB10A2 : SHADER_TYPE_NIS));
		cmdBuffer->bindTarget(g_output.tmpOutput);
		cmdBuffer->bindTexture(0, frameInfo->layers.get( 0 ).tex);
		cmdBuffer->setTextureSrgb(0, true);
		cmdBuffer->setSamplerUnnormalized(0, false);
		cmdBuffer->setSamplerNearest(0, false);
		cmdBuffer->bindTexture(VKR_NIS_COEF_SCALER_SLOT, g_output.nisScalerImage);
		cmdBuffer->setSamplerUnnormalized(VKR_NIS_COEF_SCALER_SLOT, false);
		cmdBuffer->setSamplerNearest(VKR_NIS_COEF_SCALER_SLOT, false);
		cmdBuffer->bindTexture(VKR_NIS_COEF_USM_SLOT, g_output.nisUsmImage);
		cmdBuffer->setSamplerUnnormalized(VKR_NIS_COEF_USM_SLOT, false);
		cmdBuffer->setSamplerNearest(VKR_NIS_COEF_USM_SLOT, false);
		// NISHDRMode{} is None, which X11's None macro hides.
		cmdBuffer->uploadConstants<NisPushData_t>(inputX, inputY, tempX, tempY, nisSharpness, bNisLinearHdr ? NISHDRMode::Linear : NISHDRMode{});

		int pixelsPerGroupX = 32;
		int pixelsPerGroupY = 24;

		cmdBuffer->dispatch(div_roundup(tempX, pixelsPerGroupX), div_roundup(tempY, pixelsPerGroupY));

		struct FrameInfo_t nisFrameInfo = *frameInfo;
		nisFrameInfo.layers.get( 0 ).tex = g_output.tmpOutput;
		nisFrameInfo.layers.get( 0 ).scale.x = 1.0f;
		nisFrameInfo.layers.get( 0 ).scale.y = 1.0f;

		const ShaderType nisBlitShader = blit_shader_for_target( compositeImage.get() );
		cmdBuffer->bindPipeline( g_device.pipeline(nisBlitShader, nisFrameInfo.layers.count(), nisFrameInfo.ycbcrMask(), 0u, nisFrameInfo.colorspaceMask(), outputTF ));
		bind_all_layers(cmdBuffer.get(), &nisFrameInfo);

		cmdBuffer->bindTarget(compositeImage);

		cmdBuffer->uploadConstants<BlitPushData_t>(&nisFrameInfo, uOutputRotation);

		int pixelsPerGroup = 8;

		cmdBuffer->dispatch(div_roundup(currentOutputWidth, pixelsPerGroup), div_roundup(currentOutputHeight, pixelsPerGroup));
	}
	else if ( frameInfo->blurLayer0 )
	{
		update_tmp_images(currentOutputWidth, currentOutputHeight, tmp_image_format(frameInfo));

		ShaderType type = blur_shader_for_target(SHADER_TYPE_BLUR_FIRST_PASS, g_output.tmpOutput.get());

		uint32_t blur_layer_count = 1;
		// Also blur the override on top if we have one.
		for ( uint32_t i = 1; i < (uint32_t)frameInfo->layers.count(); i++ )
		{
			if ( frameInfo->layers.get( i ).zpos > (int)g_zposOverride )
				break;
			blur_layer_count = i + 1;
		}

		cmdBuffer->bindPipeline(g_device.pipeline(type, blur_layer_count, frameInfo->ycbcrMask() & 0x3u, 0, frameInfo->colorspaceMask(), outputTF ));
		cmdBuffer->bindTarget(g_output.tmpOutput);
		for (uint32_t i = 0; i < blur_layer_count; i++)
		{
			cmdBuffer->bindTexture(i, frameInfo->layers.get( i ).tex);
			cmdBuffer->setTextureSrgb(i, false);
			cmdBuffer->setSamplerUnnormalized(i, true);
			cmdBuffer->setSamplerNearest(i, false);
		}
		cmdBuffer->uploadConstants<BlitPushData_t>(frameInfo);

		int pixelsPerGroup = 8;

		cmdBuffer->dispatch(div_roundup(currentOutputWidth, pixelsPerGroup), div_roundup(currentOutputHeight, pixelsPerGroup));

		bool useSrgbView = frameInfo->layers.get( 0 ).colorspace == GAMESCOPE_APP_TEXTURE_COLORSPACE_LINEAR;

		type = blur_shader_for_target(frameInfo->blurLayer0 == BLUR_MODE_COND ? SHADER_TYPE_BLUR_COND : SHADER_TYPE_BLUR, compositeImage.get());
		cmdBuffer->bindPipeline(g_device.pipeline(type, frameInfo->layers.count(), frameInfo->ycbcrMask(), blur_layer_count, frameInfo->colorspaceMask(), outputTF ));
		bind_all_layers(cmdBuffer.get(), frameInfo);

		cmdBuffer->bindTarget(compositeImage);

		cmdBuffer->uploadConstants<BlitPushData_t>(frameInfo, uOutputRotation);
		cmdBuffer->bindTexture(VKR_BLUR_EXTRA_SLOT, g_output.tmpOutput);
		cmdBuffer->setTextureSrgb(VKR_BLUR_EXTRA_SLOT, !useSrgbView); // Inverted because it chooses whether to view as linear (sRGB view) or sRGB (raw view). It's horrible. I need to change it.
		cmdBuffer->setSamplerUnnormalized(VKR_BLUR_EXTRA_SLOT, true);
		cmdBuffer->setSamplerNearest(VKR_BLUR_EXTRA_SLOT, false);

		cmdBuffer->dispatch(div_roundup(currentOutputWidth, pixelsPerGroup), div_roundup(currentOutputHeight, pixelsPerGroup));
	}
	else
	{
		CompositeShaderConfig_t compositeConfig;
		compositeConfig.opaqueOutput = !DRMFormatHasAlpha(compositeImage->drmFormat());
		compositeConfig.filterMask = compositeConfig.colorMgmtMask = compositeConfig.samplingMask = 0;
		for (int i = 0; i < frameInfo->layers.count(); ++i)
		{
			const auto &layer = frameInfo->layers[i];
			compositeConfig.filterMask |= uint32_t(layer.shaderFilter()) << (i * 4);
			if (layer.ctm)
				compositeConfig.colorMgmtMask |= 1u << i;
			const bool aligned = !layer.isYcbcr() && layer.scale.x == 1.0f && layer.scale.y == 1.0f &&
				float_is_integer(layer.offset.x) && float_is_integer(layer.offset.y);
			if (aligned)
				compositeConfig.samplingMask |= 1u << (i + k_nMaxLayers);
			if (aligned && !layer.ctm && layer.colorspace == GAMESCOPE_APP_TEXTURE_COLORSPACE_LINEAR &&
				frameInfo->shaperLut[EOTF_Gamma22] && ToSrgbVulkanFormat(layer.tex->format()) != layer.tex->format())
				compositeConfig.colorMgmtMask |= 1u << (k_nMaxLayers + 3 * EOTF_Count + i);
			if (layer.blackBorder)
				compositeConfig.samplingMask |= 1u << (i + 2 * k_nMaxLayers);
			if (aligned && layer.offset.x == 0.0f && layer.offset.y == 0.0f &&
				layer.tex->width() >= currentOutputWidth && layer.tex->height() >= currentOutputHeight)
				compositeConfig.samplingMask |= 1u << i;
		}
		for (uint32_t i = 0; i < EOTF_Count; ++i)
		{
			if (frameInfo->shaperLut[i])
			{
				compositeConfig.colorMgmtMask |= 1u << (k_nMaxLayers + i);
				if (frameInfo->shaperLut[i]->identityLut || (predecodedLutMask & (1u << i)))
					compositeConfig.colorMgmtMask |= 1u << (k_nMaxLayers + EOTF_Count + i);
				if (predecodedLutMask & (1u << i))
					compositeConfig.colorMgmtMask |= 1u << (2 * k_nMaxLayers + 3 * EOTF_Count + i);
				if (frameInfo->shaperLut[i]->blackPreservingLut && frameInfo->lut3D[i] && frameInfo->lut3D[i]->blackPreservingLut)
					compositeConfig.colorMgmtMask |= 1u << (k_nMaxLayers + 2 * EOTF_Count + i);
			}
			else
				compositeConfig.colorMgmtMask |= 1u << (k_nMaxLayers + 2 * EOTF_Count + i);
		}
		const bool simpleBlend = frameInfo->layers.count() > 0 && frameInfo->layers[0].opacity == 1.0f &&
			std::all_of(frameInfo->layers.data() + 1, frameInfo->layers.data() + frameInfo->layers.count(),
				[](const auto &layer) { return layer.opacity == 1.0f && layer.eAlphaBlendingMode == ALPHA_BLENDING_MODE_PREMULTIPLIED; });

		bind_all_layers(cmdBuffer.get(), frameInfo, compositeConfig.colorMgmtMask >> (k_nMaxLayers + 3 * EOTF_Count));
		cmdBuffer->bindTarget(compositeImage);
		cmdBuffer->uploadConstants<BlitPushData_t>(frameInfo, uOutputRotation);

		if (raster)
			cmdBuffer->drawComposite(frameInfo, outputTF, compositeConfig, canCopyBase);
		else
		{
			cmdBuffer->bindPipeline(g_device.pipeline(blit_shader_for_target(compositeImage.get()), frameInfo->layers.count(),
				frameInfo->ycbcrMask(), 0u, frameInfo->colorspaceMask(), outputTF, false, false, simpleBlend, compositeConfig));
			cmdBuffer->dispatch(div_roundup(currentOutputWidth, 8), div_roundup(currentOutputHeight, 8));
		}
	}

	if ( frameInfo->frameGenerationActive )
		vulkan_frame_generation_record_output_timestamp(
			cmdBuffer.get(), frameInfo->useFSRLayer0 );

	if ( pPipewireTexture != nullptr )
	{

		if (compositeImage->format() == pPipewireTexture->format() &&
			compositeImage->width() == pPipewireTexture->width() &&
		    compositeImage->height() == pPipewireTexture->height()) {
			cmdBuffer->copyImage(compositeImage, pPipewireTexture);
		} else {
			const bool ycbcr = pPipewireTexture->isYcbcr();

			float scale = (float)compositeImage->width() / pPipewireTexture->width();
			if ( ycbcr )
			{
				CaptureConvertBlitData_t constants( scale, colorspace_to_conversion_from_srgb_matrix( pPipewireTexture->streamColorspace() ) );
				constants.halfExtent[0] = pPipewireTexture->width() / 2.0f;
				constants.halfExtent[1] = pPipewireTexture->height() / 2.0f;
				cmdBuffer->uploadConstants<CaptureConvertBlitData_t>(constants);
			}
			else
			{
				BlitPushData_t constants( scale );
				cmdBuffer->uploadConstants<BlitPushData_t>(constants);
			}

			for (uint32_t i = 0; i < EOTF_Count; i++)
				cmdBuffer->bindColorMgmtLuts(i, nullptr, nullptr);

			cmdBuffer->bindPipeline(g_device.pipeline( ycbcr ? SHADER_TYPE_RGB_TO_NV12 : blit_shader_for_target(pPipewireTexture.get()), 1, 0, 0, GAMESCOPE_APP_TEXTURE_COLORSPACE_SRGB, EOTF_Count ));
			cmdBuffer->bindTexture(0, compositeImage);
			cmdBuffer->setTextureSrgb(0, true);
			cmdBuffer->setSamplerNearest(0, false);
			cmdBuffer->setSamplerUnnormalized(0, true);
			for (uint32_t i = 1; i < VKR_SAMPLER_SLOTS; i++)
			{
				cmdBuffer->bindTexture(i, nullptr);
			}
			cmdBuffer->bindTarget(pPipewireTexture);

			const int pixelsPerGroup = 8;

			// For ycbcr, we operate on 2 pixels at a time, so use the half-extent.
			const int dispatchSize = ycbcr ? pixelsPerGroup * 2 : pixelsPerGroup;

			cmdBuffer->dispatch(div_roundup(pPipewireTexture->width(), dispatchSize), div_roundup(pPipewireTexture->height(), dispatchSize));
		}

		if ( pPipewireTexture->isReadback() )
			cmdBuffer->copyImageToReadback(pPipewireTexture);
	}

	if ( compositeImage != scanout )
		cmdBuffer->copyImage(compositeImage, scanout);

	uint64_t sequence = g_device.submit(std::move(cmdBuffer), !GetBackend()->UsesVulkanSwapchain() && !pOutputOverride);

	if ( !GetBackend()->UsesVulkanSwapchain() && pOutputOverride == nullptr && increment )
	{
		g_output.nOutImage = ( g_output.nOutImage + 1 ) % 3;
	}

	return sequence;
}

namespace
{
	struct FrameGenerationQueuedFrame
	{
		FrameInfo_t::Layer_t layer;
		bool generated = false;
		bool useFSR = false;
		std::shared_ptr<gamescope::CAcquireTimelinePoint> acquirePoint;
		uint64_t outputId = 0;
		uint64_t submissionSequence = 0;
	};

	static bool FrameGenerationSupportedFormat( VkFormat format )
	{
		return format == VK_FORMAT_B8G8R8A8_UNORM || format == VK_FORMAT_B8G8R8A8_SRGB ||
			format == VK_FORMAT_R8G8B8A8_UNORM || format == VK_FORMAT_R8G8B8A8_SRGB ||
			format == VK_FORMAT_A2B10G10R10_UNORM_PACK32 ||
			format == VK_FORMAT_A2R10G10B10_UNORM_PACK32 ||
			format == VK_FORMAT_R16G16B16A16_SFLOAT;
	}

	static void FrameGenerationBindGamma22HdrLuts( FrameInfo_t &frameInfo )
	{
		if ( !frameInfo.applyOutputColorMgmt || frameInfo.outputEncodingEOTF != EOTF_Gamma22 ||
			 !ColorspaceIsHDR( frameInfo.layers[0].colorspace ) )
			return;

		// scRGB is in units of 80 nits, not normalized display luminance. The
		// HDR LUT maps it to the panel's brightness/gamut before gamma encoding;
		// encoding scRGB directly clips everything above 80 nits in UNORM.
		// paint_all binds these LUTs only after frame generation, so take them
		// from the active color-management state here. Leave PQ output alone.
		if ( g_ColorMgmtLuts[EOTF_PQ].HasLuts() )
		{
			frameInfo.shaperLut[EOTF_PQ] = g_ColorMgmtLuts[EOTF_PQ].vk_lut1d;
			frameInfo.lut3D[EOTF_PQ] = g_ColorMgmtLuts[EOTF_PQ].vk_lut3d;
		}
	}

	class VulkanFrameGenerationPacingContext
	{
	public:
		void Reset()
		{
			m_history = nullptr;
			RetireScanouts();
			m_fsrIntermediate = nullptr;
			m_queue.clear();
			m_previousReal = {};
			m_pipelinePrimed = false;
			m_lastSourceId = 0;
			m_lastSourceValid = false;
			m_currentRealSourceId = 0;
			m_currentRealOutputId = 0;
			m_lastSourceFrameTime = 0;
			m_hasHistory = false;
			m_lastPresentedGenerated = false;
			m_width = 0;
			m_height = 0;
			m_format = VK_FORMAT_UNDEFINED;
			m_drmFormat = DRM_FORMAT_INVALID;
			m_appId = 0;
			m_connector = nullptr;
			m_config = {};
			m_fsrEnabled = false;
			vulkan_frame_generation_reset_optical_flow();
		}

		bool HasPendingFrame() const
		{
			return !m_queue.empty();
		}

		bool IsDraining() const
		{
			return !m_queue.empty();
		}

		bool CanRequestSourceFrame() const
		{
			return gamescope::FrameGenerationCanRequestSourceFrame(
				uint32_t( m_queue.size() ),
				!m_queue.empty() && m_queue.front().generated );
		}

		bool CanAcceptSourceFrame() const
		{
			return gamescope::FrameGenerationCanAcceptSourceFrame(
				uint32_t( m_queue.size() ),
				!m_queue.empty() && m_queue.front().generated );
		}

		bool DropStaleGeneratedFrame(
			uint64_t now, uint64_t deadline, uint64_t interval )
		{
			if ( m_queue.empty() ||
				!gamescope::FrameGenerationShouldDropStaleGenerated(
					m_queue.front().generated, now, deadline, interval ) )
				return false;

			m_queue.pop_front();
			m_telemetry.deadlineDroppedFrames++;
			gamescope::SetFrameGenerationStatus( gamescope::FrameGenerationStatus::DeadlineMiss );
			return true;
		}

		bool LastPresentedGenerated() const
		{
			return m_lastPresentedGenerated;
		}

		FrameGenerationTelemetry Telemetry() const
		{
			FrameGenerationTelemetry result = m_telemetry;
			result.gpu = vulkan_frame_generation_get_gpu_timings();
			result.sceneCutCopies = vulkan_frame_generation_get_scene_cut_copy_count();
			result.queuedFrames = uint32_t( m_queue.size() );
			result.flowScalePercent = m_config.flowScalePercent;
			return result;
		}

		void NoteSourceCallback( bool blocked )
		{
			if ( blocked )
				m_telemetry.sourceCallbacksBlocked++;
			else
				m_telemetry.sourceCallbacks++;
		}

		void NoteOutputSlot( bool hasPendingFrame )
		{
			m_telemetry.outputSlots++;
			if ( !hasPendingFrame )
				m_telemetry.outputSlotsWithoutPending++;
		}

		void Apply( FrameInfo_t *frameInfo, gamescope::Rc<CVulkanTexture> rawSource,
			uint64_t sourceId, uint32_t appId, bool fsrEnabled, bool prepareOnly )
		{
			CollectRetiredScanouts();
			const gamescope::FrameGenerationConfig config = gamescope::GetFrameGenerationConfig();
			if ( !config.enabled )
			{
				if ( m_lastSourceValid || m_hasHistory || !m_queue.empty() )
					Reset();
				gamescope::SetFrameGenerationStatus( gamescope::FrameGenerationStatus::Disabled );
				return;
			}

			if ( !frameInfo || frameInfo->layers.count() < 1 || !frameInfo->layers[0].tex || !rawSource )
				return;

			FrameInfo_t::Layer_t &realLayer = frameInfo->layers[0];
			if ( rawSource->isYcbcr() || !FrameGenerationSupportedFormat( rawSource->format() ) )
			{
				Reset();
				gamescope::SetFrameGenerationStatus( gamescope::FrameGenerationStatus::UnsupportedFormat );
				return;
			}

			const bool stateChanged =
				m_width != rawSource->width() || m_height != rawSource->height() ||
				m_format != rawSource->format() || m_drmFormat != rawSource->drmFormat() ||
				m_colorspace != realLayer.colorspace || m_appId != appId ||
				m_connector != GetBackend()->GetCurrentConnector() || m_config != config ||
				m_fsrEnabled != fsrEnabled || m_sgsrEnabled != frameInfo->useSGSRLayer0 ||
				m_scale.x != realLayer.scale.x || m_scale.y != realLayer.scale.y ||
				m_offset.x != realLayer.offset.x || m_offset.y != realLayer.offset.y;
			if ( stateChanged )
			{
				Reset();
				m_width = rawSource->width();
				m_height = rawSource->height();
				m_format = rawSource->format();
				m_drmFormat = rawSource->drmFormat();
				m_colorspace = realLayer.colorspace;
				m_appId = appId;
				m_connector = GetBackend()->GetCurrentConnector();
				m_config = config;
				m_fsrEnabled = fsrEnabled;
				m_sgsrEnabled = frameInfo->useSGSRLayer0;
				m_scale = realLayer.scale;
				m_offset = realLayer.offset;
			}

			frameInfo->frameGenerationActive = true;
			frameInfo->frameGenerationOutputId = m_currentRealOutputId;

			if ( m_lastSourceValid && sourceId == m_lastSourceId )
			{
				if ( !prepareOnly )
					PresentQueued( frameInfo );
				return;
			}

			if ( sourceId != m_currentRealSourceId )
			{
				m_currentRealSourceId = sourceId;
				m_currentRealOutputId = ++m_nextOutputId;
			}
			frameInfo->frameGenerationOutputId = m_currentRealOutputId;

			if ( !CanAcceptSourceFrame() )
			{
				m_telemetry.deadlineDroppedFrames += std::count_if(
					m_queue.begin(), m_queue.end(),
					[]( const FrameGenerationQueuedFrame &frame ) { return frame.generated; } );
				m_queue.clear();
				m_pipelinePrimed = false;
				gamescope::SetFrameGenerationStatus( gamescope::FrameGenerationStatus::DeadlineMiss );
			}

			if ( !vulkan_frame_generation_work_complete() )
			{
				m_telemetry.deadlineDroppedFrames++;
				gamescope::SetFrameGenerationStatus( gamescope::FrameGenerationStatus::DeadlineMiss );
				return;
			}

			// A gap in source delivery can drain the complete pipeline. Hold the
			// previous real frame again so the next midpoint still gets one full
			// source interval instead of racing its first output deadline.
			if ( m_pipelinePrimed && m_queue.empty() )
			{
				m_pipelinePrimed = false;
				m_telemetry.queueUnderruns++;
			}

			if ( !m_history && !AllocateHistory( rawSource ) )
			{
				gamescope::SetFrameGenerationStatus( gamescope::FrameGenerationStatus::UnsupportedGPU );
				return;
			}

			auto cmdBuffer = g_device.commandBuffer();
			const bool reset = !m_hasHistory;
			if ( !vulkan_frame_generation_record_optical_flow(
					cmdBuffer.get(), rawSource, realLayer.colorspace,
					config.flowScalePercent, reset ) )
			{
				gamescope::SetFrameGenerationStatus( gamescope::FrameGenerationStatus::UnsupportedGPU );
				return;
			}

			m_lastSourceId = sourceId;
			m_lastSourceValid = true;
			m_telemetry.sourceFrames++;
			const uint64_t sourceFrameTime = get_time_in_nanos();
			if ( m_lastSourceFrameTime != 0 && sourceFrameTime >= m_lastSourceFrameTime )
			{
				m_telemetry.sourceFrameIntervalMilliseconds =
					double( sourceFrameTime - m_lastSourceFrameTime ) / 1'000'000.0;
			}
			m_lastSourceFrameTime = sourceFrameTime;

			if ( !m_hasHistory )
			{
				cmdBuffer->copyImage( rawSource, m_history );
				const uint64_t sequence = g_device.submit( std::move( cmdBuffer ) );
				vulkan_frame_generation_notify_submit( sequence );
				m_hasHistory = true;
				m_previousReal = {
					realLayer, false, frameInfo->useFSRLayer0, nullptr,
					m_currentRealOutputId };
				gamescope::SetFrameGenerationStatus( gamescope::FrameGenerationStatus::WarmingUp );
				return;
			}

			const bool abDuplicate = vulkan_frame_generation_ab_enabled();
			// Direct output is only final when this texture can itself be presented.
			// Backend-allocated scanout requires a later sampling/composite pass and
			// therefore must use the synchronized non-direct path below.
			const bool directOutput =
				!abDuplicate && !fsrEnabled && !ColorspaceIsHDR( realLayer.colorspace ) &&
				!GetBackend()->UsesBackendAllocatedScanout() &&
				vulkan_frame_generation_direct_output_enabled();
			gamescope::Rc<CVulkanTexture> scanout;
			if ( directOutput )
			{
				scanout = AcquireScanout(
					rawSource->width(), rawSource->height(), rawSource->drmFormat() );
				if ( !scanout )
				{
					gamescope::SetFrameGenerationStatus( gamescope::FrameGenerationStatus::UnsupportedGPU );
					return;
				}
			}

			gamescope::Rc<CVulkanTexture> midpoint =
				vulkan_frame_generation_record_interpolation(
					cmdBuffer.get(), m_history, rawSource, false, scanout );
			if ( !midpoint )
			{
				gamescope::SetFrameGenerationStatus( gamescope::FrameGenerationStatus::UnsupportedGPU );
				return;
			}

			cmdBuffer->copyImage( rawSource, m_history );

			FrameGenerationQueuedFrame generated = {};
			generated.generated = true;
			generated.layer = realLayer;
			generated.useFSR = false;
			generated.outputId = ++m_nextOutputId;
			FrameGenerationQueuedFrame real = {
				realLayer, false, frameInfo->useFSRLayer0, nullptr,
				m_currentRealOutputId };
			const bool hdrOutput = ColorspaceIsHDR( realLayer.colorspace );

			std::optional<uint64_t> sequence;
			if ( abDuplicate )
			{
				generated.layer = realLayer;
				vulkan_frame_generation_record_output_timestamp( cmdBuffer.get(), false );
				sequence = g_device.submit( std::move( cmdBuffer ) );
			}
			else
			{
				const uint32_t outputWidth = fsrEnabled ? uint32_t( g_nOutputWidth ) : rawSource->width();
				const uint32_t outputHeight = fsrEnabled ? uint32_t( g_nOutputHeight ) : rawSource->height();
				const uint32_t outputFormat = hdrOutput
					? DRM_FORMAT_XBGR2101010
					: fsrEnabled
					? ( realLayer.tex->drmFormat() != DRM_FORMAT_INVALID
						? realLayer.tex->drmFormat() : g_output.uOutputFormat )
					: rawSource->drmFormat();
				if ( !scanout )
					scanout = AcquireScanout( outputWidth, outputHeight, outputFormat );
				if ( !scanout )
				{
					gamescope::SetFrameGenerationStatus( gamescope::FrameGenerationStatus::UnsupportedGPU );
					return;
				}

				FrameInfo_t generatedFrame = {};
				generatedFrame.layers.push();
				generatedFrame.frameGenerationActive = true;
				generatedFrame.layers[0] = realLayer;
				generatedFrame.layers[0].tex = midpoint;
				if ( ColorspaceIsHDR( realLayer.colorspace ) )
					generatedFrame.layers[0].colorspace = GAMESCOPE_APP_TEXTURE_COLORSPACE_SCRGB;
				generatedFrame.layers[0].acquirePoint = nullptr;
				generatedFrame.layers[0].opacity = 1.0f;
				generatedFrame.useFSRLayer0 = fsrEnabled && !frameInfo->useSGSRLayer0;
				generatedFrame.useSGSRLayer0 = fsrEnabled && frameInfo->useSGSRLayer0;
				generatedFrame.nUpscaleSharpness = frameInfo->nUpscaleSharpness;
				generatedFrame.applyOutputColorMgmt = fsrEnabled
					? frameInfo->applyOutputColorMgmt : false;
				generatedFrame.outputEncodingEOTF = fsrEnabled
					? frameInfo->outputEncodingEOTF : EOTF_Count;

				gamescope::Rc<CVulkanTexture> fsrIntermediate;
				if ( fsrEnabled )
				{
					// The real layer may already contain Gamescope's preemptive FSR
					// result. Restore the raw midpoint's scale while preserving the
					// real frame's presentation extent, otherwise EASU remains at raw
					// resolution and RCAS reads beyond its intermediate image.
					const uint32_t presentationWidth = realLayer.integerWidth();
					const uint32_t presentationHeight = realLayer.integerHeight();
					if ( presentationWidth == 0 || presentationHeight == 0 )
					{
						gamescope::SetFrameGenerationStatus( gamescope::FrameGenerationStatus::UnsupportedFormat );
						return;
					}
					generatedFrame.layers[0].scale = {
						float( midpoint->width() ) / float( presentationWidth ),
						float( midpoint->height() ) / float( presentationHeight ),
					};
					if ( presentationWidth == uint32_t( g_nOutputWidth ) &&
						presentationHeight == uint32_t( g_nOutputHeight ) &&
						close_enough( generatedFrame.layers[0].offset.x, 0.0f ) &&
						close_enough( generatedFrame.layers[0].offset.y, 0.0f ) )
					{
						generatedFrame.layers[0].blackBorder = false;
					}

					const uint32_t contentWidth = generatedFrame.layers[0].integerWidth();
					const uint32_t contentHeight = generatedFrame.layers[0].integerHeight();
					fsrIntermediate = AcquireFsrIntermediate(
						contentWidth, contentHeight, ColorspaceIsHDR( realLayer.colorspace ) );
					if ( !fsrIntermediate )
					{
						gamescope::SetFrameGenerationStatus( gamescope::FrameGenerationStatus::UnsupportedGPU );
						return;
					}
				}
				else
				{
					// Preserve SDR midpoint values verbatim. HDR midpoints are linear
					// scRGB, so encode them into the display EOTF while writing the
					// native 10-bit scanout.
					if ( !hdrOutput )
					{
						generatedFrame.layers[0].colorspace = GAMESCOPE_APP_TEXTURE_COLORSPACE_LINEAR;
						generatedFrame.layers[0].ctm = nullptr;
					}
					else
					{
						// The midpoint remains linear scRGB with Rec.709 primaries.
						// Keep the source layer's 709 -> 2020 CTM when encoding it
						// for the HDR scanout, matching the FSR and direct-scanout paths.
						generatedFrame.applyOutputColorMgmt = true;
						generatedFrame.outputEncodingEOTF = frameInfo->outputEncodingEOTF;
					}
					generatedFrame.layers[0].scale = { 1.0f, 1.0f };
					generatedFrame.layers[0].offset = { 0.0f, 0.0f };
					generatedFrame.layers[0].blackBorder = false;
				}

				FrameGenerationBindGamma22HdrLuts( generatedFrame );

				if ( !m_completionTimeline )
					m_completionTimeline = gamescope::CTimeline::Create();
				if ( !m_completionTimeline )
				{
					gamescope::SetFrameGenerationStatus( gamescope::FrameGenerationStatus::UnsupportedGPU );
					return;
				}

				// The FidelityFX passes use their own raw image barriers while the
				// compositor uses CVulkanCmdBuffer's image-state tracker.  Keeping
				// interpolation and composition in one command buffer can leave the
				// midpoint's final storage write invisible to the composite.  Split at
				// that API boundary and complete interpolation before recording any
				// compositor work.  Direct output needs no second submission.
				if ( !directOutput )
				{
					const uint64_t completionPoint = ++m_completionPoint;
					auto compositeCommandBuffer = g_device.commandBuffer();
					if ( !compositeCommandBuffer->AddSignal(
							m_completionTimeline->ToVkSemaphore(), completionPoint ) )
					{
						gamescope::SetFrameGenerationStatus( gamescope::FrameGenerationStatus::UnsupportedGPU );
						return;
					}

					const uint64_t interpolationSequence =
						g_device.submit( std::move( cmdBuffer ) );
					vulkan_wait( interpolationSequence, false );
					cmdBuffer = std::move( compositeCommandBuffer );
					generated.acquirePoint = std::make_shared<gamescope::CAcquireTimelinePoint>(
						m_completionTimeline, completionPoint );
				}
				else
				{
					const uint64_t completionPoint = ++m_completionPoint;
					if ( !cmdBuffer->AddSignal(
							m_completionTimeline->ToVkSemaphore(), completionPoint ) )
					{
						gamescope::SetFrameGenerationStatus( gamescope::FrameGenerationStatus::UnsupportedGPU );
						return;
					}
					generated.acquirePoint = std::make_shared<gamescope::CAcquireTimelinePoint>(
						m_completionTimeline, completionPoint );
				}

				if ( directOutput )
				{
					vulkan_frame_generation_record_output_timestamp( cmdBuffer.get(), false );
					sequence = g_device.submit( std::move( cmdBuffer ) );
				}
				else
				{
					sequence = vulkan_composite( &generatedFrame, nullptr, false, scanout,
						false, std::move( cmdBuffer ), fsrIntermediate );
				}
				if ( !sequence )
				{
					gamescope::SetFrameGenerationStatus( gamescope::FrameGenerationStatus::UnsupportedGPU );
					return;
				}

				generated.layer.tex = scanout;
				if ( generatedFrame.applyOutputColorMgmt && generatedFrame.outputEncodingEOTF == EOTF_Gamma22 )
				{
					generated.layer.colorspace = GAMESCOPE_APP_TEXTURE_COLORSPACE_SRGB;
					generated.layer.outputColorMgmtApplied = true;
				}
				else if ( ColorspaceIsHDR( realLayer.colorspace ) )
					generated.layer.colorspace = generatedFrame.outputEncodingEOTF == EOTF_PQ
						? GAMESCOPE_APP_TEXTURE_COLORSPACE_HDR10_PQ
						: GAMESCOPE_APP_TEXTURE_COLORSPACE_SCRGB;
				if ( fsrEnabled )
				{
					generated.layer.scale = { 1.0f, 1.0f };
					generated.layer.offset = { 0.0f, 0.0f };
					generated.layer.blackBorder = false;
					generated.layer.filter = GamescopeUpscaleFilter::NEAREST;
				}
				if ( generatedFrame.applyOutputColorMgmt )
				{
					// This scanout already contains the compositor's display-encoded
					// result. Keep its output colorspace and HDR metadata, but do not
					// apply the source plane's color pipeline a second time in KMS.
					generated.layer.applyColorMgmt = false;
					generated.layer.ctm = nullptr;
				}

				// Keep both halves of an HDR pair in the same display-encoded scanout
				// format. Alternating the generated scanout with the raw HDR layer
				// otherwise makes KMS reprogram the plane color pipeline every output
				// slot, which is slow enough to destroy VRR frame pacing. FSR already
				// requires this precomposition; HDR requires it even without FSR.
				// Submit the real result after the midpoint so the earlier deadline
				// retains first use of the GPU.
				// Backend-allocated scanout paths always perform a final Vulkan
				// composition, so they do not alternate KMS plane color state and do
				// not need a second frame-generation intermediate for the real slot.
				if ( !GetBackend()->UsesBackendAllocatedScanout() &&
					 gamescope::FrameGenerationShouldPrecomposeRealFrame(
						fsrEnabled, hdrOutput, realLayer.tex.get() == rawSource.get() ) )
				{
					gamescope::Rc<CVulkanTexture> realScanout = AcquireScanout(
						outputWidth, outputHeight, outputFormat, scanout.get() );
					if ( !realScanout )
					{
						gamescope::SetFrameGenerationStatus( gamescope::FrameGenerationStatus::UnsupportedGPU );
						return;
					}

					FrameInfo_t realFrame = {};
					realFrame.layers.push();
					realFrame.layers[0] = realLayer;
					realFrame.layers[0].acquirePoint = nullptr;
					realFrame.layers[0].opacity = 1.0f;
					realFrame.useFSRLayer0 = fsrEnabled && !frameInfo->useSGSRLayer0;
					realFrame.useSGSRLayer0 = fsrEnabled && frameInfo->useSGSRLayer0;
					realFrame.nUpscaleSharpness = frameInfo->nUpscaleSharpness;
					realFrame.applyOutputColorMgmt = frameInfo->applyOutputColorMgmt;
					realFrame.outputEncodingEOTF = frameInfo->outputEncodingEOTF;
					if ( fsrEnabled )
					{
						if ( realFrame.layers[0].integerWidth() == uint32_t( g_nOutputWidth ) &&
							realFrame.layers[0].integerHeight() == uint32_t( g_nOutputHeight ) &&
							close_enough( realFrame.layers[0].offset.x, 0.0f ) &&
							close_enough( realFrame.layers[0].offset.y, 0.0f ) )
						{
							realFrame.layers[0].blackBorder = false;
						}
					}
					else
					{
						// The target has the raw source extent. Composite the real frame
						// one-to-one, then retain its original presentation scale below.
						// Preserve the HDR source CTM just as the FSR path does.
						realFrame.layers[0].scale = { 1.0f, 1.0f };
						realFrame.layers[0].offset = { 0.0f, 0.0f };
						realFrame.layers[0].blackBorder = false;
					}

					FrameGenerationBindGamma22HdrLuts( realFrame );

					auto realCmdBuffer = g_device.commandBuffer();
					const uint64_t realCompletionPoint = ++m_completionPoint;
					realCmdBuffer->AddSignal(
						m_completionTimeline->ToVkSemaphore(), realCompletionPoint );
					real.acquirePoint = std::make_shared<gamescope::CAcquireTimelinePoint>(
						m_completionTimeline, realCompletionPoint );

					std::optional<uint64_t> realSequence = vulkan_composite(
						&realFrame, nullptr, false, realScanout, false,
						std::move( realCmdBuffer ), fsrIntermediate );
					if ( !realSequence )
					{
						gamescope::SetFrameGenerationStatus( gamescope::FrameGenerationStatus::UnsupportedGPU );
						return;
					}
					real.layer.tex = realScanout;
					if ( fsrEnabled )
					{
						real.layer.scale = { 1.0f, 1.0f };
						real.layer.offset = { 0.0f, 0.0f };
						real.layer.blackBorder = false;
						real.layer.filter = GamescopeUpscaleFilter::NEAREST;
					}
					real.useFSR = false;
					if ( realFrame.outputEncodingEOTF == EOTF_Gamma22 )
					{
						real.layer.colorspace = GAMESCOPE_APP_TEXTURE_COLORSPACE_SRGB;
						real.layer.outputColorMgmtApplied = realFrame.applyOutputColorMgmt;
					}
					else if ( ColorspaceIsHDR( realLayer.colorspace ) )
						real.layer.colorspace = realFrame.outputEncodingEOTF == EOTF_PQ
							? GAMESCOPE_APP_TEXTURE_COLORSPACE_HDR10_PQ
							: GAMESCOPE_APP_TEXTURE_COLORSPACE_SCRGB;
					if ( realFrame.applyOutputColorMgmt )
					{
						real.layer.applyColorMgmt = false;
						real.layer.ctm = nullptr;
					}
				}
			}

			vulkan_frame_generation_notify_submit( *sequence );
			generated.submissionSequence = *sequence;
			if ( !m_pipelinePrimed )
			{
				m_queue.push_back( m_previousReal );
				m_pipelinePrimed = true;
				m_telemetry.repeatedRealFrames++;
			}
			m_queue.push_back( std::move( generated ) );
			m_previousReal = real;
			m_queue.push_back( std::move( real ) );
			m_telemetry.generatedFrames++;
			gamescope::SetFrameGenerationStatus( gamescope::FrameGenerationStatus::Active );

			if ( !prepareOnly )
				PresentQueued( frameInfo );
		}

	private:
		bool AllocateHistory( const gamescope::Rc<CVulkanTexture> &source )
		{
			uint32_t drmFormat = source->drmFormat();
			if ( drmFormat == DRM_FORMAT_INVALID )
				drmFormat = VulkanFormatToDRM( source->format(), true );
			if ( drmFormat == DRM_FORMAT_INVALID )
				return false;

			CVulkanTexture::createFlags flags;
			flags.bSampled = true;
			flags.bTransferDst = true;
			flags.bTransferSrc = true;
			m_history = new CVulkanTexture;
			return m_history->BInit( source->width(), source->height(), 1, drmFormat, flags );
		}

		gamescope::Rc<CVulkanTexture> AcquireScanout(
			uint32_t width, uint32_t height, uint32_t drmFormat,
			const CVulkanTexture *exclude = nullptr )
		{
			CollectRetiredScanouts();
			if ( drmFormat == DRM_FORMAT_INVALID )
				return nullptr;
			if ( !m_scanoutImages.empty() &&
				( m_scanoutImages.front()->width() != width ||
				  m_scanoutImages.front()->height() != height ||
				  m_scanoutImages.front()->drmFormat() != drmFormat ) )
				RetireScanouts();

			for ( auto &image : m_scanoutImages )
			{
				const bool queued = std::any_of(
					m_queue.begin(), m_queue.end(),
					[&image]( const FrameGenerationQueuedFrame &frame ) {
						return frame.layer.tex.get() == image.get();
					} );
				if ( image.get() != exclude &&
					image.get() != m_previousReal.layer.tex.get() &&
					!queued && !image->IsInUse() )
					return image;
			}
			if ( m_scanoutImages.size() >= 4 )
				return nullptr;

			CVulkanTexture::createFlags flags;
			// Backend-allocated scanout paths (currently NVIDIA DRM) always
			// composite into their own final buffer. Keep frame-generation output
			// device-local there: exporting it and transferring ownership to the
			// foreign queue family is both unnecessary and breaks the subsequent
			// Vulkan sampling pass on NVIDIA.
			flags.bFlippable = !GetBackend()->UsesBackendAllocatedScanout();
			flags.bSampled = true;
			flags.bStorage = true;
			flags.bTransferSrc = true;
			gamescope::OwningRc<CVulkanTexture> image = new CVulkanTexture;
			if ( !image->BInit( width, height, 1, drmFormat, flags ) )
				return nullptr;
			if ( flags.bFlippable && GetBackend()->IsSessionBased() && !image->GetBackendFb() )
				return nullptr;
			m_scanoutImages.push_back( std::move( image ) );
			return m_scanoutImages.back();
		}

		void RetireScanouts()
		{
			for ( auto &image : m_scanoutImages )
				m_retiredScanoutImages.push_back( std::move( image ) );
			m_scanoutImages.clear();
		}

		void CollectRetiredScanouts()
		{
			std::erase_if( m_retiredScanoutImages,
				[]( const gamescope::OwningRc<CVulkanTexture> &image ) {
					return image && !image->IsInUse();
				} );
		}

		gamescope::Rc<CVulkanTexture> AcquireFsrIntermediate(
			uint32_t width, uint32_t height, bool hdr )
		{
			const uint32_t drmFormat = hdr ? DRM_FORMAT_ABGR16161616F : DRM_FORMAT_ARGB8888;
			if ( m_fsrIntermediate && m_fsrIntermediate->width() == width &&
				m_fsrIntermediate->height() == height &&
				m_fsrIntermediate->drmFormat() == drmFormat )
				return m_fsrIntermediate;
			CVulkanTexture::createFlags flags;
			flags.bSampled = true;
			flags.bStorage = true;
			m_fsrIntermediate = new CVulkanTexture;
			if ( !m_fsrIntermediate->BInit( width, height, 1, drmFormat, flags ) )
			{
				m_fsrIntermediate = nullptr;
				return nullptr;
			}
			return m_fsrIntermediate;
		}

		void PresentQueued( FrameInfo_t *frameInfo )
		{
			m_lastPresentedGenerated = false;
			if ( m_queue.empty() )
				return;

			if ( m_queue.front().generated &&
				 m_queue.front().submissionSequence != 0 &&
				 !g_device.isComplete( m_queue.front().submissionSequence ) )
			{
				m_queue.pop_front();
				m_telemetry.deadlineDroppedFrames++;
				gamescope::SetFrameGenerationStatus( gamescope::FrameGenerationStatus::DeadlineMiss );
				if ( m_queue.empty() )
					return;
			}

			FrameGenerationQueuedFrame queued = std::move( m_queue.front() );
			m_queue.pop_front();
			frameInfo->layers[0] = std::move( queued.layer );
			frameInfo->layers[0].acquirePoint = std::move( queued.acquirePoint );
			frameInfo->useFSRLayer0 = queued.useFSR;
			frameInfo->useNISLayer0 = false;
			frameInfo->useSGSRLayer0 = false;
			frameInfo->frameGenerationActive = true;
			frameInfo->frameGenerationOutputId = queued.outputId;
			m_lastPresentedGenerated = queued.generated;
			m_telemetry.presentedGeneratedFrames += queued.generated ? 1u : 0u;
			m_telemetry.presentedRealFrames += queued.generated ? 0u : 1u;
		}

		gamescope::OwningRc<CVulkanTexture> m_history;
		std::vector<gamescope::OwningRc<CVulkanTexture>> m_scanoutImages;
		std::vector<gamescope::OwningRc<CVulkanTexture>> m_retiredScanoutImages;
		gamescope::OwningRc<CVulkanTexture> m_fsrIntermediate;
		std::deque<FrameGenerationQueuedFrame> m_queue;
		std::shared_ptr<gamescope::CTimeline> m_completionTimeline;
		uint64_t m_completionPoint = 0;
		gamescope::FrameGenerationConfig m_config;
		gamescope::IBackendConnector *m_connector = nullptr;
		uint64_t m_lastSourceId = 0;
		uint64_t m_currentRealSourceId = 0;
		uint64_t m_currentRealOutputId = 0;
		uint64_t m_nextOutputId = 0;
		uint64_t m_lastSourceFrameTime = 0;
		uint32_t m_appId = 0;
		uint32_t m_width = 0;
		uint32_t m_height = 0;
		uint32_t m_drmFormat = DRM_FORMAT_INVALID;
		VkFormat m_format = VK_FORMAT_UNDEFINED;
		GamescopeAppTextureColorspace m_colorspace = GAMESCOPE_APP_TEXTURE_COLORSPACE_LINEAR;
		vec2_t m_scale = {};
		vec2_t m_offset = {};
		bool m_hasHistory = false;
		bool m_lastSourceValid = false;
		bool m_fsrEnabled = false;
		bool m_sgsrEnabled = false;
		bool m_lastPresentedGenerated = false;
		bool m_pipelinePrimed = false;
		FrameGenerationQueuedFrame m_previousReal;
		FrameGenerationTelemetry m_telemetry;
	};

	VulkanFrameGenerationPacingContext s_frameGenerationPacingContext;
}

void vulkan_frame_generation_apply( FrameInfo_t *frameInfo,
	gamescope::Rc<CVulkanTexture> rawSource, uint64_t sourceId, uint32_t appId,
	bool prepareOnly )
{
	bool fsrEnabled = false;
	if ( frameInfo && frameInfo->layers.count() > 0 )
	{
		fsrEnabled = frameInfo->useFSRLayer0 || frameInfo->useSGSRLayer0 ||
			( rawSource && frameInfo->layers[0].tex.get() != rawSource.get() &&
			  frameInfo->layers[0].filter == GamescopeUpscaleFilter::FSR );
	}
	s_frameGenerationPacingContext.Apply(
		frameInfo, std::move( rawSource ), sourceId, appId, fsrEnabled, prepareOnly );
}

bool vulkan_frame_generation_has_pending_frame()
{
	return s_frameGenerationPacingContext.HasPendingFrame();
}

bool vulkan_frame_generation_is_draining()
{
	return s_frameGenerationPacingContext.IsDraining();
}

bool vulkan_frame_generation_can_request_source_frame()
{
	return s_frameGenerationPacingContext.CanRequestSourceFrame();
}

bool vulkan_frame_generation_can_accept_source_frame()
{
	return s_frameGenerationPacingContext.CanAcceptSourceFrame();
}

bool vulkan_frame_generation_drop_stale_generated_frame(
	uint64_t now, uint64_t deadline, uint64_t interval )
{
	return s_frameGenerationPacingContext.DropStaleGeneratedFrame(
		now, deadline, interval );
}

bool vulkan_frame_generation_last_presented_generated()
{
	return s_frameGenerationPacingContext.LastPresentedGenerated();
}

void vulkan_frame_generation_reset()
{
	s_frameGenerationPacingContext.Reset();
}

FrameGenerationTelemetry vulkan_frame_generation_get_telemetry()
{
	return s_frameGenerationPacingContext.Telemetry();
}

void vulkan_frame_generation_note_source_callback( bool blocked )
{
	s_frameGenerationPacingContext.NoteSourceCallback( blocked );
}

void vulkan_frame_generation_note_output_slot( bool hasPendingFrame )
{
	s_frameGenerationPacingContext.NoteOutputSlot( hasPendingFrame );
}

void vulkan_wait( uint64_t ulSeqNo, bool bReset )
{
	return g_device.wait( ulSeqNo, bReset );
}

bool vulkan_has_drm_props()
{
	for (const auto& ext : g_device.supportedExtensions()) {
		if ( strcmp(ext.extensionName, VK_EXT_PHYSICAL_DEVICE_DRM_EXTENSION_NAME) == 0 )
			return true;
	}

	return false;
}

bool vulkan_format_supports_features(VkFormat format, VkFormatFeatureFlags features)
{
	VkDrmFormatModifierPropertiesListEXT modifierPropList = {
		.sType = VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_EXT,
	};
	VkFormatProperties2 formatProps = {
		.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2,
		.pNext = g_device.supportsModifiers() ? &modifierPropList : nullptr,
	};

	g_device.vk.GetPhysicalDeviceFormatProperties2( g_device.physDev(), format, &formatProps );

	// Without modifiers, flippable images are created with optimal tiling
	if ( !g_device.supportsModifiers() )
		return ( formatProps.formatProperties.optimalTilingFeatures & features ) == features;

	if ( modifierPropList.drmFormatModifierCount == 0 )
		return false;

	std::vector<VkDrmFormatModifierPropertiesEXT> modifierProps(modifierPropList.drmFormatModifierCount);
	modifierPropList.pDrmFormatModifierProperties = modifierProps.data();
	g_device.vk.GetPhysicalDeviceFormatProperties2( g_device.physDev(), format, &formatProps );

	for ( size_t j = 0; j < modifierProps.size(); j++ ) {
		if ( ( modifierProps[j].drmFormatModifierTilingFeatures & features ) == features )
			return true;
	}

	return false;
}

gamescope::Rc<CVulkanTexture> vulkan_get_last_output_image( bool partial, bool defer )
{
	// Get previous image ( +2 )
	// 1 2 3
	//   |
	// |
	uint32_t nRegularImage = ( g_output.nOutImage + 2 ) % 3;

	// Get previous previous image ( +1 )
	// 1 2 3
	//   |
	//     |
	uint32_t nDeferredImage = ( g_output.nOutImage + 1 ) % 3;

	uint32_t nOutImage = defer ? nDeferredImage : nRegularImage;

	if ( partial )
	{

		//vk_log.infof( "Partial overlay frame: %d", nDeferredImage );
		return g_output.outputImagesPartialOverlay[ nOutImage ];
	}


	return g_output.outputImages[ nOutImage ];
}

bool vulkan_primary_dev_id(dev_t *id)
{
	*id = g_device.primaryDevId();
	return g_device.hasDrmPrimaryDevId();
}

bool vulkan_supports_modifiers(void)
{
	return g_device.supportsModifiers();
}

static void texture_destroy( struct wlr_texture *wlr_texture )
{
	VulkanWlrTexture_t *tex = (VulkanWlrTexture_t *)wlr_texture;
	wlr_buffer_unlock( tex->buf );
	delete tex;
}

static const struct wlr_texture_impl texture_impl = {
	.destroy = texture_destroy,
};

static const struct wlr_drm_format_set *renderer_get_texture_formats( struct wlr_renderer *wlr_renderer, uint32_t buffer_caps )
{
	if (buffer_caps & WLR_BUFFER_CAP_DMABUF)
	{
		if ( !g_device.supportsClientDmabufs() || sampledDRMFormats.len == 0 )
			return nullptr;

		return &sampledDRMFormats;
	}
	else if (buffer_caps & WLR_BUFFER_CAP_DATA_PTR)
	{
		return &sampledShmFormats;
	}
	else
	{
		return nullptr;
	}
}

static int renderer_get_drm_fd( struct wlr_renderer *wlr_renderer )
{
	return g_device.drmRenderFd();
}

static struct wlr_texture *renderer_texture_from_buffer( struct wlr_renderer *wlr_renderer, struct wlr_buffer *buf )
{
	VulkanWlrTexture_t *tex = new VulkanWlrTexture_t();
	wlr_texture_init( &tex->base, wlr_renderer, &texture_impl, buf->width, buf->height );
	tex->buf = wlr_buffer_lock( buf );
	// TODO: check format/modifier
	// TODO: if DMA-BUF, try importing it into Vulkan
	return &tex->base;
}

static struct wlr_render_pass *renderer_begin_buffer_pass( struct wlr_renderer *renderer, struct wlr_buffer *buffer, const struct wlr_buffer_pass_options *options )
{
	abort(); // unreachable
}

static const struct wlr_renderer_impl renderer_impl = {
	.get_texture_formats = renderer_get_texture_formats,
	.get_drm_fd = renderer_get_drm_fd,
	.texture_from_buffer = renderer_texture_from_buffer,
	.begin_buffer_pass = renderer_begin_buffer_pass,
};

struct wlr_renderer *vulkan_renderer_create( void )
{
	VulkanRenderer_t *renderer = new VulkanRenderer_t();
	wlr_renderer_init(&renderer->base, &renderer_impl, WLR_BUFFER_CAP_DMABUF | WLR_BUFFER_CAP_DATA_PTR);
	return &renderer->base;
}

gamescope::OwningRc<CVulkanTexture> vulkan_create_texture_from_wlr_buffer( struct wlr_buffer *buf, gamescope::OwningRc<gamescope::IBackendFb> pBackendFb )
{

	struct wlr_dmabuf_attributes dmabuf = {0};
	if ( wlr_buffer_get_dmabuf( buf, &dmabuf ) )
	{
		return vulkan_create_texture_from_dmabuf( &dmabuf, pBackendFb );
	}

	VkResult result;

	void *src;
	uint32_t drmFormat;
	size_t stride;
	if ( !wlr_buffer_begin_data_ptr_access( buf, WLR_BUFFER_DATA_PTR_ACCESS_READ, &src, &drmFormat, &stride ) )
	{
		return nullptr;
	}

	uint32_t width = buf->width;
	uint32_t height = buf->height;

	VkBufferCreateInfo bufferCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = stride * height,
		.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
	};
	VkBuffer buffer;
	result = g_device.vk.CreateBuffer( g_device.device(), &bufferCreateInfo, nullptr, &buffer );
	if ( result != VK_SUCCESS )
	{
		wlr_buffer_end_data_ptr_access( buf );
		return nullptr;
	}

	VkMemoryRequirements memRequirements;
	g_device.vk.GetBufferMemoryRequirements(g_device.device(), buffer, &memRequirements);

	uint32_t memTypeIndex =  g_device.findMemoryType(VK_MEMORY_PROPERTY_HOST_COHERENT_BIT|VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, memRequirements.memoryTypeBits );
	if ( memTypeIndex == ~0u )
	{
		wlr_buffer_end_data_ptr_access( buf );
		return nullptr;
	}

	VkMemoryAllocateInfo allocInfo = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize = memRequirements.size,
		.memoryTypeIndex = memTypeIndex,
	};

	VkDeviceMemory bufferMemory;
	result = g_device.vk.AllocateMemory( g_device.device(), &allocInfo, nullptr, &bufferMemory);
	if ( result != VK_SUCCESS )
	{
		wlr_buffer_end_data_ptr_access( buf );
		return nullptr;
	}

	result = g_device.vk.BindBufferMemory( g_device.device(), buffer, bufferMemory, 0 );
	if ( result != VK_SUCCESS )
	{
		wlr_buffer_end_data_ptr_access( buf );
		return nullptr;
	}

	void *dst;
	result = g_device.vk.MapMemory( g_device.device(), bufferMemory, 0, VK_WHOLE_SIZE, 0, &dst );
	if ( result != VK_SUCCESS )
	{
		wlr_buffer_end_data_ptr_access( buf );
		return nullptr;
	}

	memcpy( dst, src, stride * height );

	g_device.vk.UnmapMemory( g_device.device(), bufferMemory );

	wlr_buffer_end_data_ptr_access( buf );

	gamescope::OwningRc<CVulkanTexture> pTex = new CVulkanTexture();
	CVulkanTexture::createFlags texCreateFlags;
	texCreateFlags.bSampled = true;
	texCreateFlags.bTransferDst = true;
	// When client DMA-BUFs are unavailable (currently Venus), this texture is
	// an upload-only fallback. Exporting every upload for direct scanout is both
	// unnecessary and rejected by virtio KMS; composition uses the internal
	// Vulkan image directly. Native drivers retain the flippable fast path.
	texCreateFlags.bFlippable = g_device.supportsClientDmabufs();
	if ( pTex->BInit( width, height, 1u, drmFormat, texCreateFlags, nullptr, 0, 0, nullptr, pBackendFb ) == false )
		return nullptr;

	auto cmdBuffer = g_device.commandBuffer();

	cmdBuffer->copyBufferToImage( buffer, 0, stride / DRMFormatGetBPP(drmFormat), pTex);
	// TODO: Sync this copyBufferToImage

	uint64_t sequence = g_device.submit(std::move(cmdBuffer));

	g_device.wait(sequence);

	g_device.vk.DestroyBuffer(g_device.device(), buffer, nullptr);
	g_device.vk.FreeMemory(g_device.device(), bufferMemory, nullptr);

	return pTex;
}
