// DRM output stuff

#include "Script/Script.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <stdlib.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <linux/dma-buf.h>

#include <atomic>
#include <cassert>
#include <cinttypes>
#include <chrono>
#include <climits>
#include <cstddef>
#include <cstdio>
#include <condition_variable>
#include <cstring>
#include <map>
#include <mutex>
#include <optional>
#include <span>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "backend.h"
#include "color_helpers.h"
#include "drm_color_helpers.h"
#include "Utils/Defer.h"
#include "drm_include.h"
#include "edid.h"
#include "gamescope_shared.h"
#include "gpuvis_trace_utils.h"
#include "log.hpp"
#include "main.hpp"
#include "mode_list_file.h"
#include "modegen.hpp"
#include "rendervulkan.hpp"
#include "steamcompmgr.hpp"
#include "vblankmanager.hpp"
#include "wlserver.hpp"
#include "refresh_rate.h"
#include <sys/utsname.h>

#include "wlr_begin.hpp"
#include <libliftoff.h>
#include <wlr/types/wlr_buffer.h>
#include "libdisplay-info/info.h"
#include "libdisplay-info/edid.h"
#include "wlr_end.hpp"

#if HAVE_GBM
#include <gbm.h>
#endif

#include "gamescope-control-protocol.h"

static constexpr bool k_bUseCursorPlane = false;

extern int g_nPreferredOutputWidth;
extern int g_nPreferredOutputHeight;

gamescope::ConVar<bool> cv_drm_single_plane_optimizations( "drm_single_plane_optimizations", true, "Whether or not to enable optimizations for single plane usage." );

gamescope::ConVar<bool> cv_drm_debug_disable_shaper_and_3dlut( "drm_debug_disable_shaper_and_3dlut", false, "Shaper + 3DLUT chicken bit. (Force disable/DEFAULT, no logic change)" );
gamescope::ConVar<bool> cv_drm_debug_disable_degamma_tf( "drm_debug_disable_degamma_tf", false, "Degamma chicken bit. (Forces DEGAMMA_TF to DEFAULT, does not affect other logic)" );
gamescope::ConVar<bool> cv_drm_debug_disable_regamma_tf( "drm_debug_disable_regamma_tf", false, "Regamma chicken bit. (Forces REGAMMA_TF to DEFAULT, does not affect other logic)" );
gamescope::ConVar<bool> cv_drm_debug_disable_output_tf( "drm_debug_disable_output_tf", false, "Force default (identity) output TF, affects other logic. Not a property directly." );
gamescope::ConVar<bool> cv_drm_debug_disable_blend_tf( "drm_debug_disable_blend_tf", false, "Blending chicken bit. (Forces BLEND_TF to DEFAULT, does not affect other logic)" );
gamescope::ConVar<bool> cv_drm_debug_disable_ctm( "drm_debug_disable_ctm", false, "CTM chicken bit. (Forces CTM off, does not affect other logic)" );
gamescope::ConVar<bool> cv_drm_debug_disable_color_encoding( "drm_debug_disable_color_encoding", false, "YUV Color Encoding chicken bit. (Forces COLOR_ENCODING to DEFAULT, does not affect other logic)" );
gamescope::ConVar<bool> cv_drm_debug_disable_color_range( "drm_debug_disable_color_range", false, "YUV Color Range chicken bit. (Forces COLOR_RANGE to DEFAULT, does not affect other logic)" );
gamescope::ConVar<bool> cv_drm_debug_disable_explicit_sync( "drm_debug_disable_explicit_sync", false, "Force disable explicit sync on the DRM backend." );
gamescope::ConVar<bool> cv_drm_debug_disable_in_fence_fd( "drm_debug_disable_in_fence_fd", false, "Force disable IN_FENCE_FD being set to avoid over-synchronization on the DRM backend." );

gamescope::ConVar<bool> cv_drm_allow_dynamic_modes_for_external_display( "drm_allow_dynamic_modes_for_external_display", false, "Allow dynamic mode/refresh rate switching for external displays." );

int HackyDRMPresent( const FrameInfo_t *pFrameInfo, bool bAsync );

enum GamescopeBroadcastRGBMode_t : uint32_t
{
	GAMESCOPE_BROADCAST_RGB_MODE_AUTOMATIC = 0,
	GAMESCOPE_BROADCAST_RGB_MODE_FULL = 1,
	GAMESCOPE_BROADCAST_RGB_MODE_LIMITED = 2,
};

struct saved_mode {
	int width;
	int height;
	int refresh;
	GamescopeBroadcastRGBMode_t broadcast_mode;
};

gamescope::ConVar<bool> cv_drm_ignore_internal_connectors( "drm_ignore_internal_connectors", false, "Disable internal displays for good, for debugging." );

namespace gamescope
{
	class CDRMPlane;
	class CDRMCRTC;
	class CDRMConnector;
}

struct DRMColorCandidate
{
	uint32_t plane = 0;
	uint64_t root = 0;
	drm_color::Program program;
	std::vector<std::shared_ptr<gamescope::BackendBlob>> blobs;
};

struct drm_t {
	bool bUseLiftoff;

	int fd = -1;

	int preferred_width, preferred_height, preferred_refresh;

	uint64_t cursor_width, cursor_height;
	bool allow_modifiers;
	struct wlr_drm_format_set formats;
	bool is_nvidia = false;
	bool is_vc4 = false;
	bool generic_plane_color = false;
	std::map<uint32_t, std::vector<drm_color::Pipeline>> color_pipelines;
	std::map<std::vector<uint64_t>, std::vector<DRMColorCandidate>> color_program_cache;
	std::map<std::vector<uint64_t>, std::optional<drm_color::CrtcProgram>> color_output_cache;
	std::set<std::vector<uint64_t>> color_failed_frames;
	std::vector<uint64_t> color_last_roots;
	std::string color_failure;
	uint32_t color_cache_serial = 0;
#if HAVE_GBM
	struct gbm_device *gbm = nullptr;
#endif

	std::vector< std::unique_ptr< gamescope::CDRMPlane > > planes;
	std::vector< std::unique_ptr< gamescope::CDRMCRTC > > crtcs;
	std::unordered_map< uint32_t, gamescope::CDRMConnector > connectors;

	gamescope::CDRMPlane *pPrimaryPlane;
	gamescope::CDRMCRTC *pCRTC;
	gamescope::CDRMConnector *pConnector;

	struct wlr_drm_format_set primary_formats;

	drmModeAtomicReq *req;
	uint32_t flags;

	struct liftoff_device *lo_device;
	struct liftoff_output *lo_output;
	struct liftoff_layer *lo_layers[ k_nMaxLayers ];

	std::shared_ptr<gamescope::BackendBlob> sdr_static_metadata;

	struct drm_state_t {
		std::shared_ptr<gamescope::BackendBlob> mode_id;
		uint32_t color_mgmt_serial;
		std::shared_ptr<gamescope::BackendBlob> lut3d_id[ EOTF_Count ];
		std::shared_ptr<gamescope::BackendBlob> shaperlut_id[ EOTF_Count ];
		amdgpu_transfer_function output_tf = AMDGPU_TRANSFER_FUNCTION_DEFAULT;
		std::vector<std::shared_ptr<gamescope::BackendBlob>> generic_color_blobs;
	} current, pending;

	// FBs in the atomic request, but not yet submitted to KMS
	// Accessed only on req thread
	std::vector<gamescope::Rc<gamescope::IBackendFb>> m_FbIdsInRequest;
	std::vector<int> m_InFenceFdsInRequest;

	// FBs currently queued to go on screen.
	// May be accessed by page flip handler thread and req thread, thus mutex.
	std::mutex m_QueuedFbIdsMutex;
	std::vector<gamescope::Rc<gamescope::IBackendFb>> m_QueuedFbIds;
	// FBs currently on screen.
	// Accessed only on page flip handler thread.
	std::mutex m_mutVisibleFbIds;
	std::vector<gamescope::Rc<gamescope::IBackendFb>> m_VisibleFbIds;

	std::atomic < uint32_t > uPendingFlipCount = { 0 };

	std::atomic < bool > paused = { false };
	std::atomic < int > out_of_date = { false };
	std::atomic < bool > needs_modeset = { false };

	std::unordered_map< std::string, int > connector_priorities;

	char *device_name = nullptr;
};

void drm_drop_fbid( struct drm_t *drm, uint32_t fbid );
bool drm_set_mode( struct drm_t *drm, const drmModeModeInfo *mode );


using namespace std::literals;

struct drm_t g_DRM = {};

static void drm_close_in_fences( struct drm_t *drm )
{
	for ( int nFd : drm->m_InFenceFdsInRequest )
		close( nFd );
	drm->m_InFenceFdsInRequest.clear();
}
// Flip handler thread control. Keep the thread object global so we
// can join it during shutdown instead of detaching and risking the
// thread still using the DRM fd while we clean up.
static std::thread g_page_flip_handler_thread;
static std::atomic<bool> g_page_flip_handler_thread_should_exit{false};

static int g_page_flip_pipe_fds[2] = { -1, -1 };

namespace gamescope
{
	class CDRMBackend;

	std::tuple<int32_t, int32_t, int32_t> GetKernelVersion()
	{
		utsname name;
		if ( uname( &name ) != 0 )
			return std::make_tuple( 0, 0, 0 );

		std::vector<std::string_view> szVersionParts = Split( name.release, "." );

		uint32_t uVersion[3] = { 0 };
		for ( size_t i = 0; i < szVersionParts.size() && i < 3; i++ )
		{
			auto oPart = Parse<int32_t>( szVersionParts[i] );
			if ( !oPart )
				break;

			uVersion[i] = *oPart;
		}

		return std::make_tuple( uVersion[0], uVersion[1], uVersion[2] );
	}

	// Get a DRM mode in mHz
	// Taken from wlroots, but we can't access it as we don't
	// use the drm backend.
	static int32_t GetModeRefresh(const drmModeModeInfo *mode)
	{
		int32_t nRefresh = (mode->clock * 1'000'000ll / mode->htotal + mode->vtotal / 2) / mode->vtotal;

		if (mode->flags & DRM_MODE_FLAG_INTERLACE)
			nRefresh *= 2;

		if (mode->flags & DRM_MODE_FLAG_DBLSCAN)
			nRefresh /= 2;

		if (mode->vscan > 1)
			nRefresh /= mode->vscan;

		return nRefresh;
	}

	static int32_t GetVblankNs(const drmModeModeInfo *mode)
	{
		return (mode->vsync_start - mode->vdisplay) * 1'000'000'000ll / mode->vrefresh / mode->vtotal;
	}

	template <typename T>
	using CAutoDeletePtr = std::unique_ptr<T, void(*)(T*)>;

	////////////////////////////////////////
	// DRM Object Wrappers + State Trackers
	////////////////////////////////////////
	struct DRMObjectRawProperty
	{
		uint32_t uPropertyId = 0ul;
		uint64_t ulValue = 0ul;
	};
	using DRMObjectRawProperties = std::unordered_map<std::string, DRMObjectRawProperty>;

	class CDRMAtomicObject
	{
	public:
		CDRMAtomicObject( uint32_t ulObjectId );
		uint32_t GetObjectId() const { return m_ulObjectId; }

		// No copy or move constructors.
		CDRMAtomicObject( const CDRMAtomicObject& ) = delete;
		CDRMAtomicObject& operator=( const CDRMAtomicObject& ) = delete;

		CDRMAtomicObject( CDRMAtomicObject&& ) = delete;
		CDRMAtomicObject& operator=( CDRMAtomicObject&& ) = delete;
	protected:
		uint32_t m_ulObjectId = 0ul;
	};

	template < uint32_t DRMObjectType >
	class CDRMAtomicTypedObject : public CDRMAtomicObject
	{
	public:
		CDRMAtomicTypedObject( uint32_t ulObjectId );
	protected:
		std::optional<DRMObjectRawProperties> GetRawProperties();
	};

	class CDRMAtomicProperty
	{
	public:
		CDRMAtomicProperty( CDRMAtomicObject *pObject, DRMObjectRawProperty rawProperty );

		static std::optional<CDRMAtomicProperty> Instantiate( const char *pszName, CDRMAtomicObject *pObject, const DRMObjectRawProperties& rawProperties );

		uint32_t GetPropertyId() const { return m_uPropertyId; }
		uint64_t GetPendingValue() const { return m_ulPendingValue; }
		uint64_t GetCurrentValue() const { return m_ulCurrentValue; }
		uint64_t GetInitialValue() const { return m_ulInitialValue; }
		int SetPendingValue( drmModeAtomicReq *pRequest, uint64_t ulValue, bool bForce );

		void OnCommit();
		void Rollback();
	private:
		CDRMAtomicObject *m_pObject = nullptr;
		uint32_t m_uPropertyId = 0u;

		uint64_t m_ulPendingValue = 0ul;
		uint64_t m_ulCurrentValue = 0ul;
		uint64_t m_ulInitialValue = 0ul;
	};

	class CDRMPlane final : public CDRMAtomicTypedObject<DRM_MODE_OBJECT_PLANE>
	{
	public:
		// Takes ownership of pPlane.
		CDRMPlane( drmModePlane *pPlane );

		void RefreshState();

		drmModePlane *GetModePlane() const { return m_pPlane.get(); }

		struct PlaneProperties
		{
			std::optional<CDRMAtomicProperty> *begin() { return &FB_ID; }
			std::optional<CDRMAtomicProperty> *end() { return &DUMMY_END; }

			std::optional<CDRMAtomicProperty> type; // Immutable
			std::optional<CDRMAtomicProperty> IN_FORMATS; // Immutable

			std::optional<CDRMAtomicProperty> FB_ID;
			std::optional<CDRMAtomicProperty> IN_FENCE_FD;
			std::optional<CDRMAtomicProperty> CRTC_ID;
			std::optional<CDRMAtomicProperty> SRC_X;
			std::optional<CDRMAtomicProperty> SRC_Y;
			std::optional<CDRMAtomicProperty> SRC_W;
			std::optional<CDRMAtomicProperty> SRC_H;
			std::optional<CDRMAtomicProperty> CRTC_X;
			std::optional<CDRMAtomicProperty> CRTC_Y;
			std::optional<CDRMAtomicProperty> CRTC_W;
			std::optional<CDRMAtomicProperty> CRTC_H;
			std::optional<CDRMAtomicProperty> zpos;
			std::optional<CDRMAtomicProperty> alpha;
			std::optional<CDRMAtomicProperty> rotation;
			std::optional<CDRMAtomicProperty> COLOR_ENCODING;
			std::optional<CDRMAtomicProperty> COLOR_RANGE;
			std::optional<CDRMAtomicProperty> COLOR_PIPELINE;
			std::optional<CDRMAtomicProperty> AMD_PLANE_DEGAMMA_TF;
			std::optional<CDRMAtomicProperty> AMD_PLANE_DEGAMMA_LUT;
			std::optional<CDRMAtomicProperty> AMD_PLANE_CTM;
			std::optional<CDRMAtomicProperty> AMD_PLANE_HDR_MULT;
			std::optional<CDRMAtomicProperty> AMD_PLANE_SHAPER_LUT;
			std::optional<CDRMAtomicProperty> AMD_PLANE_SHAPER_TF;
			std::optional<CDRMAtomicProperty> AMD_PLANE_LUT3D;
			std::optional<CDRMAtomicProperty> AMD_PLANE_BLEND_TF;
			std::optional<CDRMAtomicProperty> AMD_PLANE_BLEND_LUT;
			std::optional<CDRMAtomicProperty> DUMMY_END;
		};
		      PlaneProperties &GetProperties()       { return m_Props; }
		const PlaneProperties &GetProperties() const { return m_Props; }
	private:
		CAutoDeletePtr<drmModePlane> m_pPlane;
		PlaneProperties m_Props;
	};

	class CDRMCRTC final : public CDRMAtomicTypedObject<DRM_MODE_OBJECT_CRTC>
	{
	public:
		// Takes ownership of pCRTC.
		CDRMCRTC( drmModeCrtc *pCRTC, uint32_t uCRTCMask );

		void RefreshState();
		uint32_t GetCRTCMask() const { return m_uCRTCMask; }
		uint32_t GetGammaSize() const { return m_uGammaSize; }
		uint32_t GetDegammaSize() const { return m_uDegammaSize; }

		struct CRTCProperties
		{
			std::optional<CDRMAtomicProperty> *begin() { return &ACTIVE; }
			std::optional<CDRMAtomicProperty> *end() { return &DUMMY_END; }

			std::optional<CDRMAtomicProperty> ACTIVE;
			std::optional<CDRMAtomicProperty> MODE_ID;
			std::optional<CDRMAtomicProperty> GAMMA_LUT;
			std::optional<CDRMAtomicProperty> DEGAMMA_LUT;
			std::optional<CDRMAtomicProperty> CTM;
			std::optional<CDRMAtomicProperty> VRR_ENABLED;
			std::optional<CDRMAtomicProperty> OUT_FENCE_PTR;
			std::optional<CDRMAtomicProperty> AMD_CRTC_REGAMMA_TF;
			std::optional<CDRMAtomicProperty> DUMMY_END;
		};
		      CRTCProperties &GetProperties()       { return m_Props; }
		const CRTCProperties &GetProperties() const { return m_Props; }
	private:
		CAutoDeletePtr<drmModeCrtc> m_pCRTC;
		uint32_t m_uCRTCMask = 0u;
		uint32_t m_uGammaSize = 0;
		uint32_t m_uDegammaSize = 0;
		CRTCProperties m_Props;
	};

	class CDRMConnector final : public CBaseBackendConnector, public CDRMAtomicTypedObject<DRM_MODE_OBJECT_CONNECTOR>
	{
	public:
		CDRMConnector( CDRMBackend *pBackend, drmModeConnector *pConnector );

		void RefreshState();

		struct ConnectorProperties
		{
			std::optional<CDRMAtomicProperty> *begin() { return &CRTC_ID; }
			std::optional<CDRMAtomicProperty> *end() { return &DUMMY_END; }

			std::optional<CDRMAtomicProperty> CRTC_ID;
			std::optional<CDRMAtomicProperty> Colorspace;
			std::optional<CDRMAtomicProperty> content_type; // "content type" with space!
			std::optional<CDRMAtomicProperty> panel_orientation; // "panel orientation" with space!
			std::optional<CDRMAtomicProperty> HDR_OUTPUT_METADATA;
			std::optional<CDRMAtomicProperty> vrr_capable;
			std::optional<CDRMAtomicProperty> EDID;
			std::optional<CDRMAtomicProperty> Broadcast_RGB;
			std::optional<CDRMAtomicProperty> Max_bpc;
			std::optional<CDRMAtomicProperty> DUMMY_END;
		};
		      ConnectorProperties &GetProperties()       { return m_Props; }
		const ConnectorProperties &GetProperties() const { return m_Props; }

		drmModeConnector *GetModeConnector() { return m_pConnector.get(); }
		const char *GetName() const override { return m_Mutable.szName; }
		const char *GetMake() const override { return m_Mutable.pszMake; }
		const char *GetModel() const override { return m_Mutable.szModel; }
		const char *GetDataString() const { return m_Mutable.szDataString; }
		uint32_t GetPossibleCRTCMask() const { return m_Mutable.uPossibleCRTCMask; }
		std::span<const uint32_t> GetValidDynamicRefreshRates() const override { return m_Mutable.ValidDynamicRefreshRates; }
		const displaycolorimetry_t& GetDisplayColorimetry() const { return m_Mutable.DisplayColorimetry; }

		std::span<const uint8_t> GetRawEDID() const override { return std::span<const uint8_t>{ m_Mutable.EdidData.begin(), m_Mutable.EdidData.end() }; }
		bool HandleEdidChange()
		{
			bool bChanged = m_Mutable.bEdidChanged;
			m_Mutable.bEdidChanged = false;
			return bChanged;
		}

		bool SupportsHDR10() const
		{
			return !!GetProperties().Colorspace && !!GetProperties().HDR_OUTPUT_METADATA && GetHDRInfo().IsHDR10();
		}

		bool SupportsHDRG22() const
		{
			return GetHDRInfo().IsHDRG22();
		}

		//////////////////////////////////////
		// IBackendConnector implementation
		//////////////////////////////////////

		GamescopeScreenType GetScreenType() const override
		{
			if ( m_pConnector->connector_type == DRM_MODE_CONNECTOR_eDP ||
				 m_pConnector->connector_type == DRM_MODE_CONNECTOR_LVDS ||
				 m_pConnector->connector_type == DRM_MODE_CONNECTOR_DSI )
				return GAMESCOPE_SCREEN_TYPE_INTERNAL;

			return GAMESCOPE_SCREEN_TYPE_EXTERNAL;
		}

		GamescopePanelOrientation GetCurrentOrientation() const override
		{
			return m_ChosenOrientation;
		}

		bool SupportsHDR() const override
		{
			return SupportsHDR10() || SupportsHDRG22();
		}

		bool IsHDRActive() const override
		{
			if ( SupportsHDR10() )
			{
				return GetProperties().Colorspace->GetCurrentValue() == DRM_MODE_COLORIMETRY_BT2020_RGB;
			}
			else if ( SupportsHDRG22() )
			{
				return true;
			}

			return false;
		}

		const BackendConnectorHDRInfo &GetHDRInfo() const override { return m_Mutable.HDR; }

		virtual bool IsVRRActive() const override
		{
			if ( !g_DRM.pCRTC || !g_DRM.pCRTC->GetProperties().VRR_ENABLED )
				return false;

			return !!g_DRM.pCRTC->GetProperties().VRR_ENABLED->GetCurrentValue();
		}

		virtual std::span<const BackendMode> GetModes() const override { return m_Mutable.BackendModes; }

		bool SupportsVRR() const override
		{
			return this->GetProperties().vrr_capable && !!this->GetProperties().vrr_capable->GetCurrentValue();
		}

        void GetNativeColorimetry(
			bool bHDR,
            displaycolorimetry_t *displayColorimetry, EOTF *displayEOTF,
            displaycolorimetry_t *outputEncodingColorimetry, EOTF *outputEncodingEOTF ) const override
		{
			*displayColorimetry = GetDisplayColorimetry();
			*displayEOTF = EOTF_Gamma22;

			if ( bHDR && GetHDRInfo().IsHDR10() )
			{
				// For HDR10 output, expected content colorspace != native colorspace.
				*outputEncodingColorimetry = displaycolorimetry_2020;
				*outputEncodingEOTF = GetHDRInfo().eOutputEncodingEOTF;
			}
			else
			{
				*outputEncodingColorimetry = GetDisplayColorimetry();
				*outputEncodingEOTF = EOTF_Gamma22;
			}
		}

		virtual int Present( const FrameInfo_t *pFrameInfo, bool bAsync ) override;

		using DRMModeGenerator = std::function<drmModeModeInfo(const drmModeModeInfo *, int)>;
		const DRMModeGenerator &GetModeGenerator() const
		{
			return m_Mutable.fnDynamicModeGenerator;
		}
		void UpdateEffectiveOrientation( const drmModeModeInfo *pMode );

	private:
		void ParseEDID();


		CDRMBackend *m_pBackend = nullptr;
		CAutoDeletePtr<drmModeConnector> m_pConnector;

		struct MutableConnectorState
		{
			int nDefaultRefresh = 0;

			uint32_t uPossibleCRTCMask = 0u;
			char szName[32]{};
			char szMakePNP[4]{};
			char szModel[16]{};
			char szDataString[16]{};
			const char *pszMake = ""; // Not owned, no free. This is a pointer to pnp db or szMakePNP.
			DRMModeGenerator fnDynamicModeGenerator;
			std::vector<uint32_t> ValidDynamicRefreshRates{};
			std::vector<uint8_t> EdidData; // Raw, unmodified.
			std::vector<BackendMode> BackendModes;
			
			displaycolorimetry_t DisplayColorimetry = displaycolorimetry_709;
			BackendConnectorHDRInfo HDR;

			bool bEdidChanged = false;
		} m_Mutable;

		GamescopePanelOrientation m_ChosenOrientation = GAMESCOPE_PANEL_ORIENTATION_AUTO;

		ConnectorProperties m_Props;
	};

	class CDRMFb final : public CBaseBackendFb
	{
	public:
		CDRMFb( uint32_t uFbId );
		~CDRMFb();

		uint32_t GetFbId() const { return m_uFbId; }
	
	private:
		uint32_t m_uFbId = 0;
	};
}

uint32_t g_nDRMFormat = DRM_FORMAT_INVALID;
uint32_t g_nDRMFormatOverlay = DRM_FORMAT_INVALID; // for partial composition, we may have more limited formats than base planes + alpha.
bool g_bRotated = false;
extern bool g_bDebugLayers;

struct DRMPresentCtx
{
	uint64_t ulPendingFlipCount = 0;
	uint64_t frameGenerationOutputId = 0;
};

// Fence completion feeds the actual rendering latency back into frame scheduling.
// Reporting only CPU submission time after removing vulkan_wait would wake too late.
class CCompositeCompletion final : public gamescope::IWaitable, public gamescope::RcObject
{
public:
	CCompositeCompletion(int fd, uint64_t start) : m_fd(fd), m_start(start) {}
	~CCompositeCompletion() { close(m_fd); }
	int GetFD() override { return m_fd; }
	void OnPollIn() override;
private:
	int m_fd;
	uint64_t m_start;
};

static gamescope::CAsyncWaiter<gamescope::Rc<CCompositeCompletion>> &composite_waiter()
{
	static gamescope::CAsyncWaiter<gamescope::Rc<CCompositeCompletion>> waiter("gamescope_comp");
	return waiter;
}

void CCompositeCompletion::OnPollIn()
{
	GetVBlankTimer().UpdateLastDrawTime(get_time_in_nanos() - m_start);
	composite_waiter().RemoveWaitable(this);
	nudge_steamcompmgr();
}

static bool watch_composite(int fd, uint64_t start)
{
	if (fd < 0)
	{
		GetVBlankTimer().UpdateLastDrawTime(get_time_in_nanos() - start);
		return true;
	}
	int copy = fcntl(fd, F_DUPFD_CLOEXEC, 0);
	if (copy < 0)
		return false;
	gamescope::OwningRc<CCompositeCompletion> completion = new CCompositeCompletion(copy, start);
	return composite_waiter().AddWaitable(completion.get());
}

static void wait_sync_file(int fd)
{
	pollfd p = { .fd = fd, .events = POLLIN };
	while (poll(&p, 1, -1) < 0 && errno == EINTR) {}
}

extern gamescope::ConVar<bool> cv_composite_force;
extern bool g_bColorSliderInUse;
extern bool fadingOut;
extern std::string g_reshade_effect;

#ifndef DRM_CAP_ATOMIC_ASYNC_PAGE_FLIP
#define DRM_CAP_ATOMIC_ASYNC_PAGE_FLIP 0x15
#endif

bool drm_update_color_mgmt(struct drm_t *drm);
bool drm_supports_color_mgmt(struct drm_t *drm);
bool drm_supports_amd_color_mgmt(struct drm_t *drm);
static void drm_init_generic_color(struct drm_t *drm);
static int drm_prepare_generic_color(struct drm_t *drm, const FrameInfo_t *frameInfo,
	const std::function<int()> &apply);
bool drm_supports_srgb_to_pq(struct drm_t *drm);
bool drm_set_connector( struct drm_t *drm, gamescope::CDRMConnector *conn );

struct drm_color_ctm2 {
	/*
	 * Conversion matrix in S31.32 sign-magnitude
	 * (not two's complement!) format.
	 */
	__u64 matrix[12];
};

bool g_bSupportsAsyncFlips = false;
bool g_bSupportsSyncObjs = false;

extern gamescope::GamescopeModeGeneration g_eGamescopeModeGeneration;
extern GamescopePanelOrientation g_DesiredInternalOrientation;

extern bool g_bForceDisableColorMgmt;

static LogScope drm_log( "drm" );
static LogScope liftoff_log_scope( "liftoff" );

static std::unordered_map< std::string, std::string > pnps = {};

static void drm_unset_mode( struct drm_t *drm, bool force );
static void drm_unset_connector( struct drm_t *drm );
static void drm_update_screen_disabled( struct drm_t *drm );

static constexpr uint32_t s_kSteamDeckLCDRates[] =
{
	40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
	50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
	60,
};

static constexpr uint32_t s_kSteamDeckOLEDRates[] =
{
	45, 47, 48, 49, 
	50, 51, 53, 55, 56, 59, 
	60, 62, 64, 65, 66, 68, 
	72, 73, 76, 77, 78, 
	80, 81, 82, 84, 85, 86, 87, 88, 
	90, 
};

void update_connector_display_info_wl(struct drm_t *drm)
{
	wlserver_lock();
	for ( const auto &control : wlserver.gamescope_controls )
	{
		wlserver_send_gamescope_control( control );
	}
	wlserver_unlock();
}

inline uint64_t drm_calc_s31_32(float val)
{
	// S31.32 sign-magnitude
	float integral = 0.0f;
	float fractional = modf( fabsf( val ), &integral );

	union
	{
		struct
		{
			uint64_t fractional : 32;
			uint64_t integral   : 31;
			uint64_t sign_part  : 1;
		} s31_32_bits;
		uint64_t s31_32;
	} color;

	color.s31_32_bits.sign_part  = val < 0 ? 1 : 0;
	color.s31_32_bits.integral   = uint64_t( integral );
	color.s31_32_bits.fractional = uint64_t( fractional * float( 1ull << 32 ) );

	return color.s31_32;
}

static gamescope::CDRMCRTC *find_crtc_for_connector( struct drm_t *drm, gamescope::CDRMConnector *pConnector )
{
	for ( std::unique_ptr< gamescope::CDRMCRTC > &pCRTC : drm->crtcs )
	{
		if ( pConnector->GetPossibleCRTCMask() & pCRTC->GetCRTCMask() )
			return pCRTC.get();
	}

	return nullptr;
}

static bool get_plane_formats( struct drm_t *drm, gamescope::CDRMPlane *pPlane, struct wlr_drm_format_set *pFormatSet )
{
	for ( uint32_t i = 0; i < pPlane->GetModePlane()->count_formats; i++ )
	{
		const uint32_t uFormat = pPlane->GetModePlane()->formats[ i ];
		wlr_drm_format_set_add( pFormatSet, uFormat, DRM_FORMAT_MOD_INVALID );
	}

	if ( pPlane->GetProperties().IN_FORMATS )
	{
		const uint64_t ulBlobId = pPlane->GetProperties().IN_FORMATS->GetCurrentValue();

		drmModePropertyBlobRes *pBlob = drmModeGetPropertyBlob( drm->fd, ulBlobId );
		if ( !pBlob )
		{
			drm_log.errorf_errno("drmModeGetPropertyBlob(IN_FORMATS) failed");
			return false;
		}
		defer( drmModeFreePropertyBlob( pBlob ) );

		drm_format_modifier_blob *pModifierBlob = reinterpret_cast<drm_format_modifier_blob *>( pBlob->data );

		uint32_t *pFormats = reinterpret_cast<uint32_t *>( reinterpret_cast<uint8_t *>( pBlob->data ) + pModifierBlob->formats_offset );
		drm_format_modifier *pMods = reinterpret_cast<drm_format_modifier *>( reinterpret_cast<uint8_t *>( pBlob->data ) + pModifierBlob->modifiers_offset );

		for ( uint32_t i = 0; i < pModifierBlob->count_modifiers; i++ )
		{
			for ( uint32_t j = 0; j < 64; j++ )
			{
				if ( pMods[i].formats & ( uint64_t(1) << j ) )
					wlr_drm_format_set_add( pFormatSet, pFormats[j + pMods[i].offset], pMods[i].modifier );
			}
		}
	}

	return true;
}

static uint32_t pick_plane_format( const struct wlr_drm_format_set *formats, uint32_t Xformat, uint32_t Aformat )
{
	const VkFormatFeatureFlags neededFeatures = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT;
	uint32_t result = DRM_FORMAT_INVALID;
	for ( size_t i = 0; i < formats->len; i++ ) {
		uint32_t fmt = formats->formats[i].format;

		// Skip formats that we cannot use with the Vulkan device
		if ( !vulkan_format_supports_features( DRMFormatToVulkan(fmt, false), neededFeatures ) )
			continue;

		if ( fmt == Xformat ) {
			// Prefer formats without alpha channel for main plane
			result = fmt;
		} else if ( result == DRM_FORMAT_INVALID && fmt == Aformat ) {
			result = fmt;
		}
	}
	return result;
}

/* Pick a primary plane that can be connected to the chosen CRTC. */
static gamescope::CDRMPlane *find_primary_plane(struct drm_t *drm)
{
	for ( std::unique_ptr< gamescope::CDRMPlane > &pPlane : drm->planes )
	{
		// Use a plane that can be used with the current CRTC. If no CRTC is
		// chosen yet, pick any primary plane.
		if ( !drm->pCRTC || (pPlane->GetModePlane()->possible_crtcs & drm->pCRTC->GetCRTCMask()) )
		{
			if ( pPlane->GetProperties().type->GetCurrentValue() == DRM_PLANE_TYPE_PRIMARY )
				return pPlane.get();
		}
	}

	return nullptr;
}

/* Pick any primary plane, ignoring CRTC routing, to bootstrap a headless start. */
static gamescope::CDRMPlane *find_any_primary_plane(struct drm_t *drm)
{
	for ( std::unique_ptr< gamescope::CDRMPlane > &pPlane : drm->planes )
	{
		if ( pPlane->GetProperties().type->GetCurrentValue() == DRM_PLANE_TYPE_PRIMARY )
			return pPlane.get();
	}

	return nullptr;
}

static bool have_overlay_planes(struct drm_t *drm)
{
	if ( !drm->pCRTC )
		return false;

	for ( std::unique_ptr< gamescope::CDRMPlane > &pPlane : drm->planes )
	{
		if ( pPlane->GetModePlane()->possible_crtcs & drm->pCRTC->GetCRTCMask() )
		{
			if ( pPlane->GetProperties().type->GetCurrentValue() == DRM_PLANE_TYPE_OVERLAY )
				return true;
		}
	}

	return false;
}

extern void mangoapp_output_update( uint64_t vblanktime, uint64_t frameGenerationOutputId );
extern std::atomic<uint32_t> g_uScanoutRefreshmHz;
static void page_flip_handler(int fd, unsigned int frame, unsigned int sec, unsigned int usec, unsigned int crtc_id, void *data)
{
	DRMPresentCtx *pCtx = reinterpret_cast<DRMPresentCtx *>( data );

	// Make this const when we move into CDRMBackend.
	if ( g_DRM.pConnector )
		g_DRM.pConnector->PresentationFeedback().m_uCompletedPresents = pCtx->ulPendingFlipCount;

	if ( !g_DRM.pCRTC )
		return;

	if ( g_DRM.pCRTC->GetObjectId() != crtc_id )
		return;

	static uint64_t ulLastVBlankTime = 0;

	// This is the last vblank time
	uint64_t vblanktime = sec * 1'000'000'000lu + usec * 1'000lu;
	GetVBlankTimer().MarkVBlank( vblanktime, true );

	// TODO: get the fbids_queued instance from data if we ever have more than one in flight

	drm_log.debugf("page_flip_handler %" PRIu64 " delta: %" PRIu64, pCtx->ulPendingFlipCount, vblanktime - ulLastVBlankTime );
	gpuvis_trace_printf("page_flip_handler %" PRIu64, pCtx->ulPendingFlipCount);

	ulLastVBlankTime = vblanktime;

	// frame is the vblank count, so its delta includes refreshes VRR scanned out without a flip.
	static unsigned int s_uRateFrame = 0;
	static uint64_t s_ulRateTime = 0;
	static unsigned int s_uRateCRTC = 0;
	if ( crtc_id != s_uRateCRTC || vblanktime < s_ulRateTime )
		s_ulRateTime = 0;
	if ( !s_ulRateTime || vblanktime - s_ulRateTime >= 500'000'000ul )
	{
		g_uScanoutRefreshmHz = s_ulRateTime
			? uint32_t( uint64_t( frame - s_uRateFrame ) * 1'000'000'000'000ul / ( vblanktime - s_ulRateTime ) )
			: 0;
		s_uRateFrame = frame;
		s_ulRateTime = vblanktime;
		s_uRateCRTC = crtc_id;
	}

	{
		std::scoped_lock lock{ g_DRM.m_QueuedFbIdsMutex, g_DRM.m_mutVisibleFbIds };
		// Swap and clear from queue -> visible to avoid allocations.
		g_DRM.m_VisibleFbIds.swap( g_DRM.m_QueuedFbIds );
		g_DRM.m_QueuedFbIds.clear();
	}

	g_DRM.uPendingFlipCount--;
	g_DRM.uPendingFlipCount.notify_all();

	mangoapp_output_update( vblanktime, pCtx->frameGenerationOutputId );

	// Nudge so that steamcompmgr releases commits.
	nudge_steamcompmgr();
}

void flip_handler_thread_run(void)
{
	pthread_setname_np( pthread_self(), "gamescope-kms" );

	// Prepare pollfds: one for DRM fd and one for the pipe read end to
	// detect when the write end is closed (POLLHUP) to exit.
	struct pollfd fds[2];
	int nfds = 0;

	fds[nfds].fd = g_DRM.fd;
	fds[nfds].events = POLLIN;
	nfds++;

	fds[nfds].fd = g_page_flip_pipe_fds[0];
	fds[nfds].events = POLLIN;
	nfds++;

	while ( !g_page_flip_handler_thread_should_exit.load( std::memory_order_acquire ) )
	{
		int ret = poll( fds, nfds, -1 );
		if ( ret < 0 ) {
			if ( errno == EINTR )
				continue;
			drm_log.errorf_errno( "polling for DRM events failed" );
			break;
		}

		// Check if the pipe read end got POLLHUP (write end closed) to exit.
		if ( fds[1].revents & (POLLIN | POLLHUP) ) {
			break;
		}

		if ( (fds[0].revents & POLLIN) ) {
			drmEventContext evctx = {
				.version = 3,
				.page_flip_handler2 = page_flip_handler,
			};
			drmHandleEvent( g_DRM.fd, &evctx );
		}
	}

	drm_log.debugf("page_flip_handler_thread exiting");
}

static bool refresh_state( drm_t *drm )
{
	drmModeRes *pResources = drmModeGetResources( drm->fd );
	if ( pResources == nullptr )
	{
		drm_log.errorf_errno( "drmModeGetResources failed" );
		return false;
	}
	defer( drmModeFreeResources( pResources ) );

	// Add connectors which appeared
	for ( int i = 0; i < pResources->count_connectors; i++ )
	{
		uint32_t uConnectorId = pResources->connectors[i];

		drmModeConnector *pConnector = drmModeGetConnector( drm->fd, uConnectorId );
		if ( !pConnector )
			continue;

		if ( cv_drm_ignore_internal_connectors )
		{
			if ( pConnector->connector_type == DRM_MODE_CONNECTOR_eDP ||
				pConnector->connector_type == DRM_MODE_CONNECTOR_LVDS ||
				pConnector->connector_type == DRM_MODE_CONNECTOR_DSI )
			{
				drmModeFreeConnector( pConnector );
				continue;
			}
		}

		if ( !drm->connectors.contains( uConnectorId ) )
		{
			drm->connectors.emplace(
				std::piecewise_construct,
				std::forward_as_tuple( uConnectorId ),
				std::forward_as_tuple( reinterpret_cast<gamescope::CDRMBackend *>( GetBackend() ), pConnector ) );
		}
	}

	// Remove connectors which disappeared
	for ( auto iter = drm->connectors.begin(); iter != drm->connectors.end(); )
	{
		gamescope::CDRMConnector *pConnector = &iter->second;

		const bool bFound = std::any_of(
			pResources->connectors,
			pResources->connectors + pResources->count_connectors,
			std::bind_front( std::equal_to{}, pConnector->GetObjectId() ) );

		if ( !bFound )
		{
			drm_log.debugf( "Connector '%s' disappeared.", pConnector->GetName() );

			if ( drm->pConnector == pConnector )
			{
				drm_log.infof( "Current connector '%s' disappeared.", pConnector->GetName() );
				drm->pConnector = nullptr;
			}

			iter = drm->connectors.erase( iter );
		}
		else
			iter++;
	}

	// Re-probe connectors props and status)
	for ( auto &iter : drm->connectors )
	{
		gamescope::CDRMConnector *pConnector = &iter.second;
		pConnector->RefreshState();
	}

	for ( std::unique_ptr< gamescope::CDRMCRTC > &pCRTC : drm->crtcs )
		pCRTC->RefreshState();

	for ( std::unique_ptr< gamescope::CDRMPlane > &pPlane : drm->planes )
		pPlane->RefreshState();

	drm_update_screen_disabled( drm );

	return true;
}

static bool get_resources(struct drm_t *drm)
{
	{
		drmModeRes *pResources = drmModeGetResources( drm->fd );
		if ( !pResources )
		{
			drm_log.errorf_errno( "drmModeGetResources failed" );
			return false;
		}
		defer( drmModeFreeResources( pResources ) );

		for ( int i = 0; i < pResources->count_crtcs; i++ )
		{
			drmModeCrtc *pCRTC = drmModeGetCrtc( drm->fd, pResources->crtcs[ i ] );
			if ( pCRTC )
				drm->crtcs.emplace_back( std::make_unique<gamescope::CDRMCRTC>( pCRTC, 1u << i ) );
		}
	}

	{
		drmModePlaneRes *pPlaneResources = drmModeGetPlaneResources( drm->fd );
		if ( !pPlaneResources )
		{
			drm_log.errorf_errno( "drmModeGetPlaneResources failed" );
			return false;
		}
		defer( drmModeFreePlaneResources( pPlaneResources ) );

		for ( uint32_t i = 0; i < pPlaneResources->count_planes; i++ )
		{
			drmModePlane *pPlane = drmModeGetPlane( drm->fd, pPlaneResources->planes[ i ] );
			if ( pPlane )
				drm->planes.emplace_back( std::make_unique<gamescope::CDRMPlane>( pPlane ) );
		}
	}

	return refresh_state( drm );
}

struct mode_blocklist_entry
{
	uint32_t width, height, refresh;
};

// Filter out reporting some modes that are required for
// certain certifications, but are completely useless,
// and probably don't fit the display pixel size.
static mode_blocklist_entry g_badModes[] =
{
	{ 4096, 2160, 0 },
};

static const drmModeModeInfo *find_mode( const drmModeConnector *connector, int hdisplay, int vdisplay, uint32_t vrefresh )
{
	for (int i = 0; i < connector->count_modes; i++) {
		const drmModeModeInfo *mode = &connector->modes[i];

		bool bad = false;
		for (const auto& badMode : g_badModes) {
			bad |= (badMode.width   == 0 || mode->hdisplay == badMode.width)
				&& (badMode.height  == 0 || mode->vdisplay == badMode.height)
				&& (badMode.refresh == 0 || mode->vrefresh == badMode.refresh);
		}

		if (bad)
			continue;

		if (hdisplay != 0 && hdisplay != mode->hdisplay)
			continue;
		if (vdisplay != 0 && vdisplay != mode->vdisplay)
			continue;
		if (vrefresh != 0 && vrefresh != mode->vrefresh)
			continue;

		return mode;
	}

	return NULL;
}

static std::unordered_map<std::string, int> parse_connector_priorities(const char *str)
{
	std::unordered_map<std::string, int> priorities{};
	if (!str) {
		return priorities;
	}
	int i = 0;
	char *buf = strdup(str);
	char *name = strtok(buf, ",");
	while (name) {
		priorities[name] = i;
		i++;
		name = strtok(nullptr, ",");
	}
	free(buf);
	return priorities;
}

static int get_connector_priority(const std::unordered_map<std::string, int> &priorities, const char *name)
{
	if ( auto iter = priorities.find( name ); iter != priorities.end() ) {
		return iter->second;
	}
	if ( auto iter = priorities.find( "*" ); iter != priorities.end() ) {
		return iter->second;
	}
	return priorities.size();
}

static int get_best_connector_priority_for_fd( int fd, const void *userdata )
{
	const auto &priorities = *static_cast<const std::unordered_map<std::string, int> *>( userdata );
	drmModeRes *resources = drmModeGetResources( fd );
	if ( !resources )
		return INT_MAX;

	int best_priority = INT_MAX;
	for ( int i = 0; i < resources->count_connectors; i++ )
	{
		drmModeConnector *connector = drmModeGetConnector( fd, resources->connectors[i] );
		if ( !connector )
			continue;

		if ( connector->connection == DRM_MODE_CONNECTED )
		{
			char name[64];
			snprintf( name, sizeof( name ), "%s-%u", drmModeGetConnectorTypeName( connector->connector_type ), connector->connector_type_id );
			best_priority = std::min( best_priority, get_connector_priority( priorities, name ) );
		}
		drmModeFreeConnector( connector );
	}

	drmModeFreeResources( resources );
	return best_priority;
}

static bool get_saved_mode(const char *description, saved_mode &mode_info)
{
	const char *mode_file = getenv("GAMESCOPE_MODE_SAVE_FILE");
	if (!mode_file || !*mode_file)
		return false;

	FILE *file = fopen(mode_file, "r");
	if (!file)
		return false;

	char line[256];
    while (fgets(line, sizeof(line), file))
	{
		char saved_description[256];
		uint32_t broadcast_mode = 0;
        int ret = sscanf(line, "%255[^:]:%dx%d@%d %u", saved_description, &mode_info.width, &mode_info.height, &mode_info.refresh, &broadcast_mode);

		mode_info.broadcast_mode = (GamescopeBroadcastRGBMode_t) broadcast_mode;
		
		bool valid = ret == 4 || ret == 5;

		if (valid && !strcmp(saved_description, description))
		{
			fclose(file);
			return true;
		}
    }
	fclose(file);
	return false;
}

/* Same identity extraction as CDRMConnector::ParseEDID. */
static void parse_edid_identity(const di_edid *pEdid, char (&szMakePNP)[4], char (&szModel)[16])
{
	memset(szMakePNP, 0, sizeof(szMakePNP));
	memset(szModel, 0, sizeof(szModel));

	const di_edid_vendor_product *pProduct = di_edid_get_vendor_product(pEdid);
	memcpy(szMakePNP, pProduct->manufacturer, 3);

	const di_edid_display_descriptor *const *pDescriptors = di_edid_get_display_descriptors(pEdid);
	for (size_t i = 0; pDescriptors[i] != nullptr; i++)
	{
		if (di_edid_display_descriptor_get_tag(pDescriptors[i]) == DI_EDID_DISPLAY_DESCRIPTOR_PRODUCT_NAME)
			strncpy(szModel, di_edid_display_descriptor_get_string(pDescriptors[i]), sizeof(szModel) - 1);
	}
}

static constexpr const char *k_pszVirtualScreenName = "Virtual screen";

/* Resolve a mode from the display we last drove, identified by the persisted EDID. */
static bool get_last_display_mode(saved_mode &mode_info)
{
	// A mode picked while headless is saved under the virtual screen's own name.
	if (get_saved_mode(k_pszVirtualScreenName, mode_info) && mode_info.width > 0 && mode_info.height > 0 && mode_info.refresh > 0)
	{
		// A pick made against an earlier display's mode list does not carry over.
		std::vector<gamescope::BackendMode> modes = gamescope::LoadModeListFile();
		if (modes.empty() || std::any_of(modes.begin(), modes.end(), [&](const gamescope::BackendMode &mode)
			{
				return mode.uWidth == (uint32_t)mode_info.width && mode.uHeight == (uint32_t)mode_info.height && mode.uRefresh == (uint32_t)mode_info.refresh;
			}))
		{
			drm_log.infof("using saved mode %dx%d@%d of the virtual screen",
				mode_info.width, mode_info.height, mode_info.refresh);
			return true;
		}
	}

	const char *pszPath = gamescope::GetPatchedEdidPath();
	if (!pszPath)
		return false;

	FILE *pFile = fopen(pszPath, "rb");
	if (!pFile)
		return false;

	uint8_t edid[4096];
	size_t ulSize = fread(edid, 1, sizeof(edid), pFile);
	fclose(pFile);
	if (!ulSize)
		return false;

	di_info *pInfo = di_info_parse_edid(edid, ulSize);
	if (!pInfo)
		return false;
	defer( di_info_destroy( pInfo ) );

	const di_edid *pEdid = di_info_get_edid(pInfo);

	char szMakePNP[4];
	char szModel[16];
	parse_edid_identity(pEdid, szMakePNP, szModel);

	const char *pszMake = szMakePNP;
	auto pnpIter = pnps.find(szMakePNP);
	if (pnpIter != pnps.end())
		pszMake = pnpIter->second.c_str();

	// Matches the description format setup_best_connector saves modes under.
	char description[256];
	snprintf(description, sizeof(description), "%s %s", pszMake, szModel);

	if (get_saved_mode(description, mode_info) && mode_info.width > 0 && mode_info.height > 0 && mode_info.refresh > 0)
	{
		drm_log.infof("using saved mode %dx%d@%d of last connected display '%s'",
			mode_info.width, mode_info.height, mode_info.refresh, description);
		return true;
	}

	const di_edid_detailed_timing_def *const *pTimings = di_edid_get_detailed_timing_defs(pEdid);
	if (pTimings[0] && !pTimings[0]->interlaced)
	{
		const di_edid_detailed_timing_def *pDef = pTimings[0];
		int64_t lTotalPixels = (int64_t)(pDef->horiz_video + pDef->horiz_blank) * (pDef->vert_video + pDef->vert_blank);
		if (pDef->horiz_video > 0 && pDef->vert_video > 0 && lTotalPixels > 0 && pDef->pixel_clock_hz > 0)
		{
			mode_info.width = pDef->horiz_video;
			mode_info.height = pDef->vert_video;
			mode_info.refresh = (int)((pDef->pixel_clock_hz + lTotalPixels / 2) / lTotalPixels);
			if (mode_info.refresh > 0)
			{
				drm_log.infof("using preferred mode %dx%d@%d of last connected display '%s'",
					mode_info.width, mode_info.height, mode_info.refresh, description);
				return true;
			}
		}
	}

	return false;
}

namespace gamescope
{
	// Stands in for a null current connector while headless.
	class CDRMHeadlessConnector final : public CBaseBackendConnector
	{
	public:
		virtual bool IsHeadless() const override
		{
			return true;
		}
		// The null-connector path this replaces reports id 0, so keep that instead of an auto-assigned id.
		virtual uint64_t GetConnectorID() const override
		{
			return 0;
		}

		virtual GamescopeScreenType GetScreenType() const override
		{
			return GAMESCOPE_SCREEN_TYPE_EXTERNAL;
		}
		virtual GamescopePanelOrientation GetCurrentOrientation() const override
		{
			return GAMESCOPE_PANEL_ORIENTATION_0;
		}
		virtual bool SupportsHDR() const override
		{
			return false;
		}
		virtual bool IsHDRActive() const override
		{
			return false;
		}
		virtual const BackendConnectorHDRInfo &GetHDRInfo() const override
		{
			return m_HDRInfo;
		}
		virtual bool IsVRRActive() const override
		{
			return false;
		}
		virtual std::span<const BackendMode> GetModes() const override
		{
			return m_Modes;
		}

		virtual bool SupportsVRR() const override
		{
			return false;
		}

		virtual std::span<const uint8_t> GetRawEDID() const override
		{
			return std::span<const uint8_t>{};
		}
		virtual std::span<const uint32_t> GetValidDynamicRefreshRates() const override
		{
			return std::span<const uint32_t>{};
		}

		virtual void GetNativeColorimetry(
			bool bHDR10,
			displaycolorimetry_t *displayColorimetry, EOTF *displayEOTF,
			displaycolorimetry_t *outputEncodingColorimetry, EOTF *outputEncodingEOTF ) const override
		{
			*displayColorimetry = displaycolorimetry_709;
			*displayEOTF = EOTF_Gamma22;
			*outputEncodingColorimetry = displaycolorimetry_709;
			*outputEncodingEOTF = EOTF_Gamma22;
		}

		virtual const char *GetName() const override
		{
			return "Headless";
		}
		virtual const char *GetMake() const override
		{
			return "Gamescope";
		}
		virtual const char *GetModel() const override
		{
			return k_pszVirtualScreenName;
		}

		virtual int Present( const FrameInfo_t *pFrameInfo, bool bAsync ) override
		{
			return 0;
		}

		void RebuildModes()
		{
			m_Modes = LoadModeListFile();
			if ( m_Modes.empty() )
				m_Modes.push_back( BackendMode{ (uint32_t)g_nOutputWidth, (uint32_t)g_nOutputHeight, (uint32_t)ConvertmHzToHz( g_nOutputRefresh ) } );
		}

	private:
		BackendConnectorHDRInfo m_HDRInfo{};
		std::vector<BackendMode> m_Modes;
	};
}

static gamescope::CDRMHeadlessConnector s_HeadlessConnector;

static GamescopeBroadcastRGBMode_t s_ExternalBroadcastRGBMode = GAMESCOPE_BROADCAST_RGB_MODE_AUTOMATIC;

static bool setup_best_connector(struct drm_t *drm, bool force, bool initial)
{
	if (drm->pConnector && drm->pConnector->GetModeConnector()->connection != DRM_MODE_CONNECTED) {
		drm_log.infof("current connector '%s' disconnected", drm->pConnector->GetName());
		drm->pConnector = nullptr;
	}

	gamescope::CDRMConnector *best = nullptr;
	int nBestPriority = INT_MAX;
	for ( auto &iter : drm->connectors )
	{
		gamescope::CDRMConnector *pConnector = &iter.second;

		if ( pConnector->GetModeConnector()->connection != DRM_MODE_CONNECTED )
			continue;

		if ( g_bForceInternal && pConnector->GetScreenType() == gamescope::GAMESCOPE_SCREEN_TYPE_EXTERNAL )
			continue;

		int nPriority = get_connector_priority( drm->connector_priorities, pConnector->GetName() );
		if ( nPriority < nBestPriority )
		{
			best = pConnector;
			nBestPriority = nPriority;
		}
	}

	if ( best && best == drm->pConnector )
	{
		// If the device's EDID changed from user us, force a mode-change
		// as we might
		if ( best->HandleEdidChange() )
		{
			force = true;
		}
	}

	if (!force) {
		if ((!best && drm->pConnector) || (best && best == drm->pConnector)) {
			// Let's keep our current connector
			return true;
		}
	}

	if (best == nullptr) {
		drm_log.infof("cannot find any connected connector!");
		drm_unset_connector(drm);
		drm_unset_mode(drm, force);
		s_HeadlessConnector.RebuildModes();

		// Steam keys saved modes by the description, so get_last_display_mode reads this name back.
		const struct wlserver_output_info wlserver_output_info = {
			.description = k_pszVirtualScreenName,
		};
		wlserver_lock();
		wlserver_set_output_info(&wlserver_output_info);
		wlserver_unlock();
		update_connector_display_info_wl( drm );
		return true;
	}

	if (!drm_set_connector(drm, best)) {
		return false;
	}

	char description[256];
	if (best->GetScreenType() == gamescope::GAMESCOPE_SCREEN_TYPE_INTERNAL) {
		snprintf(description, sizeof(description), "Internal screen");
	} else if (best->GetMake() && best->GetModel()) {
		snprintf(description, sizeof(description), "%s %s", best->GetMake(), best->GetModel());
	} else if (best->GetModel()) {
		snprintf(description, sizeof(description), "%s", best->GetModel());
	} else {
		snprintf(description, sizeof(description), "External screen");
	}

	s_ExternalBroadcastRGBMode = GAMESCOPE_BROADCAST_RGB_MODE_AUTOMATIC;

	const drmModeModeInfo *mode = nullptr;
	if ( drm->preferred_width != 0 || drm->preferred_height != 0 || drm->preferred_refresh != 0 )
	{
		mode = find_mode(best->GetModeConnector(), drm->preferred_width, drm->preferred_height, gamescope::ConvertmHzToHz( drm->preferred_refresh ));
	}

	if (!mode && best->GetScreenType() == gamescope::GAMESCOPE_SCREEN_TYPE_EXTERNAL) {
		saved_mode mode_info{};
		if (get_saved_mode(description, mode_info))
		{
			s_ExternalBroadcastRGBMode = mode_info.broadcast_mode;
			mode = find_mode(best->GetModeConnector(), mode_info.width, mode_info.height, mode_info.refresh);
		}
	}

	if (!mode) {
		mode = find_mode(best->GetModeConnector(), 0, 0, 0);
	}

	if (!mode) {
		drm_log.errorf("could not find mode!");
		return false;
	}

	if (!drm_set_mode(drm, mode)) {
		return false;
	}

	// Don't allow rollback of mode_id after connector change
	drm->current.mode_id = drm->pending.mode_id;

	const struct wlserver_output_info wlserver_output_info = {
		.description = description,
		.phys_width = (int) best->GetModeConnector()->mmWidth,
		.phys_height = (int) best->GetModeConnector()->mmHeight,
	};
	wlserver_lock();
	wlserver_set_output_info(&wlserver_output_info);
	wlserver_unlock();

	if (!initial)
	{
		WritePatchedEdid( best->GetRawEDID(), best->GetHDRInfo(), g_bRotated );
		WriteModeListFile( best->GetModes() );
	}

	update_connector_display_info_wl( drm );

	return true;
}

void load_pnps(void)
{
#ifdef HWDATA_PNP_IDS
	const char *filename = HWDATA_PNP_IDS;
	FILE *f = fopen(filename, "r");
	if (!f) {
		drm_log.infof("failed to open PNP IDs file at '%s'", filename);
		return;
	}

	char *line = NULL;
	size_t line_size = 0;
	while (getline(&line, &line_size, f) >= 0) {
		char *nl = strchr(line, '\n');
		if (nl) {
			*nl = '\0';
		}

		char *sep = strchr(line, '\t');
		if (!sep) {
			continue;
		}
		*sep = '\0';

		std::string id(line);
		std::string name(sep + 1);
		pnps[id] = name;
	}

	free(line);
	fclose(f);
#endif
}

extern bool env_to_bool(const char *env);

static void
gamescope_liftoff_log_handler(enum liftoff_log_priority liftoff_priority, const char *fmt, va_list args)
{
	enum LogPriority priority = LOG_DEBUG;
	
	switch ( liftoff_priority )
	{
		case LIFTOFF_ERROR:
			priority = LOG_ERROR;
			break;
		case LIFTOFF_DEBUG:
			priority = LOG_DEBUG;
			break;
		case LIFTOFF_SILENT:
			priority = LOG_SILENT;
			break;
	}

	liftoff_log_scope.vlogf(priority, fmt, args);
}

bool init_drm(struct drm_t *drm, int fd, int width, int height, int refresh)
{
	load_pnps();

	drm->bUseLiftoff = true;

	drm->preferred_width = width;
	drm->preferred_height = height;
	drm->preferred_refresh = refresh;

	drm->fd = fd;
	drm->device_name = nullptr;

	struct stat fd_stat = {};
	if ( fstat( drm->fd, &fd_stat ) == 0 )
	{
		drmDevice *drm_dev = nullptr;
		if ( drmGetDeviceFromDevId( fd_stat.st_rdev, 0, &drm_dev ) == 0 )
		{
			if ( drm_dev->available_nodes & (1 << DRM_NODE_PRIMARY) )
				drm->device_name = strdup( drm_dev->nodes[DRM_NODE_PRIMARY] );
			drmFreeDevice( &drm_dev );
		}
	}

	drm_log.infof( "opening DRM node '%s'", drm->device_name ? drm->device_name : "unknown" );

	if ( !drmIsKMS( drm->fd ) )
	{
		drm_log.errorf( "'%s' is not a KMS device", drm->device_name ? drm->device_name : "unknown" );
		wlsession_close_kms();
		return -1;
	}

	if (drmSetClientCap(drm->fd, DRM_CLIENT_CAP_ATOMIC, 1) != 0) {
		drm_log.errorf("drmSetClientCap(ATOMIC) failed");
		return false;
	}

	drm->is_nvidia = false;
	drm->is_vc4 = false;
	if ( drmVersion *pVersion = drmGetVersion( drm->fd ) )
	{
		drm->is_nvidia = pVersion->name && strcmp( pVersion->name, "nvidia-drm" ) == 0;
		drm->is_vc4 = pVersion->name && strcmp( pVersion->name, "vc4" ) == 0;
		drmFreeVersion( pVersion );
	}

#if !HAVE_GBM
	if ( drm->is_nvidia )
		drm_log.errorf( "Gamescope was built without GBM support; Vulkan scanout allocation will be used on NVIDIA." );
#endif

	if (drmGetCap(drm->fd, DRM_CAP_CURSOR_WIDTH, &drm->cursor_width) != 0) {
		drm->cursor_width = 64;
	}
	if (drmGetCap(drm->fd, DRM_CAP_CURSOR_HEIGHT, &drm->cursor_height) != 0) {
		drm->cursor_height = 64;
	}

	uint64_t cap;
	g_bSupportsSyncObjs = drmGetCap(drm->fd, DRM_CAP_SYNCOBJ, &cap) == 0 && cap != 0;
	if ( !g_bSupportsSyncObjs ) {
		drm_log.errorf("Syncobjs are not supported by the KMS driver");
	}

	if (drmGetCap(drm->fd, DRM_CAP_ADDFB2_MODIFIERS, &cap) == 0 && cap != 0) {
		drm->allow_modifiers = true;
	}

	g_bSupportsAsyncFlips = drmGetCap(drm->fd, DRM_CAP_ATOMIC_ASYNC_PAGE_FLIP, &cap) == 0 && cap != 0;
	if (!g_bSupportsAsyncFlips)
		drm_log.errorf("Immediate flips are not supported by the KMS driver");

	static bool async_disabled = env_to_bool(getenv("GAMESCOPE_DISABLE_ASYNC_FLIPS"));

	if ( async_disabled )
	{
		g_bSupportsAsyncFlips = false;
		drm_log.errorf("Immediate flips disabled from environment");
	}

	if (!get_resources(drm)) {
		return false;
	}
	drm_init_generic_color(drm);

	drm->lo_device = liftoff_device_create( drm->fd );
	if ( drm->lo_device == nullptr )
		return false;
	if ( liftoff_device_register_all_planes( drm->lo_device ) < 0 )
		return false;
	
	drm_log.infof("Connectors:");
	for ( auto &iter : drm->connectors )
	{
		gamescope::CDRMConnector *pConnector = &iter.second;

		const char *status_str = "disconnected";
		if ( pConnector->GetModeConnector()->connection == DRM_MODE_CONNECTED )
			status_str = "connected";

		drm_log.infof("  %s (%s)", pConnector->GetName(), status_str);
	}

	drm->connector_priorities = parse_connector_priorities( g_sOutputName );

	if (!setup_best_connector(drm, true, true)) {
		return false;
	}

	// Fetch formats which can be scanned out
	for ( std::unique_ptr< gamescope::CDRMPlane > &pPlane : drm->planes )
	{
		if ( !get_plane_formats( drm, pPlane.get(), &drm->formats ) )
			return false;
	}

	// TODO: intersect primary planes formats instead
	if ( !drm->pPrimaryPlane )
		drm->pPrimaryPlane = find_primary_plane( drm );

	// Headless start, there is no CRTC to route through yet.
	if ( !drm->pPrimaryPlane )
		drm->pPrimaryPlane = find_any_primary_plane( drm );

	if ( !drm->pPrimaryPlane )
	{
		drm_log.errorf("Failed to find a primary plane");
		return false;
	}

	if ( !get_plane_formats( drm, drm->pPrimaryPlane, &drm->primary_formats ) )
	{
		return false;
	}

	// Pick a 10-bit format at first for our composition buffer, for a couple of reasons:
	//
	// 1. Many game engines automatically render to 10-bit formats such as UE4 which means
	// that when we have to composite, we can keep the same HW dithering that we would get if
	// we just scanned them out directly.
	//
	// 2. When compositing HDR content as a fallback when we undock, it avoids introducing
	// a bunch of horrible banding when going to G2.2 curve.
	// It ensures that we can dither that.
	g_nDRMFormat = drm->is_vc4
		? pick_plane_format(&drm->primary_formats, DRM_FORMAT_XBGR8888, DRM_FORMAT_ABGR8888)
		: pick_plane_format(&drm->primary_formats, DRM_FORMAT_XRGB2101010, DRM_FORMAT_ARGB2101010);
	if ( g_nDRMFormat == DRM_FORMAT_INVALID ) {
		g_nDRMFormat = pick_plane_format(&drm->primary_formats, DRM_FORMAT_XBGR2101010, DRM_FORMAT_ABGR2101010);
		if ( g_nDRMFormat == DRM_FORMAT_INVALID ) {
			g_nDRMFormat = pick_plane_format(&drm->primary_formats, DRM_FORMAT_XRGB8888, DRM_FORMAT_ARGB8888);
			if ( g_nDRMFormat == DRM_FORMAT_INVALID ) {
				drm_log.errorf("Primary plane doesn't support any formats >= 8888");
				return false;
			}
		}
	}

	if (have_overlay_planes(drm)) {
		// ARGB8888 is the Xformat and AFormat here in this function as we want transparent overlay
		g_nDRMFormatOverlay = pick_plane_format(&drm->formats, DRM_FORMAT_ARGB2101010, DRM_FORMAT_ARGB2101010);
		if ( g_nDRMFormatOverlay == DRM_FORMAT_INVALID ) {
			g_nDRMFormatOverlay = pick_plane_format(&drm->formats, DRM_FORMAT_ABGR2101010, DRM_FORMAT_ABGR2101010);
			if ( g_nDRMFormatOverlay == DRM_FORMAT_INVALID ) {
				g_nDRMFormatOverlay = pick_plane_format(&drm->formats, DRM_FORMAT_ARGB8888, DRM_FORMAT_ARGB8888);
				if ( g_nDRMFormatOverlay == DRM_FORMAT_INVALID ) {
					drm_log.errorf("Overlay plane doesn't support any formats >= 8888");
					return false;
				}
			}
		}
	} else {
		switch (g_nDRMFormat) {
		case DRM_FORMAT_XRGB2101010:
		case DRM_FORMAT_ARGB2101010:
			g_nDRMFormatOverlay = DRM_FORMAT_ARGB2101010;
			break;
		case DRM_FORMAT_XBGR2101010:
		case DRM_FORMAT_ABGR2101010:
			g_nDRMFormatOverlay = DRM_FORMAT_ABGR2101010;
			break;
		case DRM_FORMAT_XRGB8888:
		case DRM_FORMAT_ARGB8888:
			g_nDRMFormatOverlay = DRM_FORMAT_ARGB8888;
			break;
		default:
			return false;
		}
	}

	// Create a pipe to wake the flip handler poll for immediate exit.
	// Closing the write end will cause the read end to get POLLHUP.
	if ( pipe2( g_page_flip_pipe_fds, O_CLOEXEC ) != 0 ) {
		drm_log.errorf_errno( "page-flip pipe creation failed" );
		return false;
	}

	g_page_flip_handler_thread_should_exit = false;
	g_page_flip_handler_thread = std::thread( flip_handler_thread_run );

	// Set log priority to the max, liftoff_log_scope will filter for us.
	liftoff_log_set_priority(LIFTOFF_DEBUG);
	liftoff_log_set_handler(gamescope_liftoff_log_handler);

	hdr_output_metadata sdr_metadata;
	memset(&sdr_metadata, 0, sizeof(sdr_metadata));
	drm->sdr_static_metadata = GetBackend()->CreateBackendBlob( sdr_metadata );

	drm->needs_modeset = true;

#if HAVE_GBM
	if ( drm->is_nvidia || drm->is_vc4 )
	{
		drm->gbm = gbm_create_device( drm->fd );
		if ( !drm->gbm )
			drm_log.errorf( "Failed to create GBM device for scanout buffers; Vulkan scanout allocation will be used." );
	}
#endif

	return true;
}

void OnSleepScreenChanged( gamescope::ConVar<bool> & )
{
	force_repaint();
}

gamescope::ConVar<bool> cv_drm_sleep_screens[] =
{
	{ "drm_sleep_internal_screen", false, "Force the internal screen to be asleep", OnSleepScreenChanged },
	{ "drm_sleep_external_screen", false, "Force the external screen to be asleep", OnSleepScreenChanged },
};

static std::mutex s_ScreenSleepMutex;
static std::condition_variable s_ScreenSleepCV;
static bool s_bScreenDisabled[ gamescope::GAMESCOPE_SCREEN_TYPE_COUNT ] = {};

static void drm_update_screen_disabled( struct drm_t *drm )
{
	{
		std::lock_guard lock( s_ScreenSleepMutex );
		for ( bool &bDisabled : s_bScreenDisabled )
			bDisabled = true;

		for ( auto &iter : drm->connectors )
		{
			gamescope::CDRMConnector *pConnector = &iter.second;
			if ( pConnector->GetModeConnector()->connection == DRM_MODE_CONNECTED &&
			     pConnector->GetProperties().CRTC_ID->GetCurrentValue() != 0 )
				s_bScreenDisabled[ pConnector->GetScreenType() ] = false;
		}
	}
	s_ScreenSleepCV.notify_all();
}



void finish_drm(struct drm_t *drm)
{
	// Disable all connectors, CRTCs and planes. This is necessary to leave a
	// clean KMS state behind. Some other KMS clients might not support all of
	// the properties we use, e.g. "rotation" and Xorg don't play well
	// together.

	drmModeAtomicReq *req = drmModeAtomicAlloc();

	for ( auto &iter : drm->connectors )
	{
		gamescope::CDRMConnector *pConnector = &iter.second;

		pConnector->GetProperties().CRTC_ID->SetPendingValue( req, 0, true );

		if ( pConnector->GetProperties().Colorspace )
			pConnector->GetProperties().Colorspace->SetPendingValue( req, 0, true );

		if ( pConnector->GetProperties().HDR_OUTPUT_METADATA )
		{
			if ( drm->sdr_static_metadata && pConnector->GetHDRInfo().IsHDR10() )
				pConnector->GetProperties().HDR_OUTPUT_METADATA->SetPendingValue( req, drm->sdr_static_metadata->GetBlobValue(), true );
			else
				pConnector->GetProperties().HDR_OUTPUT_METADATA->SetPendingValue( req, 0, true );
		}

		if ( pConnector->GetProperties().content_type )
			pConnector->GetProperties().content_type->SetPendingValue( req, 0, true );
	}

	for ( std::unique_ptr< gamescope::CDRMCRTC > &pCRTC : drm->crtcs )
	{
		pCRTC->GetProperties().ACTIVE->SetPendingValue( req, 0, true );
		pCRTC->GetProperties().MODE_ID->SetPendingValue( req, 0, true );

		if ( pCRTC->GetProperties().GAMMA_LUT )
			pCRTC->GetProperties().GAMMA_LUT->SetPendingValue( req, 0, true );

		if ( pCRTC->GetProperties().DEGAMMA_LUT )
			pCRTC->GetProperties().DEGAMMA_LUT->SetPendingValue( req, 0, true );

		if ( pCRTC->GetProperties().CTM )
			pCRTC->GetProperties().CTM->SetPendingValue( req, 0, true );

		if ( pCRTC->GetProperties().VRR_ENABLED )
			pCRTC->GetProperties().VRR_ENABLED->SetPendingValue( req, 0, true );

		if ( pCRTC->GetProperties().OUT_FENCE_PTR )
			pCRTC->GetProperties().OUT_FENCE_PTR->SetPendingValue( req, 0, true );

		if ( pCRTC->GetProperties().AMD_CRTC_REGAMMA_TF )
			pCRTC->GetProperties().AMD_CRTC_REGAMMA_TF->SetPendingValue( req, 0, true );
	}

	for ( std::unique_ptr< gamescope::CDRMPlane > &pPlane : drm->planes )
	{
		pPlane->GetProperties().FB_ID->SetPendingValue( req, 0, true );
		pPlane->GetProperties().IN_FENCE_FD->SetPendingValue( req, -1, true );
		pPlane->GetProperties().CRTC_ID->SetPendingValue( req, 0, true );
		pPlane->GetProperties().SRC_X->SetPendingValue( req, 0, true );
		pPlane->GetProperties().SRC_Y->SetPendingValue( req, 0, true );
		pPlane->GetProperties().SRC_W->SetPendingValue( req, 0, true );
		pPlane->GetProperties().SRC_H->SetPendingValue( req, 0, true );
		pPlane->GetProperties().CRTC_X->SetPendingValue( req, 0, true );
		pPlane->GetProperties().CRTC_Y->SetPendingValue( req, 0, true );
		pPlane->GetProperties().CRTC_W->SetPendingValue( req, 0, true );
		pPlane->GetProperties().CRTC_H->SetPendingValue( req, 0, true );

		if ( pPlane->GetProperties().rotation )
			pPlane->GetProperties().rotation->SetPendingValue( req, DRM_MODE_ROTATE_0, true );

		if ( pPlane->GetProperties().alpha )
			pPlane->GetProperties().alpha->SetPendingValue( req, 0xFFFF, true );

		//if ( pPlane->GetProperties().zpos )
		//	pPlane->GetProperties().zpos->SetPendingValue( req, , true );

		if (drm->generic_plane_color)
		{
			if (pPlane->GetProperties().COLOR_PIPELINE)
				pPlane->GetProperties().COLOR_PIPELINE->SetPendingValue(req, 0, true);
			continue;
		}

		if ( pPlane->GetProperties().AMD_PLANE_DEGAMMA_TF )
			pPlane->GetProperties().AMD_PLANE_DEGAMMA_TF->SetPendingValue( req, AMDGPU_TRANSFER_FUNCTION_DEFAULT, true );

		if ( pPlane->GetProperties().AMD_PLANE_DEGAMMA_LUT )
			pPlane->GetProperties().AMD_PLANE_DEGAMMA_LUT->SetPendingValue( req, 0, true );

		if ( pPlane->GetProperties().AMD_PLANE_CTM )
			pPlane->GetProperties().AMD_PLANE_CTM->SetPendingValue( req, 0, true );

		if ( pPlane->GetProperties().AMD_PLANE_HDR_MULT )
			pPlane->GetProperties().AMD_PLANE_HDR_MULT->SetPendingValue( req, 0x100000000ULL, true );

		if ( pPlane->GetProperties().AMD_PLANE_SHAPER_TF )
			pPlane->GetProperties().AMD_PLANE_SHAPER_TF->SetPendingValue( req, AMDGPU_TRANSFER_FUNCTION_DEFAULT, true );

		if ( pPlane->GetProperties().AMD_PLANE_SHAPER_LUT )
			pPlane->GetProperties().AMD_PLANE_SHAPER_LUT->SetPendingValue( req, 0, true );

		if ( pPlane->GetProperties().AMD_PLANE_LUT3D )
			pPlane->GetProperties().AMD_PLANE_LUT3D->SetPendingValue( req, 0, true );

		if ( pPlane->GetProperties().AMD_PLANE_BLEND_TF )
			pPlane->GetProperties().AMD_PLANE_BLEND_TF->SetPendingValue( req, AMDGPU_TRANSFER_FUNCTION_DEFAULT, true );

		if ( pPlane->GetProperties().AMD_PLANE_BLEND_LUT )
			pPlane->GetProperties().AMD_PLANE_BLEND_LUT->SetPendingValue( req, 0, true );
	}

	// We can't do a non-blocking commit here or else risk EBUSY in case the
	// previous page-flip is still in flight.
	uint32_t flags = DRM_MODE_ATOMIC_ALLOW_MODESET;
	int ret = drmModeAtomicCommit( drm->fd, req, flags, nullptr );
	if ( ret != 0 ) {
		drm_log.errorf_errno( "finish_drm: drmModeAtomicCommit failed" );
	}
	drmModeAtomicFree(req);

	// Signal the page-flip handler thread to exit and join it so it won't be
	// using the DRM fd while we clean it up. Closing the pipe write end
	// causes the read end to get POLLHUP, waking the thread.
	if ( g_page_flip_handler_thread.joinable() ) {
		g_page_flip_handler_thread_should_exit.store( true, std::memory_order_release );

		close( g_page_flip_pipe_fds[1] );
		g_page_flip_pipe_fds[1] = -1;

		g_page_flip_handler_thread.join();

		close( g_page_flip_pipe_fds[0] );
		g_page_flip_pipe_fds[0] = -1;
	}


	free(drm->device_name);

	wlr_drm_format_set_finish( &drm->formats );
	wlr_drm_format_set_finish( &drm->primary_formats );
	drm->m_FbIdsInRequest.clear();
	drm_close_in_fences( drm );
	{
		std::unique_lock lock( drm->m_QueuedFbIdsMutex );
		drm->m_QueuedFbIds.clear();
	}
	{
		std::unique_lock lock( drm->m_mutVisibleFbIds );
		drm->m_VisibleFbIds.clear();
	}
	drm->sdr_static_metadata = nullptr;
	drm->color_program_cache.clear();
	drm->color_output_cache.clear();
	drm->color_failed_frames.clear();
	drm->color_last_roots.clear();
	drm->color_pipelines.clear();
	drm->current = drm_t::drm_state_t{};
	drm->pending = drm_t::drm_state_t{};
	drm->planes.clear();
	drm->crtcs.clear();
	drm->connectors.clear();

#if HAVE_GBM
	if ( drm->gbm )
	{
		gbm_device_destroy( drm->gbm );
		drm->gbm = nullptr;
	}
#endif



	wlsession_close_kms();
	g_DRM.fd = -1;
}

gamescope::OwningRc<gamescope::IBackendFb> drm_fbid_from_dmabuf( struct drm_t *drm, struct wlr_dmabuf_attributes *dma_buf )
{
	gamescope::OwningRc<gamescope::IBackendFb> pBackendFb;
	uint32_t fb_id = 0;

	if ( !wlr_drm_format_set_has( &drm->formats, dma_buf->format, dma_buf->modifier ) )
	{
		drm_log.errorf( "Cannot import FB to DRM: format 0x%" PRIX32 " and modifier 0x%" PRIX64 " not supported for scan-out", dma_buf->format, dma_buf->modifier );
		return nullptr;
	}

	uint32_t handles[4] = {0};
	uint64_t modifiers[4] = {0};
	for ( int i = 0; i < dma_buf->n_planes; i++ ) {
		if ( drmPrimeFDToHandle( drm->fd, dma_buf->fd[i], &handles[i] ) != 0 )
		{
			drm_log.errorf_errno("drmPrimeFDToHandle failed");
			goto out;
		}

		/* KMS requires all planes to have the same modifier */
		modifiers[i] = dma_buf->modifier;
	}

	if ( dma_buf->modifier != DRM_FORMAT_MOD_INVALID )
	{
		if ( !drm->allow_modifiers )
		{
			drm_log.errorf("Cannot import DMA-BUF: has a modifier (0x%" PRIX64 "), but KMS doesn't support them", dma_buf->modifier);
			goto out;
		}

		if ( drmModeAddFB2WithModifiers( drm->fd, dma_buf->width, dma_buf->height, dma_buf->format, handles, dma_buf->stride, dma_buf->offset, modifiers, &fb_id, DRM_MODE_FB_MODIFIERS ) != 0 )
		{
			drm_log.errorf_errno("drmModeAddFB2WithModifiers failed");
			goto out;
		}
	}
	else
	{
		if ( drmModeAddFB2( drm->fd, dma_buf->width, dma_buf->height, dma_buf->format, handles, dma_buf->stride, dma_buf->offset, &fb_id, 0 ) != 0 )
		{
			drm_log.errorf_errno("drmModeAddFB2 failed");
			goto out;
		}
	}

	drm_log.debugf("make fbid %u", fb_id);

	pBackendFb = new gamescope::CDRMFb( fb_id );

out:
	for ( int i = 0; i < dma_buf->n_planes; i++ ) {
		if ( handles[i] == 0 )
			continue;

		// GEM handles aren't ref'counted by the kernel. Two DMA-BUFs may
		// return the same GEM handle, we need to be careful not to
		// double-close them.
		bool already_closed = false;
		for ( int j = 0; j < i; j++ ) {
			if ( handles[i] == handles[j] )
				already_closed = true;
		}
		if ( already_closed )
			continue;

		struct drm_gem_close args = { .handle = handles[i] };
		if ( drmIoctl( drm->fd, DRM_IOCTL_GEM_CLOSE, &args ) != 0 ) {
			drm_log.errorf_errno( "drmIoctl(GEM_CLOSE) failed" );
		}
	}

	return pBackendFb;
}

static void update_drm_effective_orientations( struct drm_t *drm, const drmModeModeInfo *pMode )
{
	gamescope::IBackendConnector *pInternalConnector = GetBackend()->GetConnector( gamescope::GAMESCOPE_SCREEN_TYPE_INTERNAL );
	if ( pInternalConnector )
	{
		gamescope::CDRMConnector *pDRMInternalConnector = static_cast<gamescope::CDRMConnector *>( pInternalConnector );
		const drmModeModeInfo *pInternalMode = pMode;
		if ( pDRMInternalConnector != drm->pConnector )
			pInternalMode = find_mode( pDRMInternalConnector->GetModeConnector(), 0, 0, 0 );

		pDRMInternalConnector->UpdateEffectiveOrientation( pInternalMode );
	}

	gamescope::IBackendConnector *pExternalConnector = GetBackend()->GetConnector( gamescope::GAMESCOPE_SCREEN_TYPE_EXTERNAL );
	if ( pExternalConnector )
	{
		gamescope::CDRMConnector *pDRMExternalConnector = static_cast<gamescope::CDRMConnector *>( pExternalConnector );
		const drmModeModeInfo *pExternalMode = pMode;
		if ( pDRMExternalConnector != drm->pConnector )
			pExternalMode = find_mode( pDRMExternalConnector->GetModeConnector(), 0, 0, 0 );

		pDRMExternalConnector->UpdateEffectiveOrientation( pExternalMode );
	}
}

// Only used for NV12 buffers
static drm_color_encoding drm_get_color_encoding(EStreamColorspace colorspace)
{
	switch (colorspace)
	{
		default:
		case k_EStreamColorspace_Unknown:
			return DRM_COLOR_YCBCR_BT709;

		case k_EStreamColorspace_BT601:
			return DRM_COLOR_YCBCR_BT601;
		case k_EStreamColorspace_BT601_Full:
			return DRM_COLOR_YCBCR_BT601;

		case k_EStreamColorspace_BT709:
			return DRM_COLOR_YCBCR_BT709;
		case k_EStreamColorspace_BT709_Full:
			return DRM_COLOR_YCBCR_BT709;
	}
}

static drm_color_range drm_get_color_range(EStreamColorspace colorspace)
{
	switch (colorspace)
	{
		default:
		case k_EStreamColorspace_Unknown:
			return DRM_COLOR_YCBCR_FULL_RANGE;

		case k_EStreamColorspace_BT601:
			return DRM_COLOR_YCBCR_LIMITED_RANGE;
		case k_EStreamColorspace_BT601_Full:
			return DRM_COLOR_YCBCR_FULL_RANGE;

		case k_EStreamColorspace_BT709:
			return DRM_COLOR_YCBCR_LIMITED_RANGE;
		case k_EStreamColorspace_BT709_Full:
			return DRM_COLOR_YCBCR_FULL_RANGE;
	}
}

template <typename T>
void hash_combine(size_t& s, const T& v)
{
	std::hash<T> h;
	s^= h(v) + 0x9e3779b9 + (s<< 6) + (s>> 2);
}

struct LiftoffStateCacheEntry
{
	LiftoffStateCacheEntry()
	{
		memset(this, 0, sizeof(LiftoffStateCacheEntry));
	}

    int nLayerCount;

	struct LiftoffLayerState_t
	{
		bool ycbcr;
		uint32_t zpos;
		uint32_t srcW, srcH;
		uint32_t crtcX, crtcY, crtcW, crtcH;
		uint16_t opacity;
		drm_color_encoding colorEncoding;
		drm_color_range    colorRange;
		GamescopeAppTextureColorspace colorspace;
		AlphaBlendingMode_t eAlphaBlendingMode;
	} layerState[ k_nMaxLayers ];

	bool operator == (const LiftoffStateCacheEntry& entry) const
	{
		return !memcmp(this, &entry, sizeof(LiftoffStateCacheEntry));
	}
};

struct LiftoffStateCacheEntryKasher
{
	size_t operator()(const LiftoffStateCacheEntry& k) const
	{
		size_t hash = 0;
		hash_combine(hash, k.nLayerCount);
		for ( int i = 0; i < k.nLayerCount; i++ )
		{
			hash_combine(hash, k.layerState[i].ycbcr);
			hash_combine(hash, k.layerState[i].zpos);
			hash_combine(hash, k.layerState[i].srcW);
			hash_combine(hash, k.layerState[i].srcH);
			hash_combine(hash, k.layerState[i].crtcX);
			hash_combine(hash, k.layerState[i].crtcY);
			hash_combine(hash, k.layerState[i].crtcW);
			hash_combine(hash, k.layerState[i].crtcH);
			hash_combine(hash, k.layerState[i].opacity);
			hash_combine(hash, k.layerState[i].colorEncoding);
			hash_combine(hash, k.layerState[i].colorRange);
			hash_combine(hash, k.layerState[i].colorspace);
			hash_combine(hash, k.layerState[i].eAlphaBlendingMode);
		}

		return hash;
  	}
};


std::unordered_set<LiftoffStateCacheEntry, LiftoffStateCacheEntryKasher> g_LiftoffStateCache;

static inline amdgpu_transfer_function colorspace_to_plane_degamma_tf(GamescopeAppTextureColorspace colorspace)
{
	switch ( colorspace )
	{
		default: // Linear in this sense is SRGB. Linear = sRGB image view doing automatic sRGB -> Linear which doesn't happen on DRM side.
		case GAMESCOPE_APP_TEXTURE_COLORSPACE_SRGB:
			return AMDGPU_TRANSFER_FUNCTION_SRGB_EOTF;
		case GAMESCOPE_APP_TEXTURE_COLORSPACE_PASSTHRU:
		case GAMESCOPE_APP_TEXTURE_COLORSPACE_SCRGB:
			// Use LINEAR TF for scRGB float format as 80 nit = 1.0 in scRGB, which matches
			// what PQ TF decodes to/encodes from.
			// AMD internal format is FP16, and generally expected for 1.0 -> 80 nit.
			// which just so happens to match scRGB.
			return AMDGPU_TRANSFER_FUNCTION_IDENTITY;
		case GAMESCOPE_APP_TEXTURE_COLORSPACE_HDR10_PQ:
			return AMDGPU_TRANSFER_FUNCTION_PQ_EOTF;
	}
}

static inline amdgpu_transfer_function colorspace_to_plane_shaper_tf(GamescopeAppTextureColorspace colorspace)
{
	switch ( colorspace )
	{
		default:
		case GAMESCOPE_APP_TEXTURE_COLORSPACE_SRGB:
			return AMDGPU_TRANSFER_FUNCTION_SRGB_INV_EOTF;
		case GAMESCOPE_APP_TEXTURE_COLORSPACE_SCRGB: // scRGB Linear -> PQ for shaper + 3D LUT
		case GAMESCOPE_APP_TEXTURE_COLORSPACE_HDR10_PQ:
			return AMDGPU_TRANSFER_FUNCTION_PQ_INV_EOTF;
		case GAMESCOPE_APP_TEXTURE_COLORSPACE_PASSTHRU:
			return AMDGPU_TRANSFER_FUNCTION_DEFAULT;
	}
}

static inline amdgpu_transfer_function inverse_tf(amdgpu_transfer_function tf)
{
	switch ( tf )
	{
		default:
		case AMDGPU_TRANSFER_FUNCTION_DEFAULT:
			return AMDGPU_TRANSFER_FUNCTION_DEFAULT;
		case AMDGPU_TRANSFER_FUNCTION_IDENTITY:
			return AMDGPU_TRANSFER_FUNCTION_IDENTITY;
		case AMDGPU_TRANSFER_FUNCTION_SRGB_EOTF:
			return AMDGPU_TRANSFER_FUNCTION_SRGB_INV_EOTF;
		case AMDGPU_TRANSFER_FUNCTION_BT709_OETF:
			return AMDGPU_TRANSFER_FUNCTION_BT709_INV_OETF;
		case AMDGPU_TRANSFER_FUNCTION_PQ_EOTF:
			return AMDGPU_TRANSFER_FUNCTION_PQ_INV_EOTF;
		case AMDGPU_TRANSFER_FUNCTION_GAMMA22_EOTF:
			return AMDGPU_TRANSFER_FUNCTION_GAMMA22_INV_EOTF;
		case AMDGPU_TRANSFER_FUNCTION_GAMMA24_EOTF:
			return AMDGPU_TRANSFER_FUNCTION_GAMMA24_INV_EOTF;
		case AMDGPU_TRANSFER_FUNCTION_GAMMA26_EOTF:
			return AMDGPU_TRANSFER_FUNCTION_GAMMA26_INV_EOTF;
		case AMDGPU_TRANSFER_FUNCTION_SRGB_INV_EOTF:
			return AMDGPU_TRANSFER_FUNCTION_SRGB_EOTF;
		case AMDGPU_TRANSFER_FUNCTION_BT709_INV_OETF:
			return AMDGPU_TRANSFER_FUNCTION_BT709_OETF;
		case AMDGPU_TRANSFER_FUNCTION_PQ_INV_EOTF:
			return AMDGPU_TRANSFER_FUNCTION_PQ_EOTF;
		case AMDGPU_TRANSFER_FUNCTION_GAMMA22_INV_EOTF:
			return AMDGPU_TRANSFER_FUNCTION_GAMMA22_EOTF;
		case AMDGPU_TRANSFER_FUNCTION_GAMMA24_INV_EOTF:
			return AMDGPU_TRANSFER_FUNCTION_GAMMA24_EOTF;
		case AMDGPU_TRANSFER_FUNCTION_GAMMA26_INV_EOTF:
			return AMDGPU_TRANSFER_FUNCTION_GAMMA26_EOTF;
	}
}

static inline uint32_t ColorSpaceToEOTFIndex( GamescopeAppTextureColorspace colorspace )
{
	switch ( colorspace )
	{
		default:
		case GAMESCOPE_APP_TEXTURE_COLORSPACE_LINEAR: // Not actually linear, just Linear vs sRGB image views in Vulkan. Still viewed as sRGB on the DRM side.
		case GAMESCOPE_APP_TEXTURE_COLORSPACE_SRGB:
			// SDR sRGB content treated as native Gamma 22 curve. No need to do sRGB -> 2.2 or whatever.
			return EOTF_Gamma22;
		case GAMESCOPE_APP_TEXTURE_COLORSPACE_SCRGB:
			// Okay, so this is WEIRD right? OKAY Let me explain it to you.
			// The plan for scRGB content is to go from scRGB -> PQ in a SHAPER_TF
			// before indexing into the shaper. (input from colorspace_to_plane_regamma_tf!)
			return EOTF_PQ;
		case GAMESCOPE_APP_TEXTURE_COLORSPACE_HDR10_PQ:
			return EOTF_PQ;
	}
}


LiftoffStateCacheEntry FrameInfoToLiftoffStateCacheEntry( struct drm_t *drm, const FrameInfo_t *frameInfo )
{
	LiftoffStateCacheEntry entry{};

	entry.nLayerCount = frameInfo->layers.count();
	for ( int i = 0; i < entry.nLayerCount; i++ )
	{
		const uint16_t srcWidth  = frameInfo->layers.get( i ).tex->width();
		const uint16_t srcHeight = frameInfo->layers.get( i ).tex->height();

		int32_t crtcX = -frameInfo->layers.get( i ).offset.x;
		int32_t crtcY = -frameInfo->layers.get( i ).offset.y;
		uint64_t crtcW = srcWidth / frameInfo->layers.get( i ).scale.x;
		uint64_t crtcH = srcHeight / frameInfo->layers.get( i ).scale.y;

		if (g_bRotated && g_uOutputRotation == 0)
		{
			int64_t imageH = frameInfo->layers.get( i ).tex->contentHeight() / frameInfo->layers.get( i ).scale.y;

			const int32_t x = crtcX;
			const uint64_t w = crtcW;
			crtcX = g_nOutputHeight - imageH - crtcY;
			crtcY = x;
			crtcW = crtcH;
			crtcH = w;
		}

		entry.layerState[i].zpos  = frameInfo->layers.get( i ).zpos;
		entry.layerState[i].srcW  = srcWidth  << 16;
		entry.layerState[i].srcH  = srcHeight << 16;
		entry.layerState[i].crtcX = crtcX;
		entry.layerState[i].crtcY = crtcY;
		entry.layerState[i].crtcW = crtcW;
		entry.layerState[i].crtcH = crtcH;
		entry.layerState[i].opacity = frameInfo->layers.get( i ).opacity * 0xffff;
		entry.layerState[i].ycbcr = frameInfo->layers.get( i ).isYcbcr();
		if ( entry.layerState[i].ycbcr )
		{
			entry.layerState[i].colorEncoding = drm_get_color_encoding( g_ForcedNV12ColorSpace );
			entry.layerState[i].colorRange    = drm_get_color_range( g_ForcedNV12ColorSpace );
			entry.layerState[i].colorspace = GAMESCOPE_APP_TEXTURE_COLORSPACE_SRGB;
		}
		else
		{
			entry.layerState[i].colorspace = frameInfo->layers.get( i ).colorspace;
		}
		entry.layerState[i].eAlphaBlendingMode = frameInfo->layers.get( i ).eAlphaBlendingMode;
	}

	return entry;
}

static bool is_liftoff_caching_enabled()
{
	static bool disabled = env_to_bool(getenv("GAMESCOPE_LIFTOFF_CACHE_DISABLE"));
	return !disabled;
}

namespace gamescope
{
	////////////////////
	// CDRMAtomicObject
	////////////////////
	CDRMAtomicObject::CDRMAtomicObject( uint32_t ulObjectId )
		: m_ulObjectId{ ulObjectId }
	{
	}


	/////////////////////////
	// CDRMAtomicTypedObject
	/////////////////////////
	template < uint32_t DRMObjectType >
	CDRMAtomicTypedObject<DRMObjectType>::CDRMAtomicTypedObject( uint32_t ulObjectId )
		: CDRMAtomicObject{ ulObjectId }
	{
	}

	template < uint32_t DRMObjectType >
	std::optional<DRMObjectRawProperties> CDRMAtomicTypedObject<DRMObjectType>::GetRawProperties()
	{
		drmModeObjectProperties *pProperties = drmModeObjectGetProperties( g_DRM.fd, m_ulObjectId, DRMObjectType );
		if ( !pProperties )
		{
			drm_log.errorf_errno( "drmModeObjectGetProperties failed" );
			return std::nullopt;
		}
		defer( drmModeFreeObjectProperties( pProperties ) );

		DRMObjectRawProperties rawProperties;
		for ( uint32_t i = 0; i < pProperties->count_props; i++ )
		{
			drmModePropertyRes *pProperty = drmModeGetProperty( g_DRM.fd, pProperties->props[ i ] );
			if ( !pProperty )
				continue;
			defer( drmModeFreeProperty( pProperty ) );

			rawProperties[ pProperty->name ] = DRMObjectRawProperty{ pProperty->prop_id, pProperties->prop_values[ i ] };
		}

		return rawProperties;
	}


	/////////////////////////
	// CDRMAtomicProperty
	/////////////////////////
	CDRMAtomicProperty::CDRMAtomicProperty( CDRMAtomicObject *pObject, DRMObjectRawProperty rawProperty )
		: m_pObject{ pObject }
		, m_uPropertyId{ rawProperty.uPropertyId }
		, m_ulPendingValue{ rawProperty.ulValue }
		, m_ulCurrentValue{ rawProperty.ulValue }
		, m_ulInitialValue{ rawProperty.ulValue }
	{
	}

	/*static*/ std::optional<CDRMAtomicProperty> CDRMAtomicProperty::Instantiate( const char *pszName, CDRMAtomicObject *pObject, const DRMObjectRawProperties& rawProperties )
	{
		auto iter = rawProperties.find( pszName );
		if ( iter == rawProperties.end() )
			return std::nullopt;

		return CDRMAtomicProperty{ pObject, iter->second };
	}

	int CDRMAtomicProperty::SetPendingValue( drmModeAtomicReq *pRequest, uint64_t ulValue, bool bForce /*= false*/ )
	{
		// In instances where we rolled back due to -EINVAL, or we want to ensure a value from an unclean state
		// eg. from an unclean or other initial state, you can force an update in the request with bForce.

		if ( ulValue == m_ulPendingValue && !bForce )
			return 0;

		int ret = drmModeAtomicAddProperty( pRequest, m_pObject->GetObjectId(), m_uPropertyId, ulValue );
		if ( ret < 0 )
			return ret;

		m_ulPendingValue = ulValue;
		return ret;
	}

	void CDRMAtomicProperty::OnCommit()
	{
		m_ulCurrentValue = m_ulPendingValue;
	}

	void CDRMAtomicProperty::Rollback()
	{
		m_ulPendingValue = m_ulCurrentValue;
	}

	/////////////////////////
	// CDRMPlane
	/////////////////////////
	CDRMPlane::CDRMPlane( drmModePlane *pPlane )
		: CDRMAtomicTypedObject<DRM_MODE_OBJECT_PLANE>( pPlane->plane_id )
		, m_pPlane{ pPlane, []( drmModePlane *pPlane ){ drmModeFreePlane( pPlane ); } }
	{
		RefreshState();
	}

	void CDRMPlane::RefreshState()
	{
		auto rawProperties = GetRawProperties();
		if ( rawProperties )
		{
			m_Props.type                  = CDRMAtomicProperty::Instantiate( "type",                  this, *rawProperties );
			m_Props.IN_FORMATS            = CDRMAtomicProperty::Instantiate( "IN_FORMATS",            this, *rawProperties );

			m_Props.FB_ID                    = CDRMAtomicProperty::Instantiate( "FB_ID",                    this, *rawProperties );
			m_Props.IN_FENCE_FD              = CDRMAtomicProperty::Instantiate( "IN_FENCE_FD",              this, *rawProperties );
			m_Props.CRTC_ID                  = CDRMAtomicProperty::Instantiate( "CRTC_ID",                  this, *rawProperties );
			m_Props.SRC_X                    = CDRMAtomicProperty::Instantiate( "SRC_X",                    this, *rawProperties );
			m_Props.SRC_Y                    = CDRMAtomicProperty::Instantiate( "SRC_Y",                    this, *rawProperties );
			m_Props.SRC_W                    = CDRMAtomicProperty::Instantiate( "SRC_W",                    this, *rawProperties );
			m_Props.SRC_H                    = CDRMAtomicProperty::Instantiate( "SRC_H",                    this, *rawProperties );
			m_Props.CRTC_X                   = CDRMAtomicProperty::Instantiate( "CRTC_X",                   this, *rawProperties );
			m_Props.CRTC_Y                   = CDRMAtomicProperty::Instantiate( "CRTC_Y",                   this, *rawProperties );
			m_Props.CRTC_W                   = CDRMAtomicProperty::Instantiate( "CRTC_W",                   this, *rawProperties );
			m_Props.CRTC_H                   = CDRMAtomicProperty::Instantiate( "CRTC_H",                   this, *rawProperties );
			m_Props.zpos                     = CDRMAtomicProperty::Instantiate( "zpos",                     this, *rawProperties );
			m_Props.alpha                    = CDRMAtomicProperty::Instantiate( "alpha",                    this, *rawProperties );
			m_Props.rotation                 = CDRMAtomicProperty::Instantiate( "rotation",                 this, *rawProperties );
			m_Props.COLOR_ENCODING           = CDRMAtomicProperty::Instantiate( "COLOR_ENCODING",           this, *rawProperties );
			m_Props.COLOR_RANGE              = CDRMAtomicProperty::Instantiate( "COLOR_RANGE",              this, *rawProperties );
			m_Props.COLOR_PIPELINE           = CDRMAtomicProperty::Instantiate( "COLOR_PIPELINE",           this, *rawProperties );
			m_Props.AMD_PLANE_DEGAMMA_TF     = CDRMAtomicProperty::Instantiate( "AMD_PLANE_DEGAMMA_TF",     this, *rawProperties );
			m_Props.AMD_PLANE_DEGAMMA_LUT    = CDRMAtomicProperty::Instantiate( "AMD_PLANE_DEGAMMA_LUT",    this, *rawProperties );
			m_Props.AMD_PLANE_CTM            = CDRMAtomicProperty::Instantiate( "AMD_PLANE_CTM",            this, *rawProperties );
			m_Props.AMD_PLANE_HDR_MULT       = CDRMAtomicProperty::Instantiate( "AMD_PLANE_HDR_MULT",       this, *rawProperties );
			m_Props.AMD_PLANE_SHAPER_LUT     = CDRMAtomicProperty::Instantiate( "AMD_PLANE_SHAPER_LUT",     this, *rawProperties );
			m_Props.AMD_PLANE_SHAPER_TF      = CDRMAtomicProperty::Instantiate( "AMD_PLANE_SHAPER_TF",      this, *rawProperties );
			m_Props.AMD_PLANE_LUT3D          = CDRMAtomicProperty::Instantiate( "AMD_PLANE_LUT3D",          this, *rawProperties );
			m_Props.AMD_PLANE_BLEND_TF       = CDRMAtomicProperty::Instantiate( "AMD_PLANE_BLEND_TF",       this, *rawProperties );
			m_Props.AMD_PLANE_BLEND_LUT      = CDRMAtomicProperty::Instantiate( "AMD_PLANE_BLEND_LUT",      this, *rawProperties );
		}
	}

	/////////////////////////
	// CDRMCRTC
	/////////////////////////
	CDRMCRTC::CDRMCRTC( drmModeCrtc *pCRTC, uint32_t uCRTCMask )
		: CDRMAtomicTypedObject<DRM_MODE_OBJECT_CRTC>( pCRTC->crtc_id )
		, m_pCRTC{ pCRTC, []( drmModeCrtc *pCRTC ){ drmModeFreeCrtc( pCRTC ); } }
		, m_uCRTCMask{ uCRTCMask }
	{
		RefreshState();
	}

	void CDRMCRTC::RefreshState()
	{
		auto rawProperties = GetRawProperties();
		if ( rawProperties )
		{
			m_Props.ACTIVE              = CDRMAtomicProperty::Instantiate( "ACTIVE",              this, *rawProperties );
			auto gammaSize = rawProperties->find("GAMMA_LUT_SIZE");
			auto degammaSize = rawProperties->find("DEGAMMA_LUT_SIZE");
			m_uGammaSize = gammaSize != rawProperties->end() ? gammaSize->second.ulValue : 0;
			m_uDegammaSize = degammaSize != rawProperties->end() ? degammaSize->second.ulValue : 0;
			m_Props.MODE_ID             = CDRMAtomicProperty::Instantiate( "MODE_ID",             this, *rawProperties );
			m_Props.GAMMA_LUT           = CDRMAtomicProperty::Instantiate( "GAMMA_LUT",           this, *rawProperties );
			m_Props.DEGAMMA_LUT         = CDRMAtomicProperty::Instantiate( "DEGAMMA_LUT",         this, *rawProperties );
			m_Props.CTM                 = CDRMAtomicProperty::Instantiate( "CTM",                 this, *rawProperties );
			m_Props.VRR_ENABLED         = CDRMAtomicProperty::Instantiate( "VRR_ENABLED",         this, *rawProperties );
			m_Props.OUT_FENCE_PTR       = CDRMAtomicProperty::Instantiate( "OUT_FENCE_PTR",       this, *rawProperties );
			m_Props.AMD_CRTC_REGAMMA_TF = CDRMAtomicProperty::Instantiate( "AMD_CRTC_REGAMMA_TF", this, *rawProperties );
		}
	}

	/////////////////////////
	// CDRMConnector
	/////////////////////////
	CDRMConnector::CDRMConnector( CDRMBackend *pBackend, drmModeConnector *pConnector )
		: CDRMAtomicTypedObject<DRM_MODE_OBJECT_CONNECTOR>( pConnector->connector_id )
		, m_pBackend{ pBackend }
		, m_pConnector{ pConnector, []( drmModeConnector *pConnector ){ drmModeFreeConnector( pConnector ); } }
	{
		RefreshState();
	}

	void CDRMConnector::RefreshState()
	{
		// For the connector re-poll the drmModeConnector to get new modes, etc.
		// This isn't needed for CRTC/Planes in which the state is immutable for their lifetimes.
		// Connectors can be re-plugged.

		// TODO: Clean this up.
		m_pConnector = CAutoDeletePtr< drmModeConnector >
		{
			drmModeGetConnector( g_DRM.fd, m_pConnector->connector_id ),
			[]( drmModeConnector *pConnector ){ drmModeFreeConnector( pConnector ); }
		};

		// Sort the modes to our preference.
		std::stable_sort( m_pConnector->modes, m_pConnector->modes + m_pConnector->count_modes, []( const drmModeModeInfo &a, const drmModeModeInfo &b )
		{
			bool bGoodRefreshA = a.vrefresh >= 60;
			bool bGoodRefreshB = b.vrefresh >= 60;
			if (bGoodRefreshA != bGoodRefreshB)
				return bGoodRefreshA;

			// Some tvs may report 60hz as their preferred output even
			// if they support 120hz. Prefer 120hz modes if available.
			// Still honor preferred mode, above 120hz things might get
			// funny, such as 360hz. Leave those for manual override.
			// Only prefer 120hz if the resolution is high, fixes
			// SuiPlay 0x1 having a weird 160hz mode at 800p
			int nAreaA = a.hdisplay * a.vdisplay;
			int nAreaB = b.hdisplay * b.vdisplay;
			bGoodRefreshA = a.vrefresh >= 120;
			bGoodRefreshB = b.vrefresh >= 120;
			if (nAreaA > nAreaB && bGoodRefreshA && !bGoodRefreshB)
				return true;

			bool bPreferredA = a.type & DRM_MODE_TYPE_PREFERRED;
			bool bPreferredB = b.type & DRM_MODE_TYPE_PREFERRED;
			if (bPreferredA != bPreferredB)
				return bPreferredA;

			if (nAreaA != nAreaB)
				return nAreaA > nAreaB;

			return a.vrefresh > b.vrefresh;
		} );

		std::vector<uint8_t> oldEdid = std::move( m_Mutable.EdidData );
		m_Mutable.EdidData.clear();

		// Clear this information out.
		m_Mutable = MutableConnectorState{};

		m_Mutable.uPossibleCRTCMask = drmModeConnectorGetPossibleCrtcs( g_DRM.fd, GetModeConnector() );

		// These are string constants from libdrm, no free.
		const char *pszTypeStr = drmModeGetConnectorTypeName( GetModeConnector()->connector_type );
		if ( !pszTypeStr )
			pszTypeStr = "Unknown";

		snprintf( m_Mutable.szName, sizeof( m_Mutable.szName ), "%s-%d", pszTypeStr, GetModeConnector()->connector_type_id );
		m_Mutable.szName[ sizeof( m_Mutable.szName ) - 1 ] = '\0';

		for ( int i = 0; i < m_pConnector->count_modes; i++ )
		{
			drmModeModeInfo *pMode = &m_pConnector->modes[i];
			m_Mutable.BackendModes.emplace_back( BackendMode
			{
				.uWidth   = pMode->hdisplay,
				.uHeight  = pMode->vdisplay,
				.uRefresh = pMode->vrefresh,
			});
		}

		auto rawProperties = GetRawProperties();
		if ( rawProperties )
		{
			m_Props.CRTC_ID                  = CDRMAtomicProperty::Instantiate( "CRTC_ID",                this, *rawProperties );
			m_Props.Colorspace               = CDRMAtomicProperty::Instantiate( "Colorspace",             this, *rawProperties );
			m_Props.content_type             = CDRMAtomicProperty::Instantiate( "content type",           this, *rawProperties );
			m_Props.panel_orientation        = CDRMAtomicProperty::Instantiate( "panel orientation",      this, *rawProperties );
			m_Props.HDR_OUTPUT_METADATA      = CDRMAtomicProperty::Instantiate( "HDR_OUTPUT_METADATA",    this, *rawProperties );
			m_Props.vrr_capable              = CDRMAtomicProperty::Instantiate( "vrr_capable",            this, *rawProperties );
			m_Props.EDID                     = CDRMAtomicProperty::Instantiate( "EDID",                   this, *rawProperties );
			m_Props.Broadcast_RGB            = CDRMAtomicProperty::Instantiate( "Broadcast RGB",          this, *rawProperties );
			m_Props.Max_bpc                  = CDRMAtomicProperty::Instantiate( "max bpc",                this, *rawProperties );
		}

		ParseEDID();

		if ( m_Mutable.EdidData != oldEdid )
		{
			m_Mutable.bEdidChanged = true;
		}
	}

	int CDRMConnector::Present( const FrameInfo_t *pFrameInfo, bool bAsync )
	{
		return HackyDRMPresent( pFrameInfo, bAsync );
	}

	void CDRMConnector::UpdateEffectiveOrientation( const drmModeModeInfo *pMode )
	{
		auto setEffectiveOrientation = [this]( GamescopePanelOrientation eOrientation )
		{
			m_ChosenOrientation = eOrientation;
		};

		if ( this->GetScreenType() == GAMESCOPE_SCREEN_TYPE_INTERNAL && g_DesiredInternalOrientation != GAMESCOPE_PANEL_ORIENTATION_AUTO )
		{
			setEffectiveOrientation( g_DesiredInternalOrientation );
		}
		else
		{
			if ( this->GetProperties().panel_orientation )
			{
				switch ( this->GetProperties().panel_orientation->GetCurrentValue() )
				{
					case DRM_MODE_PANEL_ORIENTATION_NORMAL:
						setEffectiveOrientation( GAMESCOPE_PANEL_ORIENTATION_0 );
						return;
					case DRM_MODE_PANEL_ORIENTATION_BOTTOM_UP:
						setEffectiveOrientation( GAMESCOPE_PANEL_ORIENTATION_180 );
						return;
					case DRM_MODE_PANEL_ORIENTATION_LEFT_UP:
						setEffectiveOrientation( GAMESCOPE_PANEL_ORIENTATION_90 );
						return;
					case DRM_MODE_PANEL_ORIENTATION_RIGHT_UP:
						setEffectiveOrientation( GAMESCOPE_PANEL_ORIENTATION_270 );
						return;
					default:
						break;
				}
			}

			if ( this->GetScreenType() == gamescope::GAMESCOPE_SCREEN_TYPE_INTERNAL && pMode )
			{
				// Auto-detect portait mode for internal displays
				setEffectiveOrientation( pMode->hdisplay < pMode->vdisplay
					? GAMESCOPE_PANEL_ORIENTATION_270
					: GAMESCOPE_PANEL_ORIENTATION_0 );
			}
			else
			{
				setEffectiveOrientation( GAMESCOPE_PANEL_ORIENTATION_0 );
			}
		}
	}

	void CDRMConnector::ParseEDID()
	{
		if ( !GetProperties().EDID )
			return;

		uint64_t ulBlobId = GetProperties().EDID->GetCurrentValue();
		if ( !ulBlobId )
			return;

		drmModePropertyBlobRes *pBlob = drmModeGetPropertyBlob( g_DRM.fd, ulBlobId );
		if ( !pBlob )
			return;
		defer( drmModeFreePropertyBlob( pBlob ) );

		const uint8_t *pDataPointer = reinterpret_cast<const uint8_t *>( pBlob->data );
		m_Mutable.EdidData = std::vector<uint8_t>{ pDataPointer, pDataPointer + pBlob->length };

		di_info *pInfo = di_info_parse_edid( m_Mutable.EdidData.data(), m_Mutable.EdidData.size() );
		if ( !pInfo )
		{
			drm_log.errorf( "Failed to parse edid for connector: %s", m_Mutable.szName );
			return;
		}
		defer( di_info_destroy( pInfo ) );

		const di_edid *pEdid = di_info_get_edid( pInfo );

		const di_edid_vendor_product *pProduct = di_edid_get_vendor_product( pEdid );
		m_Mutable.szMakePNP[0] = pProduct->manufacturer[0];
		m_Mutable.szMakePNP[1] = pProduct->manufacturer[1];
		m_Mutable.szMakePNP[2] = pProduct->manufacturer[2];
		m_Mutable.szMakePNP[3] = '\0';

		m_Mutable.pszMake = m_Mutable.szMakePNP;
		auto pnpIter = pnps.find( m_Mutable.szMakePNP );
		if ( pnpIter != pnps.end() )
			m_Mutable.pszMake = pnpIter->second.c_str();

		const di_edid_display_descriptor *const *pDescriptors = di_edid_get_display_descriptors( pEdid );
		for ( size_t i = 0; pDescriptors[i] != nullptr; i++ )
		{
			const di_edid_display_descriptor *pDesc = pDescriptors[i];
			const di_edid_display_descriptor_tag eTag = di_edid_display_descriptor_get_tag( pDesc );
			if ( eTag == DI_EDID_DISPLAY_DESCRIPTOR_PRODUCT_NAME )
			{
				// Max length of di_edid_display_descriptor_get_string is 14
				// m_szModel is 16 bytes.
				const char *pszModel = di_edid_display_descriptor_get_string( pDesc );
				strncpy( m_Mutable.szModel, pszModel, sizeof( m_Mutable.szModel ) );
			}
			else if ( eTag == DI_EDID_DISPLAY_DESCRIPTOR_DATA_STRING )
			{
				const char *pszDataString = di_edid_display_descriptor_get_string( pDesc );
				snprintf( m_Mutable.szDataString, sizeof( m_Mutable.szDataString ), "%s", pszDataString );
			}
		}

		drm_log.infof("Connector %s -> %s - %s", m_Mutable.szName, m_Mutable.szMakePNP, m_Mutable.szModel );

		bool bHasKnownColorimetry = false;
		bool bHasKnownHDRInfo = false;
		sol::optional<float> ofScriptMaxCLL, ofScriptMaxFALL, ofScriptMinCLL;

		m_Mutable.ValidDynamicRefreshRates.clear();
		m_Mutable.fnDynamicModeGenerator = nullptr;
		{
			CScriptScopedLock script;

			auto oKnownDisplay = script.Manager().Gamescope().Config.LookupDisplay( script, m_Mutable.szMakePNP, pProduct->product, m_Mutable.szModel, m_Mutable.szDataString );
			if ( oKnownDisplay )
			{
				sol::table tTable = oKnownDisplay->second;

				std::string_view psvPrettyName = tTable.get_or( "pretty_name", std::string_view{ "Untitled Display" } );
				drm_log.infof( "Got known display: %.*s (%.*s)",
					(int)oKnownDisplay->first.size(), oKnownDisplay->first.data(),
					(int)psvPrettyName.size(), psvPrettyName.data() );


				sol::optional<sol::table> otDynamicRefreshRates = tTable["dynamic_refresh_rates"];
				sol::optional<sol::function> ofnDynamicModegen = tTable["dynamic_modegen"];

				if ( otDynamicRefreshRates && !ofnDynamicModegen )
					m_Mutable.ValidDynamicRefreshRates = TableToVector<uint32_t>( *otDynamicRefreshRates );
				else if ( otDynamicRefreshRates && ofnDynamicModegen )
				{
					m_Mutable.ValidDynamicRefreshRates = TableToVector<uint32_t>( *otDynamicRefreshRates );

					m_Mutable.fnDynamicModeGenerator = [ fnDynamicModegen = *ofnDynamicModegen ]( const drmModeModeInfo *pBaseMode, int nRefreshHz ) -> drmModeModeInfo
					{
						CScriptScopedLock script;

						sol::table tInMode = script->create_table();
						tInMode["clock"] = pBaseMode->clock;

						tInMode["hdisplay"] = pBaseMode->hdisplay;
						tInMode["hsync_start"] = pBaseMode->hsync_start;
						tInMode["hsync_end"] = pBaseMode->hsync_end;
						tInMode["htotal"] = pBaseMode->htotal;

						tInMode["vdisplay"] = pBaseMode->vdisplay;
						tInMode["vsync_start"] = pBaseMode->vsync_start;
						tInMode["vsync_end"] = pBaseMode->vsync_end;
						tInMode["vtotal"] = pBaseMode->vtotal;

						tInMode["vrefresh"] = pBaseMode->vrefresh;

						sol::function_result ret = fnDynamicModegen(tInMode, nRefreshHz);
						if ( !ret.valid() || !ret.get<sol::table>() )
							return *pBaseMode;
						
						sol::table tOutMode = ret;

						drmModeModeInfo outMode = *pBaseMode;
						outMode.clock = tOutMode["clock"];

						outMode.hdisplay = tOutMode["hdisplay"];
						outMode.hsync_start = tOutMode["hsync_start"];
						outMode.hsync_end = tOutMode["hsync_end"];
						outMode.htotal = tOutMode["htotal"];

						outMode.vdisplay = tOutMode["vdisplay"];
						outMode.vsync_start = tOutMode["vsync_start"];
						outMode.vsync_end = tOutMode["vsync_end"];
						outMode.vtotal = tOutMode["vtotal"];

						outMode.vrefresh = tOutMode["vrefresh"];

						snprintf( outMode.name, sizeof( outMode.name ), "%dx%d@%d.00", outMode.hdisplay, outMode.vdisplay, nRefreshHz );

						return outMode;
					};
				}

				if ( sol::optional<sol::table> otColorimetry = tTable["colorimetry"] )
				{
					sol::table tColorimetry = *otColorimetry;

					// TODO: Add a vec2 + colorimetry type?
					sol::optional<sol::table> otR = tColorimetry["r"];
					sol::optional<sol::table> otG = tColorimetry["g"];
					sol::optional<sol::table> otB = tColorimetry["b"];
					sol::optional<sol::table> otW = tColorimetry["w"];

					if ( otR && otG && otB && otW )
					{
						m_Mutable.DisplayColorimetry.primaries.r = TableToVec<glm::vec2>( *otR );
						m_Mutable.DisplayColorimetry.primaries.g = TableToVec<glm::vec2>( *otG );
						m_Mutable.DisplayColorimetry.primaries.b = TableToVec<glm::vec2>( *otB );
						m_Mutable.DisplayColorimetry.white = TableToVec<glm::vec2>( *otW );

						bHasKnownColorimetry = true;
					}
				}

				if ( sol::optional<sol::table> otHDRInfo = tTable["hdr"] )
				{
					m_Mutable.HDR.bExposeHDRSupport = otHDRInfo->get_or( "supported", false );
					m_Mutable.HDR.eOutputEncodingEOTF = otHDRInfo->get_or( "eotf", EOTF_Gamma22 );

					ofScriptMaxCLL = (*otHDRInfo)["max_content_light_level"].get<sol::optional<float>>();
					ofScriptMaxFALL = (*otHDRInfo)["max_frame_average_luminance"].get<sol::optional<float>>();
					ofScriptMinCLL = (*otHDRInfo)["min_content_light_level"].get<sol::optional<float>>();

					bHasKnownHDRInfo = true;
				}
			}
			else
			{
				if ( g_customRefreshRates.size() > 0 && GetScreenType() == GAMESCOPE_SCREEN_TYPE_INTERNAL )
					m_Mutable.ValidDynamicRefreshRates = g_customRefreshRates;
				// Unknown display, see if there are any other refresh rates in the EDID we can get.
				else if ( GetScreenType() == GAMESCOPE_SCREEN_TYPE_INTERNAL  || cv_drm_allow_dynamic_modes_for_external_display )
				{
					const drmModeModeInfo *pPreferredMode = find_mode( m_pConnector.get(), 0, 0, 0 );

					if ( pPreferredMode )
					{
						// See if the EDID has any modes for us.
						for (int i = 0; i < m_pConnector->count_modes; i++)
						{
							const drmModeModeInfo *pMode = &m_pConnector->modes[i];

							if ( pMode->hdisplay != pPreferredMode->hdisplay || pMode->vdisplay != pPreferredMode->vdisplay )
								continue;

							
							if ( !Algorithm::Contains( m_Mutable.ValidDynamicRefreshRates, pMode->vrefresh ) )
							{
								m_Mutable.ValidDynamicRefreshRates.push_back( pMode->vrefresh );
							}
						}

						std::sort( m_Mutable.ValidDynamicRefreshRates.begin(), m_Mutable.ValidDynamicRefreshRates.end() );
					}
				}
			}
		}

		if ( !bHasKnownColorimetry )
		{
			// Steam Deck OLED has calibrated chromaticity coordinates in the EDID
			// for each unit.
			// Other external displays probably have this too.

			const di_edid_chromaticity_coords *pChroma = di_edid_get_chromaticity_coords( pEdid );
			if ( pChroma && pChroma->red_x != 0.0f )
			{
				drm_log.infof( "[colorimetry]: EDID with colorimetry detected. Using it" );
				m_Mutable.DisplayColorimetry = displaycolorimetry_t
				{
					.primaries = { { pChroma->red_x, pChroma->red_y }, { pChroma->green_x, pChroma->green_y }, { pChroma->blue_x, pChroma->blue_y } },
					.white = { pChroma->white_x, pChroma->white_y },
				};
			}
			else
			{
				// Assume 709 if we have no data at all.
				m_Mutable.DisplayColorimetry = displaycolorimetry_709;
			}
		}

		drm_log.infof( "[colorimetry]: r %f %f", m_Mutable.DisplayColorimetry.primaries.r.x, m_Mutable.DisplayColorimetry.primaries.r.y );
		drm_log.infof( "[colorimetry]: g %f %f", m_Mutable.DisplayColorimetry.primaries.g.x, m_Mutable.DisplayColorimetry.primaries.g.y );
		drm_log.infof( "[colorimetry]: b %f %f", m_Mutable.DisplayColorimetry.primaries.b.x, m_Mutable.DisplayColorimetry.primaries.b.y );
		drm_log.infof( "[colorimetry]: w %f %f", m_Mutable.DisplayColorimetry.white.x, m_Mutable.DisplayColorimetry.white.y );

		/////////////////////
		// Parse HDR stuff.
		/////////////////////
		if ( !bHasKnownHDRInfo || !ofScriptMaxCLL || !ofScriptMaxFALL || !ofScriptMinCLL )
		{
			// libdisplay-info 0.4.0 also finds CTA-861 blocks in DisplayID 2 extensions.
			const di_hdr_static_metadata *pHDRStaticMetadata = di_info_get_hdr_static_metadata( pInfo );
			const di_supported_signal_colorimetry *pColorimetry = di_info_get_supported_signal_colorimetry( pInfo );

			if ( pColorimetry->bt2020_rgb && pHDRStaticMetadata->pq )
			{
				if ( !bHasKnownHDRInfo )
				{
				m_Mutable.HDR.bExposeHDRSupport = true;
				if (disableInternalPq && GetScreenType() == GAMESCOPE_SCREEN_TYPE_INTERNAL)
					// Current handheld internal displays have issues
					// with PQ, e.g., Ayaneo 3, Steam Deck etc.
					// Use Gamma 2.2 as the safest option for now.
					m_Mutable.HDR.eOutputEncodingEOTF = EOTF_Gamma22;
				else
					m_Mutable.HDR.eOutputEncodingEOTF = EOTF_PQ;
				}
				m_Mutable.HDR.uMaxContentLightLevel =
					pHDRStaticMetadata->desired_content_max_luminance
					? nits_to_u16( pHDRStaticMetadata->desired_content_max_luminance )
					: nits_to_u16( 1499.0f );
				m_Mutable.HDR.uMaxFrameAverageLuminance =
					pHDRStaticMetadata->desired_content_max_frame_avg_luminance
					? nits_to_u16( pHDRStaticMetadata->desired_content_max_frame_avg_luminance )
					: nits_to_u16( std::min( 799.f, nits_from_u16( m_Mutable.HDR.uMaxContentLightLevel ) ) );
				m_Mutable.HDR.uMinContentLightLevel =
					pHDRStaticMetadata->desired_content_min_luminance
					? nits_to_u16_dark( pHDRStaticMetadata->desired_content_min_luminance )
					: nits_to_u16_dark( 0.0f );
			}
			else if ( !bHasKnownHDRInfo )
			{
				m_Mutable.HDR.bExposeHDRSupport = false;
			}
		}

		// Script values win over the EDID field by field.
		if ( ofScriptMaxCLL )
			m_Mutable.HDR.uMaxContentLightLevel = nits_to_u16( *ofScriptMaxCLL );
		if ( ofScriptMaxFALL )
			m_Mutable.HDR.uMaxFrameAverageLuminance = nits_to_u16( *ofScriptMaxFALL );
		if ( ofScriptMinCLL )
			m_Mutable.HDR.uMinContentLightLevel = nits_to_u16_dark( *ofScriptMinCLL );

		// Generate a default HDR10 infoframe.
		if ( m_Mutable.HDR.IsHDR10() )
		{
			hdr_output_metadata defaultHDRMetadata{};
			hdr_metadata_infoframe *pInfoframe = &defaultHDRMetadata.hdmi_metadata_type1;

			// The mastering display and content light levels describe the source, not
			// this connector. Leave them zero (unknown) until the app supplies them.
			pInfoframe->eotf = HDMI_EOTF_ST2084;
			pInfoframe->metadata_type = 0;

			m_Mutable.HDR.pDefaultMetadataBlob = GetBackend()->CreateBackendBlob( defaultHDRMetadata );
		}
	}

	/////////////////////////
	// CDRMFb
	/////////////////////////
	CDRMFb::CDRMFb( uint32_t uFbId )
		: m_uFbId{ uFbId }
	{

	}
	CDRMFb::~CDRMFb()
	{
		// I own the fbid.
		if ( drmModeRmFB( g_DRM.fd, m_uFbId ) != 0 )
			drm_log.errorf_errno( "drmModeRmFB failed" );
		m_uFbId = 0;
	}
}

#if defined(DRM_CLIENT_CAP_PLANE_COLOR_PIPELINE) && defined(DRM_MODE_OBJECT_COLOROP)
static drm_color::Properties drm_color_properties(int fd, uint32_t object, uint32_t type)
{
	drm_color::Properties result;
	drmModeObjectProperties *properties = drmModeObjectGetProperties(fd, object, type);
	if (!properties) return result;
	defer(drmModeFreeObjectProperties(properties));
	for (uint32_t i = 0; i < properties->count_props; i++)
	{
		drmModePropertyRes *property = drmModeGetProperty(fd, properties->props[i]);
		if (!property) continue;
		defer(drmModeFreeProperty(property));
		drm_color::Property value{property->prop_id, properties->prop_values[i],
			bool(property->flags & DRM_MODE_PROP_IMMUTABLE), {}};
		for (int j = 0; j < property->count_enums; j++)
			value.enums[property->enums[j].name] = property->enums[j].value;
		result[property->name] = std::move(value);
	}
	return result;
}
#endif

static void drm_init_generic_color(struct drm_t *drm)
{
	// Choose the private API before enabling the mutually exclusive client cap.
	for (const auto &plane : drm->planes)
	{
		const auto &p = plane->GetProperties();
		if (p.type && p.type->GetCurrentValue() == DRM_PLANE_TYPE_PRIMARY &&
			p.AMD_PLANE_CTM && p.AMD_PLANE_DEGAMMA_TF && p.AMD_PLANE_SHAPER_TF &&
			p.AMD_PLANE_SHAPER_LUT && p.AMD_PLANE_LUT3D && !g_bForceDisableColorMgmt)
		{
			drm_log.infof("Color API: AMD private (generic CRTC available for incompatible outputs)");
			return;
		}
	}
#if defined(DRM_CLIENT_CAP_PLANE_COLOR_PIPELINE) && defined(DRM_MODE_OBJECT_COLOROP)
	if (!g_bForceDisableColorMgmt && drmSetClientCap(drm->fd, DRM_CLIENT_CAP_PLANE_COLOR_PIPELINE, 1) == 0)
	{
		drm->generic_plane_color = true;
		for (const auto &plane : drm->planes)
		{
			plane->RefreshState();
			auto properties = drm_color_properties(drm->fd, plane->GetObjectId(), DRM_MODE_OBJECT_PLANE);
			auto roots = properties.find("COLOR_PIPELINE");
			if (roots == properties.end()) continue;
			for (const auto &[name, root] : roots->second.enums)
			{
				if (!root || root > UINT32_MAX) continue;
				drm_color::Pipeline pipeline{plane->GetObjectId(), root, {}};
				std::unordered_set<uint32_t> visited;
				uint32_t id = root;
				while (id && visited.insert(id).second && visited.size() <= 256)
				{
					auto props = drm_color_properties(drm->fd, id, DRM_MODE_OBJECT_COLOROP);
					auto type = props.find("TYPE"), next = props.find("NEXT");
					if (type == props.end() || next == props.end() || next->second.value > UINT32_MAX) break;
					std::string typeName;
					for (const auto &[label, value] : type->second.enums)
						if (value == type->second.value) typeName = label;
					const uint32_t operationId = id;
					id = next->second.value;
					pipeline.operations.push_back({operationId, typeName, std::move(props)});
				}
				if (drm_color::ValidPipeline(pipeline))
					drm->color_pipelines[plane->GetObjectId()].push_back(std::move(pipeline));
			}
		}
		if (drm->color_pipelines.empty())
		{
			drmSetClientCap(drm->fd, DRM_CLIENT_CAP_PLANE_COLOR_PIPELINE, 0);
			drm->generic_plane_color = false;
		}
	}
#endif
	drm_log.infof("Color API: %s; frame eligibility is checked atomically",
		drm->generic_plane_color ? "generic plane pipelines + CRTC" : "generic CRTC");
	if (drm->is_nvidia)
		drm_log.infof("NVIDIA retains forced composition and Vulkan color management");
}

static std::shared_ptr<gamescope::BackendBlob> drm_color_blob(std::span<const uint8_t> data)
{
	return GetBackend()->CreateBackendBlob(typeid(drm_color::Program), data);
}

static bool drm_upload_color_program(DRMColorCandidate &candidate)
{
	for (auto &write : candidate.program)
	{
		if (write.blob.empty()) continue;
		auto blob = drm_color_blob(write.blob);
		if (!blob) return false;
		write.value = blob->GetBlobValue();
		candidate.blobs.push_back(std::move(blob));
	}
	return true;
}

static bool drm_add_crtc_color(DRMColorCandidate &candidate, gamescope::CDRMCRTC *crtc,
	const drm_color::CrtcProgram &program)
{
	const auto &props = crtc->GetProperties();
	const std::vector<uint8_t> *blobs[] = {&program.degamma, &program.ctm, &program.gamma};
	const std::optional<gamescope::CDRMAtomicProperty> *properties[] = {&props.DEGAMMA_LUT, &props.CTM, &props.GAMMA_LUT};
	for (int i = 0; i < 3; i++)
	{
		if (!*properties[i])
		{
			if (!blobs[i]->empty()) return false;
			continue;
		}
		candidate.program.push_back({crtc->GetObjectId(), (*properties[i])->GetPropertyId(), 0, *blobs[i]});
	}
	return true;
}

static bool drm_single_opaque_color_layer(const FrameInfo_t *frame)
{
	if (frame->layers.count() != 1) return false;
	const auto &layer = frame->layers[0];
	return layer.opacity == 1.f && (!layer.hasAlpha() || layer.eAlphaBlendingMode == ALPHA_BLENDING_MODE_NONE) &&
		layer.offset.x == 0.f && layer.offset.y == 0.f &&
		layer.integerWidth() == uint32_t(g_nOutputWidth) && layer.integerHeight() == uint32_t(g_nOutputHeight);
}

static int drm_prepare_generic_color(struct drm_t *drm, const FrameInfo_t *frame,
	const std::function<int()> &apply)
{
	using namespace drm_color;
	auto reject = [&](const char *reason)
	{
		if (drm->color_failure != reason)
		{
			drm_log.debugf("Color scanout requires composition: %s", reason);
			drm->color_failure = reason;
		}
		return -EINVAL;
	};
	if (!drm->pCRTC) return -EINVAL;
	if (drm->color_cache_serial != g_ColorMgmt.serial || drm->needs_modeset ||
		drm->color_program_cache.size() > 64 || drm->color_failed_frames.size() > 256)
	{
		drm->color_program_cache.clear();
		drm->color_output_cache.clear();
		drm->color_failed_frames.clear();
		drm->color_cache_serial = g_ColorMgmt.serial;
	}
	const bool blend = frame->applyOutputColorMgmt && !frame->blendInDisplayEncoding &&
		!drm_single_opaque_color_layer(frame);
	const int count = frame->layers.count();
	const auto geometry = FrameInfoToLiftoffStateCacheEntry(drm, frame);
	std::vector<uint64_t> frameKey{uint64_t(count), uint64_t(drm->flags)};
	std::vector<const std::vector<DRMColorCandidate> *> candidates;
	std::vector<Stage> outputStages;
	if (blend)
		outputStages.push_back(Transfer(frame->outputEncodingEOTF == EOTF_PQ ? Curve::PQNormalizedEncode : Curve::SRGBEncode));
	const auto &crtcProps = drm->pCRTC->GetProperties();
	std::vector<uint64_t> outputKey{drm->pCRTC->GetObjectId(), uint64_t(blend), uint64_t(frame->outputEncodingEOTF),
		drm->pCRTC->GetDegammaSize(), uint64_t(bool(crtcProps.CTM)), drm->pCRTC->GetGammaSize()};
	auto [outputEntry, newOutput] = drm->color_output_cache.try_emplace(outputKey);
	if (newOutput)
		outputEntry->second = CompileCRTC(outputStages, drm->pCRTC->GetDegammaSize(), bool(crtcProps.CTM), drm->pCRTC->GetGammaSize());
	const auto &output = outputEntry->second;
	if (!output) return reject("CRTC cannot encode the shared linear-light blend space at the required precision");

	for (int i = 0; i < k_nMaxLayers; i++)
		liftoff_layer_unset_property(drm->lo_layers[i], "COLOR_PIPELINE");
	for (const auto &plane : drm->planes)
		if (drm->generic_plane_color && plane->GetProperties().COLOR_PIPELINE)
			plane->GetProperties().COLOR_PIPELINE->SetPendingValue(drm->req, 0, true);

	for (int i = 0; i < count; i++)
	{
		const auto &layer = frame->layers[i];
		Transform transform;
		transform.input = layer.colorspace == GAMESCOPE_APP_TEXTURE_COLORSPACE_SCRGB ? Encoding::ScRGB :
			layer.colorspace == GAMESCOPE_APP_TEXTURE_COLORSPACE_HDR10_PQ ? Encoding::PQ :
			layer.colorspace == GAMESCOPE_APP_TEXTURE_COLORSPACE_PASSTHRU ? Encoding::Passthrough : Encoding::SDR;
		transform.output = frame->outputEncodingEOTF;
		transform.managed = frame->applyOutputColorMgmt && layer.applyColorMgmt && !layer.outputColorMgmtApplied;
		transform.blend = blend;
		if (transform.managed && g_bForceDisableColorMgmt) return reject("hardware color management is disabled");
		const auto &luts = g_ColorMgmtLuts[ColorSpaceToEOTFIndex(layer.colorspace)];
		if (transform.managed)
		{
			if (!luts.HasLuts()) return reject("color LUTs are unavailable");
			transform.shaper = luts.lut1d;
			transform.cube = luts.lut3d;
			transform.analytic = luts.matrixTransform;
		}
		if (layer.ctm && !cv_drm_debug_disable_ctm)
		{
			const auto &matrix = layer.ctm->View<glm::mat3x4>();
			Matrix converted;
			for (int row = 0; row < 3; row++)
				for (int col = 0; col < 4; col++) converted[4 * row + col] = matrix[row][col];
			if (converted != Identity()) transform.ctm = converted;
		}
		std::vector<uint64_t> key{uint64_t(drm->pCRTC->GetObjectId()), uint64_t(layer.colorspace),
			uint64_t(frame->outputEncodingEOTF), uint64_t(transform.managed), uint64_t(blend), uint64_t(layer.isYcbcr()),
			uint64_t(drm_single_opaque_color_layer(frame)), uint64_t(g_bForceDisableColorMgmt)};
		if (transform.ctm)
			for (double value : *transform.ctm) { uint64_t bits; std::memcpy(&bits, &value, sizeof(bits)); key.push_back(bits); }
		const auto &entry = geometry;
		key.push_back(entry.layerState[i].colorEncoding);
		key.push_back(entry.layerState[i].colorRange);
		frameKey.push_back(key.size());
		frameKey.insert(frameKey.end(), key.begin(), key.end());
		const auto &g = geometry.layerState[i];
		frameKey.insert(frameKey.end(), {g.zpos, g.srcW, g.srcH, g.crtcX, g.crtcY, g.crtcW, g.crtcH,
			g.opacity, uint64_t(g.eAlphaBlendingMode), layer.tex->drmFormat(), layer.tex->dmabuf().modifier});
		auto [cache, inserted] = drm->color_program_cache.try_emplace(key);
		if (inserted)
		{
			auto variants = BuildVariants(transform);
			const bool yuv = drm->generic_plane_color && layer.isYcbcr();
			for (auto &[plane, pipelines] : drm->color_pipelines)
			{
				auto physical = std::find_if(drm->planes.begin(), drm->planes.end(),
					[&](const auto &p) { return p->GetObjectId() == plane; });
				if (physical == drm->planes.end() || !((*physical)->GetModePlane()->possible_crtcs & drm->pCRTC->GetCRTCMask())) continue;
				for (const auto &pipeline : pipelines)
					for (auto stages : variants)
					{
						if (yuv)
						{
							Stage conversion;
							conversion.type = Stage::Type::FixedMatrix;
							const auto encoding = entry.layerState[i].colorEncoding;
							conversion.fixedMatrix = std::string("YCbCr ") +
								(encoding == DRM_COLOR_YCBCR_BT601 ? "601" : encoding == DRM_COLOR_YCBCR_BT2020 ? "2020 NC" : "709") +
								(entry.layerState[i].colorRange == DRM_COLOR_YCBCR_FULL_RANGE ? " Full to RGB" : " Limited to RGB");
							stages.insert(stages.begin(), std::move(conversion));
						}
						auto program = Compile(pipeline, stages);
						DRMColorCandidate candidate{plane, pipeline.root, {}, {}};
						if (program)
						{
							candidate.program = std::move(*program);
							if (drm_add_crtc_color(candidate, drm->pCRTC, *output) && drm_upload_color_program(candidate))
								cache->second.push_back(std::move(candidate));
						}
						// A single opaque plane may use the CRTC for its final curve.
						else if (!blend && drm_single_opaque_color_layer(frame) && !stages.empty() && stages.back().type == Stage::Type::Curve)
						{
							auto tail = CompileCRTC(std::span<const Stage>(&stages.back(), 1), 0, false, drm->pCRTC->GetGammaSize());
							stages.pop_back();
							program = Compile(pipeline, stages);
							if (tail && program)
							{
								candidate.program = std::move(*program);
								if (drm_add_crtc_color(candidate, drm->pCRTC, *tail) && drm_upload_color_program(candidate))
								cache->second.push_back(std::move(candidate));
							}
						}
					}
			}
			// CRTC-only color is safe only for one opaque, screen-filling RGB
			// plane. It cannot replace independent transforms before blending.
			if (!layer.isYcbcr() || !drm->generic_plane_color)
				for (const auto &stages : variants)
				{
					if (!stages.empty() && (!drm_single_opaque_color_layer(frame) || transform.input == Encoding::ScRGB)) continue;
					auto crtc = CompileCRTC(stages, drm->pCRTC->GetDegammaSize(), bool(crtcProps.CTM), drm->pCRTC->GetGammaSize());
					DRMColorCandidate candidate;
					if (crtc && drm_add_crtc_color(candidate, drm->pCRTC, *crtc) && drm_upload_color_program(candidate))
						cache->second.push_back(std::move(candidate));
				}
		}
		if (cache->second.empty()) return reject(layer.isYcbcr() ?
			"no pipeline can preserve YCbCr conversion and the requested color stages" :
			"available LUT sizes/precision, matrices, or transfer functions cannot represent the complete layer transform");
		candidates.push_back(&cache->second);
	}
	if (is_liftoff_caching_enabled() && drm->color_failed_frames.contains(frameKey))
		return -EINVAL;
	// Every root enum value belongs to its physical plane. Normal libliftoff
	// enum validation therefore enforces the mapping, without a private hook.
	const int cursor = drmModeAtomicGetCursor(drm->req);
	std::unordered_set<uint32_t> used;
	std::vector<const DRMColorCandidate *> selection;
	int result = -EINVAL;
	auto assign = [&](auto &&self, int layer) -> bool
	{
		if (layer < count)
		{
			const uint64_t preferred = drm->color_last_roots.size() == size_t(count) ? drm->color_last_roots[layer] : UINT64_MAX;
			for (int pass = 0; pass < 2; pass++)
			for (const auto &candidate : *candidates[layer])
			{
				if ((candidate.root == preferred) != (pass == 0)) continue;
				if (candidate.plane && used.contains(candidate.plane)) continue;
				if (candidate.plane) used.insert(candidate.plane);
				selection.push_back(&candidate);
				if (self(self, layer + 1)) return true;
				selection.pop_back();
				used.erase(candidate.plane);
			}
			return false;
		}
		drmModeAtomicSetCursor(drm->req, cursor);
		drm->pending.generic_color_blobs.clear();
		for (int i = 0; i < count; i++)
		{
			const auto &candidate = *selection[i];
			if (drm->generic_plane_color)
				liftoff_layer_set_property(drm->lo_layers[i], "COLOR_PIPELINE", candidate.root);
			for (const auto &write : candidate.program)
				if (drmModeAtomicAddProperty(drm->req, write.object, write.property, write.value) < 0) return false;
			drm->pending.generic_color_blobs.insert(drm->pending.generic_color_blobs.end(), candidate.blobs.begin(), candidate.blobs.end());
		}
		result = apply();
		if (result == 0 && liftoff_output_needs_composition(drm->lo_output)) result = -EINVAL;
		// Stop on session loss or permission errors; the caller handles its
		// existing fence workaround and VT switch behavior.
		return result == 0 || result == -EACCES || result == -EPERM;
	};
	assign(assign, 0);
	if (result == 0)
	{
		// Only publish pending property values after libliftoff has accepted
		// the complete mapping. Candidate tests must not update these trackers.
		for (int i = 0; i < count; i++)
		{
			const auto &candidate = *selection[i];
			if (drm->generic_plane_color)
			{
				auto *mapped = liftoff_layer_get_plane(drm->lo_layers[i]);
				if (!mapped) return -EINVAL;
				const uint32_t planeId = liftoff_plane_get_id(mapped);
				for (const auto &plane : drm->planes)
					if (plane->GetObjectId() == planeId && plane->GetProperties().COLOR_PIPELINE &&
						plane->GetProperties().COLOR_PIPELINE->SetPendingValue(drm->req, candidate.root, false) < 0)
						return -ENOMEM;
			}
			auto &properties = drm->pCRTC->GetProperties();
			std::optional<gamescope::CDRMAtomicProperty> *tracked[] = {&properties.DEGAMMA_LUT, &properties.CTM, &properties.GAMMA_LUT};
			for (const auto &write : candidate.program)
				if (write.object == drm->pCRTC->GetObjectId())
					for (auto *property : tracked)
						if (*property && (*property)->GetPropertyId() == write.property &&
							(*property)->SetPendingValue(drm->req, write.value, false) < 0)
							return -ENOMEM;
		}
		drm->color_last_roots.clear();
		for (const auto *candidate : selection) drm->color_last_roots.push_back(candidate->root);
		return 0;
	}
	if (result == -EACCES || result == -EPERM) return result;
	if (result == -EINVAL && is_liftoff_caching_enabled()) drm->color_failed_frames.insert(std::move(frameKey));
	drmModeAtomicSetCursor(drm->req, cursor);
	return reject("atomic testing rejected all compatible color pipeline mappings");
}

static int
drm_prepare_liftoff( struct drm_t *drm, const struct FrameInfo_t *frameInfo, bool needs_modeset )
{
	auto entry = FrameInfoToLiftoffStateCacheEntry( drm, frameInfo );

	// If we are modesetting, reset the state cache, we might
	// move to another CRTC or whatever which might have differing caps.
	// (same with different modes)
	if (needs_modeset || drm->color_cache_serial != g_ColorMgmt.serial)
	{
		g_LiftoffStateCache.clear();
		drm->color_program_cache.clear();
		drm->color_output_cache.clear();
		drm->color_failed_frames.clear();
		drm->color_last_roots.clear();
		drm->color_cache_serial = g_ColorMgmt.serial;
	}

	if (is_liftoff_caching_enabled() && drm_supports_amd_color_mgmt(drm))
	{
		if (g_LiftoffStateCache.count(entry) != 0)
			return -EINVAL;
	}

	bool bSinglePlane = frameInfo->layers.count() < 2 && cv_drm_single_plane_optimizations;

	for ( int i = 0; i < k_nMaxLayers; i++ )
	{
		if ( i < frameInfo->layers.count() )
		{
			const FrameInfo_t::Layer_t *pLayer = &frameInfo->layers.get( i );
			gamescope::CDRMFb *pDrmFb = static_cast<gamescope::CDRMFb *>( (pLayer->tex && pLayer->tex->GetBackendFb()) ? pLayer->tex->GetBackendFb()->EnsureImported() : nullptr );

			if ( pDrmFb == nullptr )
			{
				drm_log.debugf("drm_prepare_liftoff: layer %d has no FB", i );
				return -EINVAL;
			}

			int nFence = -1;
			if (!cv_drm_debug_disable_in_fence_fd)
			{
				if (pLayer->acquireFenceFd >= 0)
					nFence = fcntl(pLayer->acquireFenceFd, F_DUPFD_CLOEXEC, 0);
				else if (pLayer->acquirePoint)
					nFence = pLayer->acquirePoint->CreateSyncFile();
				if (nFence >= 0)
					drm->m_InFenceFdsInRequest.push_back(nFence);
				else if (pLayer->acquireFenceFd >= 0)
					wait_sync_file(pLayer->acquireFenceFd);
				else if (pLayer->acquirePoint)
					pLayer->acquirePoint->Wait();
			}


			liftoff_layer_set_property( drm->lo_layers[ i ], "FB_ID", pDrmFb->GetFbId());
			liftoff_layer_set_property( drm->lo_layers[ i ], "IN_FENCE_FD", nFence );
			drm->m_FbIdsInRequest.emplace_back( pDrmFb );

			liftoff_layer_set_property( drm->lo_layers[ i ], "zpos", entry.layerState[i].zpos );
			liftoff_layer_set_property( drm->lo_layers[ i ], "alpha", frameInfo->layers.get( i ).opacity * 0xffff);

			if ( entry.layerState[i].zpos != g_zposBase )
			{
				liftoff_layer_set_property( drm->lo_layers[ i ], "pixel blend mode", (uint64_t) frameInfo->layers.get( i ).eAlphaBlendingMode );
			}
			else
			{
				liftoff_layer_unset_property( drm->lo_layers[ i ], "pixel blend mode" );
			}

			liftoff_layer_set_property( drm->lo_layers[ i ], "SRC_X", 0);
			liftoff_layer_set_property( drm->lo_layers[ i ], "SRC_Y", 0);
			liftoff_layer_set_property( drm->lo_layers[ i ], "SRC_W", entry.layerState[i].srcW );
			liftoff_layer_set_property( drm->lo_layers[ i ], "SRC_H", entry.layerState[i].srcH );

			uint64_t ulOrientation = DRM_MODE_ROTATE_0;
			switch ( g_uOutputRotation != 0 ? GAMESCOPE_PANEL_ORIENTATION_0 : drm->pConnector->GetCurrentOrientation() )
			{
			default:
			case GAMESCOPE_PANEL_ORIENTATION_0:
				ulOrientation = DRM_MODE_ROTATE_0;
				break;
			case GAMESCOPE_PANEL_ORIENTATION_270:
				ulOrientation = DRM_MODE_ROTATE_270;
				break;
			case GAMESCOPE_PANEL_ORIENTATION_90:
				ulOrientation = DRM_MODE_ROTATE_90;
				break;
			case GAMESCOPE_PANEL_ORIENTATION_180:
				ulOrientation = DRM_MODE_ROTATE_180;
				break;
			}
			liftoff_layer_set_property( drm->lo_layers[ i ], "rotation", ulOrientation );

			liftoff_layer_set_property( drm->lo_layers[ i ], "CRTC_X", entry.layerState[i].crtcX);
			liftoff_layer_set_property( drm->lo_layers[ i ], "CRTC_Y", entry.layerState[i].crtcY);

			liftoff_layer_set_property( drm->lo_layers[ i ], "CRTC_W", entry.layerState[i].crtcW);
			liftoff_layer_set_property( drm->lo_layers[ i ], "CRTC_H", entry.layerState[i].crtcH);

			if ( frameInfo->layers.get( i ).applyColorMgmt )
			{
				bool bYCbCr = entry.layerState[i].ycbcr;

				if ( !drm->generic_plane_color && !cv_drm_debug_disable_color_encoding && bYCbCr )
				{
					liftoff_layer_set_property( drm->lo_layers[ i ], "COLOR_ENCODING", entry.layerState[i].colorEncoding );
				}
				else
				{
					liftoff_layer_unset_property( drm->lo_layers[ i ], "COLOR_ENCODING" );
				}

				if ( !drm->generic_plane_color && !cv_drm_debug_disable_color_range && bYCbCr )
				{
					liftoff_layer_set_property( drm->lo_layers[ i ], "COLOR_RANGE",    entry.layerState[i].colorRange );
				}
				else
				{
					liftoff_layer_unset_property( drm->lo_layers[ i ], "COLOR_RANGE" );
				}

				if ( drm_supports_amd_color_mgmt( drm ) )
				{
					amdgpu_transfer_function degamma_tf = colorspace_to_plane_degamma_tf( entry.layerState[i].colorspace );
					amdgpu_transfer_function shaper_tf = colorspace_to_plane_shaper_tf( entry.layerState[i].colorspace );

					if ( bYCbCr )
					{
						// JoshA: Based on the Steam In-Home Streaming Shader,
						// it looks like Y is actually sRGB, not HDTV G2.4
						//
						// Matching BT709 for degamma -> regamma on shaper TF here
						// is identity and works on YUV NV12 planes to preserve this.
						//
						// Doing LINEAR/DEFAULT here introduces banding so... this is the best way.
						// (sRGB DEGAMMA does NOT work on YUV planes!)
						degamma_tf = AMDGPU_TRANSFER_FUNCTION_BT709_INV_OETF;
						shaper_tf = AMDGPU_TRANSFER_FUNCTION_BT709_OETF;
					}

					bool bUseDegamma = !cv_drm_debug_disable_degamma_tf;
					if ( bUseDegamma )
						liftoff_layer_set_property( drm->lo_layers[ i ], "AMD_PLANE_DEGAMMA_TF", degamma_tf );
					else
						liftoff_layer_set_property( drm->lo_layers[ i ], "AMD_PLANE_DEGAMMA_TF", 0 );

					bool bUseShaperAnd3DLUT = !cv_drm_debug_disable_shaper_and_3dlut;
					if ( bUseShaperAnd3DLUT )
					{
						liftoff_layer_set_property( drm->lo_layers[ i ], "AMD_PLANE_SHAPER_LUT", drm->pending.shaperlut_id[ ColorSpaceToEOTFIndex( entry.layerState[i].colorspace ) ]->GetBlobValue() );
						liftoff_layer_set_property( drm->lo_layers[ i ], "AMD_PLANE_SHAPER_TF", shaper_tf );
						liftoff_layer_set_property( drm->lo_layers[ i ], "AMD_PLANE_LUT3D", drm->pending.lut3d_id[ ColorSpaceToEOTFIndex( entry.layerState[i].colorspace ) ]->GetBlobValue() );
						// Josh: See shaders/colorimetry.h colorspace_blend_tf if you have questions as to why we start doing sRGB for BLEND_TF despite potentially working in Gamma 2.2 space prior.
					}
					else
					{
						liftoff_layer_set_property( drm->lo_layers[ i ], "AMD_PLANE_SHAPER_LUT", 0 );
						liftoff_layer_set_property( drm->lo_layers[ i ], "AMD_PLANE_SHAPER_TF", 0 );
						liftoff_layer_set_property( drm->lo_layers[ i ], "AMD_PLANE_LUT3D", 0 );
					}
				}
			}
			else
			{
				if ( drm_supports_amd_color_mgmt( drm ) )
				{
					liftoff_layer_set_property( drm->lo_layers[ i ], "AMD_PLANE_DEGAMMA_TF", AMDGPU_TRANSFER_FUNCTION_DEFAULT );
					liftoff_layer_set_property( drm->lo_layers[ i ], "AMD_PLANE_SHAPER_LUT", 0 );
					liftoff_layer_set_property( drm->lo_layers[ i ], "AMD_PLANE_SHAPER_TF", 0 );
					liftoff_layer_set_property( drm->lo_layers[ i ], "AMD_PLANE_LUT3D", 0 );
					liftoff_layer_set_property( drm->lo_layers[ i ], "AMD_PLANE_CTM", 0 );
				}
			}

			if  ( drm_supports_amd_color_mgmt(drm) && drm_supports_srgb_to_pq( drm ) )
			{
				if (!cv_drm_debug_disable_blend_tf && !bSinglePlane)
					liftoff_layer_set_property( drm->lo_layers[ i ], "AMD_PLANE_BLEND_TF", drm->pending.output_tf );
				else
					liftoff_layer_set_property( drm->lo_layers[ i ], "AMD_PLANE_BLEND_TF", AMDGPU_TRANSFER_FUNCTION_DEFAULT );	
			}
			if ( drm_supports_amd_color_mgmt( drm ) )
			{
				if (!cv_drm_debug_disable_ctm && frameInfo->layers[i].ctm != nullptr)
					liftoff_layer_set_property( drm->lo_layers[ i ], "AMD_PLANE_CTM", frameInfo->layers[i].ctm->GetBlobValue() );
				else
					liftoff_layer_set_property( drm->lo_layers[ i ], "AMD_PLANE_CTM", 0 );
			}
		}
		else
		{
			liftoff_layer_set_property( drm->lo_layers[ i ], "FB_ID", 0 );
			liftoff_layer_set_property( drm->lo_layers[ i ], "IN_FENCE_FD", -1 );

			liftoff_layer_unset_property( drm->lo_layers[ i ], "COLOR_ENCODING" );
			liftoff_layer_unset_property( drm->lo_layers[ i ], "COLOR_RANGE" );
			liftoff_layer_unset_property( drm->lo_layers[ i ], "pixel blend mode" );

			if ( drm_supports_amd_color_mgmt(drm) && drm_supports_srgb_to_pq( drm ) )
			{
				liftoff_layer_set_property( drm->lo_layers[ i ], "AMD_PLANE_BLEND_TF", AMDGPU_TRANSFER_FUNCTION_DEFAULT );
			}
			if ( drm_supports_amd_color_mgmt( drm ) )
			{
				liftoff_layer_set_property( drm->lo_layers[ i ], "AMD_PLANE_DEGAMMA_TF", AMDGPU_TRANSFER_FUNCTION_DEFAULT );
				liftoff_layer_set_property( drm->lo_layers[ i ], "AMD_PLANE_SHAPER_LUT", 0 );
				liftoff_layer_set_property( drm->lo_layers[ i ], "AMD_PLANE_SHAPER_TF", 0 );
				liftoff_layer_set_property( drm->lo_layers[ i ], "AMD_PLANE_LUT3D", 0 );
				liftoff_layer_set_property( drm->lo_layers[ i ], "AMD_PLANE_CTM", 0 );
			}
		}
	}

	struct liftoff_output_apply_options lo_options = {
		.timeout_ns = std::numeric_limits<int64_t>::max()
	};

	auto apply = [&]() { return liftoff_output_apply( drm->lo_output, drm->req, drm->flags, &lo_options ); };
	int ret = drm_supports_amd_color_mgmt(drm) ? apply() : drm_prepare_generic_color(drm, frameInfo, apply);

	// The NVIDIA 555 series drivers started advertising DRM_CAP_SYNCOBJ, but do
	// not support IN_FENCE_FD. However, there is no way to hide the IN_FENCE_FD
	// property in a DRM-KMS driver, so the driver returns EPERM when an
	// application sets IN_FENCE_FD. To work around this, the first time a
	// commit fails with -EPERM, try it again with the IN_FENCE_FD property
	// reset to its default value. If this succeeds, disable use of the
	// IN_FENCE_FD property.
	static bool attempted_in_fence_fallback = false;
	if ( ret == -EPERM && !attempted_in_fence_fallback && !cv_drm_debug_disable_in_fence_fd )
	{
		attempted_in_fence_fallback = true;
		for ( int i = 0; i < frameInfo->layers.count(); i++ )
		{
			if ( frameInfo->layers[i].acquirePoint )
				frameInfo->layers[i].acquirePoint->Wait();
			if (frameInfo->layers[i].acquireFenceFd >= 0)
				wait_sync_file(frameInfo->layers[i].acquireFenceFd);
			liftoff_layer_set_property( drm->lo_layers[ i ], "IN_FENCE_FD", -1 );
		}

		ret = drm_supports_amd_color_mgmt(drm) ? apply() : drm_prepare_generic_color(drm, frameInfo, apply);

		if ( ret == 0 )
		{
			// IN_FENCE_FD isn't actually supported. Avoid it in the future.
			cv_drm_debug_disable_in_fence_fd  = true;
		}
	}

	if ( ret == 0 )
	{
		// We don't support partial composition yet
		if ( liftoff_output_needs_composition( drm->lo_output ) )
			ret = -EINVAL;
	}

	// If we aren't modesetting and we got -EINVAL, that means that we
	// probably can't do this layout, so add it to our state cache so we don't
	// try it again.
	if (!needs_modeset && drm_supports_amd_color_mgmt(drm))
	{
		if (ret == -EINVAL)
			g_LiftoffStateCache.insert(entry);
	}

	if ( ret == 0 )
		drm_log.debugf( "can drm present %i layers", frameInfo->layers.count() );
	else
		drm_log.debugf( "can NOT drm present %i layers", frameInfo->layers.count() );

	return ret;
}

bool g_bForceAsyncFlips = false;

void drm_rollback( struct drm_t *drm )
{
	drm->pending = drm->current;

	for ( std::unique_ptr< gamescope::CDRMCRTC > &pCRTC : drm->crtcs )
	{
		for ( std::optional<gamescope::CDRMAtomicProperty> &oProperty : pCRTC->GetProperties() )
		{
			if ( oProperty )
				oProperty->Rollback();
		}
	}

	for ( std::unique_ptr< gamescope::CDRMPlane > &pPlane : drm->planes )
	{
		for ( std::optional<gamescope::CDRMAtomicProperty> &oProperty : pPlane->GetProperties() )
		{
			if ( oProperty )
				oProperty->Rollback();
		}
	}

	for ( auto &iter : drm->connectors )
	{
		gamescope::CDRMConnector *pConnector = &iter.second;
		for ( std::optional<gamescope::CDRMAtomicProperty> &oProperty : pConnector->GetProperties() )
		{
			if ( oProperty )
				oProperty->Rollback();
		}
	}
}

/* Unlinks planes a previous DRM master left on CRTCs that no longer have a mode.
 * liftoff zeroes the CRTC_ID of every plane it isn't using, which drags the dead
 * CRTC into our next flip and gets us "requesting event but off". */
static void drm_unlink_foreign_planes( struct drm_t *drm )
{
	drmModeAtomicReq *req = drmModeAtomicAlloc();
	bool bAnyForeign = false;

	const uint32_t uOurCRTCId = drm->pCRTC ? drm->pCRTC->GetObjectId() : 0;
	for ( std::unique_ptr< gamescope::CDRMPlane > &pPlane : drm->planes )
	{
		// Ask the kernel, liftoff moves planes behind our property cache.
		drmModePlane *pKernelPlane = drmModeGetPlane( drm->fd, pPlane->GetObjectId() );
		if ( !pKernelPlane )
			continue;

		const uint32_t uCRTCId = pKernelPlane->crtc_id;
		drmModeFreePlane( pKernelPlane );

		if ( uCRTCId == 0 || uCRTCId == uOurCRTCId )
			continue;

		// A CRTC that still has a mode is either disabled by the modeset below, or
		// would have amdgpu reject us for taking its primary plane away.
		drmModeCrtc *pKernelCRTC = drmModeGetCrtc( drm->fd, uCRTCId );
		const bool bHasMode = !pKernelCRTC || pKernelCRTC->mode_valid;
		drmModeFreeCrtc( pKernelCRTC );

		if ( bHasMode )
			continue;

		drm_log.debugf( "Unlinking plane %u left on foreign CRTC %u", pPlane->GetObjectId(), uCRTCId );

		bAnyForeign = true;
		pPlane->GetProperties().FB_ID->SetPendingValue( req, 0, true );
		pPlane->GetProperties().CRTC_ID->SetPendingValue( req, 0, true );
	}

	if ( bAnyForeign )
	{
		int ret = drmModeAtomicCommit( drm->fd, req, DRM_MODE_ATOMIC_ALLOW_MODESET, nullptr );
		// -EACCES just means we're VT-switched away, our caller handles that.
		if ( ret != 0 && ret != -EACCES )
			drm_log.errorf_errno( "drm_unlink_foreign_planes: commit failed" );
	}

	drmModeAtomicFree( req );
}

/* Prepares an atomic commit for the provided scene-graph. Returns 0 on success,
 * negative errno on failure or if the scene-graph can't be presented directly. */
int drm_prepare( struct drm_t *drm, bool async, const struct FrameInfo_t *frameInfo )
{
	// Keep one KMS flip in flight, while processing clients between submissions.
	uint32_t pending = drm->uPendingFlipCount.load();
	if (pending)
		drm->uPendingFlipCount.wait(pending);

	if ( !drm->pConnector )
		return -EACCES;

	drm_update_color_mgmt(drm);

	const bool bIsVRRCapable = drm->pConnector && drm->pConnector->GetProperties().vrr_capable && !!drm->pConnector->GetProperties().vrr_capable->GetCurrentValue();
	const bool bHasVRREnable = drm->pCRTC && drm->pCRTC->GetProperties().VRR_ENABLED;

	const bool bVRREnabled = bIsVRRCapable && bHasVRREnable && frameInfo->allowVRR;
	if ( bIsVRRCapable )
	{
		if ( bVRREnabled != !!drm->pCRTC->GetProperties().VRR_ENABLED->GetCurrentValue() )
			drm->needs_modeset = true;
	}

	if (drm->pCRTC && drm->pCRTC->GetProperties().ACTIVE->GetCurrentValue() != !frameInfo->dpms)
		drm->needs_modeset = true;

	drm_colorspace uColorimetry = DRM_MODE_COLORIMETRY_DEFAULT;

	const bool bWantsHDR10 = g_bOutputHDREnabled && frameInfo->outputEncodingEOTF == EOTF_PQ;
	gamescope::BackendBlob *pHDRMetadata = nullptr;
	if ( drm->pConnector && drm->pConnector->SupportsHDR10() )
	{
		if ( bWantsHDR10 )
		{
			pHDRMetadata = drm->pConnector->GetHDRInfo().pDefaultMetadataBlob.get();

			wlserver_vk_swapchain_feedback* pFeedback = steamcompmgr_get_base_layer_swapchain_feedback();
			if ( pFeedback && pFeedback->hdr_metadata_blob != nullptr )
				pHDRMetadata = pFeedback->hdr_metadata_blob.get();
			uColorimetry = DRM_MODE_COLORIMETRY_BT2020_RGB;
		}
		else
		{
			pHDRMetadata = drm->sdr_static_metadata.get();
			uColorimetry = DRM_MODE_COLORIMETRY_DEFAULT;
		}

		if ( uColorimetry != drm->pConnector->GetProperties().Colorspace->GetCurrentValue() )
			drm->needs_modeset = true;
	}

	drm->m_FbIdsInRequest.clear();
	drm_close_in_fences( drm );

	bool needs_modeset = drm->needs_modeset.exchange(false);

	assert( drm->req == nullptr );
	drm->req = drmModeAtomicAlloc();

	bool bSinglePlane = frameInfo->layers.count() < 2 && cv_drm_single_plane_optimizations;

	if ( drm_supports_amd_color_mgmt(drm) && drm_supports_srgb_to_pq( drm ) && frameInfo->applyOutputColorMgmt )
	{
		if ( !cv_drm_debug_disable_output_tf && !bSinglePlane )
		{
			drm->pending.output_tf = g_bOutputHDREnabled
				? AMDGPU_TRANSFER_FUNCTION_PQ_EOTF
				: AMDGPU_TRANSFER_FUNCTION_SRGB_EOTF;
		}
		else
		{
			drm->pending.output_tf = AMDGPU_TRANSFER_FUNCTION_DEFAULT;
		}
	}
	else
	{
		drm->pending.output_tf = AMDGPU_TRANSFER_FUNCTION_DEFAULT;
	}

	uint32_t flags = DRM_MODE_ATOMIC_NONBLOCK;

	// We do internal refcounting with these events

	bool bSleep = false;
	if ( drm->pConnector )
	{
		bSleep = cv_drm_sleep_screens[ drm->pConnector->GetScreenType() ];

		bool bCurrentlyAsleep = drm->pConnector->GetProperties().CRTC_ID->GetCurrentValue() == 0;

		if ( bCurrentlyAsleep != bSleep )
			needs_modeset = true;

		// Make screen sleep commits synchronous so we can easily wait for them
		if ( bSleep && bCurrentlyAsleep != bSleep )
			flags &= ~DRM_MODE_ATOMIC_NONBLOCK;
	}

	if ( frameInfo->dpms )
		bSleep = true;

	if ( !bSleep )
	{
		if ( drm->pCRTC != nullptr )
			flags |= DRM_MODE_PAGE_FLIP_EVENT;

		if ( async || g_bForceAsyncFlips )
			flags |= DRM_MODE_PAGE_FLIP_ASYNC;
	}

	bool bForceInRequest = needs_modeset;

	if ( needs_modeset )
	{
		flags |= DRM_MODE_ATOMIC_ALLOW_MODESET;

		drm_unlink_foreign_planes( drm );

		// Disable all connectors and CRTCs

		for ( auto &iter : drm->connectors )
		{
			gamescope::CDRMConnector *pConnector = &iter.second;
			if ( pConnector->GetProperties().CRTC_ID->GetCurrentValue() == 0 )
				continue;

			pConnector->GetProperties().CRTC_ID->SetPendingValue( drm->req, 0, bForceInRequest );

			if ( pConnector->GetProperties().Colorspace )
				pConnector->GetProperties().Colorspace->SetPendingValue( drm->req, 0, bForceInRequest );

			if ( pConnector->GetProperties().HDR_OUTPUT_METADATA )
				pConnector->GetProperties().HDR_OUTPUT_METADATA->SetPendingValue( drm->req, 0, bForceInRequest );

			if ( pConnector->GetProperties().content_type )
				pConnector->GetProperties().content_type->SetPendingValue( drm->req, 0, bForceInRequest );

			if ( pConnector->GetProperties().Broadcast_RGB )
				pConnector->GetProperties().Broadcast_RGB->SetPendingValue( drm->req, 0, bForceInRequest );
		}

		for ( std::unique_ptr< gamescope::CDRMCRTC > &pCRTC : drm->crtcs )
		{
			// We can't disable a CRTC if it's already disabled, or else the
			// kernel will error out with "requesting event but off".
			if ( pCRTC->GetProperties().ACTIVE->GetCurrentValue() == 0 )
				continue;

			pCRTC->GetProperties().ACTIVE->SetPendingValue( drm->req, 0, bForceInRequest );
			pCRTC->GetProperties().MODE_ID->SetPendingValue( drm->req, 0, bForceInRequest );

			if ( pCRTC->GetProperties().GAMMA_LUT )
				pCRTC->GetProperties().GAMMA_LUT->SetPendingValue( drm->req, 0, bForceInRequest );

			if ( pCRTC->GetProperties().DEGAMMA_LUT )
				pCRTC->GetProperties().DEGAMMA_LUT->SetPendingValue( drm->req, 0, bForceInRequest );

			if ( pCRTC->GetProperties().CTM )
				pCRTC->GetProperties().CTM->SetPendingValue( drm->req, 0, bForceInRequest );

			if ( pCRTC->GetProperties().VRR_ENABLED )
				pCRTC->GetProperties().VRR_ENABLED->SetPendingValue( drm->req, 0, bForceInRequest );

			if ( pCRTC->GetProperties().OUT_FENCE_PTR )
				pCRTC->GetProperties().OUT_FENCE_PTR->SetPendingValue( drm->req, 0, bForceInRequest );

			if ( pCRTC->GetProperties().AMD_CRTC_REGAMMA_TF )
				pCRTC->GetProperties().AMD_CRTC_REGAMMA_TF->SetPendingValue( drm->req, 0, bForceInRequest );
		}

		if ( drm->pConnector && !bSleep )
		{
			// Always set our CRTC_ID for the modeset, especially
			// as we zero-ed it above.
			drm->pConnector->GetProperties().CRTC_ID->SetPendingValue( drm->req, drm->pCRTC->GetObjectId(), bForceInRequest );

			if ( drm->pConnector->GetProperties().Colorspace )
				drm->pConnector->GetProperties().Colorspace->SetPendingValue( drm->req, uColorimetry, bForceInRequest );
		}

		if ( drm->pCRTC && !bSleep )
		{
			if ( frameInfo->dpms ) {
				// We can't disable a CRTC if it's already disabled
				if (drm->pCRTC->GetProperties().ACTIVE->GetCurrentValue() != 0)
					drm->pCRTC->GetProperties().ACTIVE->SetPendingValue(drm->req, 0, true);
			}
			else
				drm->pCRTC->GetProperties().ACTIVE->SetPendingValue( drm->req, 1u, true );
			drm->pCRTC->GetProperties().MODE_ID->SetPendingValue( drm->req, drm->pending.mode_id ? drm->pending.mode_id->GetBlobValue() : 0lu, true );

			// Clear color properties inherited from a previous DRM master (i.e. KDE's
			// night light).
			if ( drm->pCRTC->GetProperties().GAMMA_LUT )
				drm->pCRTC->GetProperties().GAMMA_LUT->SetPendingValue( drm->req, 0, true );

			if ( drm->pCRTC->GetProperties().DEGAMMA_LUT )
				drm->pCRTC->GetProperties().DEGAMMA_LUT->SetPendingValue( drm->req, 0, true );

			if ( drm->pCRTC->GetProperties().CTM )
				drm->pCRTC->GetProperties().CTM->SetPendingValue( drm->req, 0, true );

			if ( drm->pCRTC->GetProperties().VRR_ENABLED )
				drm->pCRTC->GetProperties().VRR_ENABLED->SetPendingValue( drm->req, bVRREnabled, true );
		}
	}

	if ( drm->pConnector && !bSleep )
	{
		if ( drm->is_vc4 && drm->pConnector->GetProperties().Max_bpc )
			drm->pConnector->GetProperties().Max_bpc->SetPendingValue( drm->req, 8, bForceInRequest );

		if ( drm->pConnector->GetProperties().HDR_OUTPUT_METADATA )
			drm->pConnector->GetProperties().HDR_OUTPUT_METADATA->SetPendingValue( drm->req, pHDRMetadata ? pHDRMetadata->GetBlobValue() : 0lu, bForceInRequest );

		if ( drm->pConnector->GetProperties().content_type )
			drm->pConnector->GetProperties().content_type->SetPendingValue( drm->req, DRM_MODE_CONTENT_TYPE_GAME, bForceInRequest );

		GamescopeBroadcastRGBMode_t eBroadcastRGB = drm->pConnector->GetScreenType() == gamescope::GAMESCOPE_SCREEN_TYPE_EXTERNAL
			? s_ExternalBroadcastRGBMode
			: GAMESCOPE_BROADCAST_RGB_MODE_AUTOMATIC;

		if ( drm->pConnector->GetProperties().Broadcast_RGB )
			drm->pConnector->GetProperties().Broadcast_RGB->SetPendingValue( drm->req, eBroadcastRGB, bForceInRequest );
	}

	if ( drm->pCRTC && !bSleep )
	{
		if ( drm->pCRTC->GetProperties().AMD_CRTC_REGAMMA_TF )
		{
			if ( !cv_drm_debug_disable_regamma_tf )
				drm->pCRTC->GetProperties().AMD_CRTC_REGAMMA_TF->SetPendingValue( drm->req, inverse_tf( drm->pending.output_tf ), bForceInRequest );
			else
				drm->pCRTC->GetProperties().AMD_CRTC_REGAMMA_TF->SetPendingValue( drm->req, AMDGPU_TRANSFER_FUNCTION_DEFAULT, bForceInRequest );
		}
	}

	drm->flags = flags;

	int ret;
	if ( drm->pCRTC == nullptr || bSleep ) {
		ret = 0;
	} else if ( drm->bUseLiftoff ) {
		ret = drm_prepare_liftoff( drm, frameInfo, needs_modeset );
	} else {
		ret = 0;
	}

	if ( ret != 0 ) {
		drm_rollback( drm );

		drmModeAtomicFree( drm->req );
		drm->req = nullptr;

		drm->m_FbIdsInRequest.clear();
		drm_close_in_fences( drm );

		if ( needs_modeset )
			drm->needs_modeset = true;
	}

	return ret;
}

bool drm_poll_state( struct drm_t *drm )
{
	int out_of_date = drm->out_of_date.exchange(false);
	if ( !out_of_date )
		return false;

	refresh_state( drm );

	setup_best_connector(drm, out_of_date >= 2, false);

	return true;
}

static bool drm_set_crtc( struct drm_t *drm, gamescope::CDRMCRTC *pCRTC )
{
	drm->pCRTC = pCRTC;
	drm->needs_modeset = true;

	drm->pPrimaryPlane = find_primary_plane( drm );
	if ( drm->pPrimaryPlane == nullptr ) {
		drm_log.errorf("could not find a suitable primary plane");
		return false;
	}

	struct liftoff_output *lo_output = liftoff_output_create( drm->lo_device, pCRTC->GetObjectId() );
	if ( lo_output == nullptr )
		return false;

	for ( int i = 0; i < k_nMaxLayers; i++ )
	{
		liftoff_layer_destroy( drm->lo_layers[ i ] );
		drm->lo_layers[ i ] = liftoff_layer_create( lo_output );
		if ( drm->lo_layers[ i ] == nullptr )
			return false;
	}

	liftoff_output_destroy( drm->lo_output );
	drm->lo_output = lo_output;

	return true;
}

bool drm_set_connector( struct drm_t *drm, gamescope::CDRMConnector *conn )
{
	drm_log.infof("selecting connector %s", conn->GetName());

	gamescope::CDRMCRTC *pCRTC = find_crtc_for_connector(drm, conn);
	if (pCRTC == nullptr)
	{
		drm_log.errorf("no CRTC found!");
		return false;
	}

	if (!drm_set_crtc(drm, pCRTC)) {
		return false;
	}

	// If we are changing connector, zero out the current and pending mode IDs.
	// So we don't try to use one mode from the old connector on the new one if we roll back.
	drm->pending.mode_id = nullptr;
	drm->current.mode_id = nullptr;

	drm->pConnector = conn;
	drm->needs_modeset = true;

	return true;
}

static void drm_unset_connector( struct drm_t *drm )
{
	drm->pCRTC = nullptr;
	drm->pPrimaryPlane = nullptr;

	for ( int i = 0; i < k_nMaxLayers; i++ )
	{
		liftoff_layer_destroy( drm->lo_layers[ i ] );
		drm->lo_layers[ i ] = nullptr;
	}

	liftoff_output_destroy(drm->lo_output);
	drm->lo_output = nullptr;

	drm->pConnector = nullptr;
	drm->needs_modeset = true;
}

bool drm_get_vrr_in_use(struct drm_t *drm)
{
	if ( !drm->pCRTC || !drm->pCRTC->GetProperties().VRR_ENABLED )
		return false;

	return !!drm->pCRTC->GetProperties().VRR_ENABLED->GetCurrentValue();
}

gamescope::GamescopeScreenType drm_get_screen_type(struct drm_t *drm)
{
	if ( !drm->pConnector )
		return gamescope::GAMESCOPE_SCREEN_TYPE_INTERNAL;

	return drm->pConnector->GetScreenType();
}

bool drm_update_color_mgmt(struct drm_t *drm)
{
	if ( !drm_supports_amd_color_mgmt( drm ) )
		return true;

	if ( g_ColorMgmt.serial == drm->current.color_mgmt_serial )
		return true;

	drm->pending.color_mgmt_serial = g_ColorMgmt.serial;

	for ( uint32_t i = 0; i < EOTF_Count; i++ )
	{
		drm->pending.shaperlut_id[ i ] = 0;
		drm->pending.lut3d_id[ i ] = 0;
	}

	for ( uint32_t i = 0; i < EOTF_Count; i++ )
	{
		if ( !g_ColorMgmtLuts[i].HasLuts() )
			continue;

		drm->pending.shaperlut_id[ i ] = GetBackend()->CreateBackendBlob( g_ColorMgmtLuts[i].lut1d );
		drm->pending.lut3d_id[ i ] = GetBackend()->CreateBackendBlob( g_ColorMgmtLuts[i].lut3d );
	}

	return true;
}

int g_nDynamicRefreshHz = 0;

static void drm_unset_mode( struct drm_t *drm, bool force )
{
	drm->pending.mode_id = 0;
	drm->needs_modeset = true;

	const bool bPreferredSize = drm->preferred_width != 0 || drm->preferred_height != 0;
	const bool bPreferredRefresh = drm->preferred_refresh != 0;

	// Forced on a cold start, a mode pick while headless and a session resume.
	saved_mode mode_info{};
	const bool bLastMode = ( !bPreferredSize || !bPreferredRefresh ) && force && get_last_display_mode( mode_info );

	// An explicit -W/-H or -r wins for its part, else a display that went away mid-session keeps its mode.
	if ( bPreferredSize )
	{
		g_nOutputWidth = drm->preferred_width;
		g_nOutputHeight = drm->preferred_height;
	}
	else if ( bLastMode )
	{
		g_nOutputWidth = mode_info.width;
		g_nOutputHeight = mode_info.height;
	}

	if ( bPreferredRefresh )
		g_nOutputRefresh = drm->preferred_refresh;
	else if ( bLastMode )
		g_nOutputRefresh = gamescope::ConvertHztomHz( mode_info.refresh );

	if (g_nOutputHeight == 0)
		g_nOutputHeight = 720;
	if (g_nOutputWidth == 0)
		g_nOutputWidth = g_nOutputHeight * 16 / 9;
	if (g_nOutputRefresh == 0)
		g_nOutputRefresh = gamescope::ConvertHztomHz( 60 );
	g_nDynamicRefreshHz = 0;

	g_bRotated = false;
	g_uOutputRotation = 0;
}

// Bitmask of DRM_MODE_ROTATE_* the plane can do at scanout (just ROTATE_0 if it can't rotate).
static uint64_t drm_plane_supported_rotations( struct drm_t *drm, gamescope::CDRMPlane *pPlane )
{
	if ( !pPlane->GetProperties().rotation )
		return DRM_MODE_ROTATE_0;

	drmModePropertyRes *pProp = drmModeGetProperty( drm->fd, pPlane->GetProperties().rotation->GetPropertyId() );
	if ( !pProp )
		return DRM_MODE_ROTATE_0;
	defer( drmModeFreeProperty( pProp ) );

	if ( !( pProp->flags & DRM_MODE_PROP_BITMASK ) )
		return DRM_MODE_ROTATE_0;

	uint64_t ulSupported = 0;
	for ( int i = 0; i < pProp->count_enums; i++ )
		if ( pProp->enums[i].value < 64 )
			ulSupported |= 1ull << pProp->enums[i].value;

	return ulSupported;
}

bool drm_set_mode( struct drm_t *drm, const drmModeModeInfo *mode )
{
	if (!drm->pConnector || !drm->pConnector->GetModeConnector())
		return false;

	drm_log.infof("selecting mode %dx%d@%uHz", mode->hdisplay, mode->vdisplay, mode->vrefresh);

	drm->pending.mode_id = GetBackend()->CreateBackendBlob( *mode );
	drm->needs_modeset = true;

	g_nOutputRefresh = gamescope::GetModeRefresh( mode );
	g_nDynamicRefreshHz = 0;

	g_nsVsync = gamescope::GetVblankNs( mode );
	drm_log.infof("Vblank ns: %lu", g_nsVsync);

	update_drm_effective_orientations(drm, mode);

	// 90/270 transpose the output (g_bRotated); 180 flips in place.
	uint32_t uStep = 0;
	uint64_t ulNeeded = 0;
	switch ( drm->pConnector->GetCurrentOrientation() )
	{
	default:
	case GAMESCOPE_PANEL_ORIENTATION_0:
		g_bRotated = false;
		g_nOutputWidth = mode->hdisplay;
		g_nOutputHeight = mode->vdisplay;


		break;
	case GAMESCOPE_PANEL_ORIENTATION_180:
		g_bRotated = false;
		g_nOutputWidth = mode->hdisplay;
		g_nOutputHeight = mode->vdisplay;
		uStep = 2u;
		ulNeeded = DRM_MODE_ROTATE_180;
		break;
	case GAMESCOPE_PANEL_ORIENTATION_90:
		g_bRotated = true;
		g_nOutputWidth = mode->vdisplay;
		g_nOutputHeight = mode->hdisplay;
		uStep = 1u;
		ulNeeded = DRM_MODE_ROTATE_90;
		break;
	case GAMESCOPE_PANEL_ORIENTATION_270:
		g_bRotated = true;
		g_nOutputWidth = mode->vdisplay;
		g_nOutputHeight = mode->hdisplay;
		uStep = 3u;
		ulNeeded = DRM_MODE_ROTATE_270;
		break;
	}

	// Rotate in the compositor when the scanout plane can't do the panel's orientation.
	g_uOutputRotation = 0;
	if ( uStep )
	{
		const bool bScanoutCanRotate = drm->pPrimaryPlane && ( drm_plane_supported_rotations( drm, drm->pPrimaryPlane ) & ulNeeded );
		if ( g_bForceCompositionRotation || !bScanoutCanRotate )
			g_uOutputRotation = uStep;
	}

	return true;
}

bool drm_set_refresh( struct drm_t *drm, int refresh )
{
	int width = g_nOutputWidth;
	int height = g_nOutputHeight;

	if ( g_bRotated ) {
		int tmp = width;
		width = height;
		height = tmp;
	}
	if (!drm->pConnector || !drm->pConnector->GetModeConnector())
		return false;

	drmModeConnector *connector = drm->pConnector->GetModeConnector();
	const drmModeModeInfo *existing_mode = find_mode(connector, width, height, refresh);
	drmModeModeInfo mode = {0};
	if ( existing_mode )
	{
		mode = *existing_mode;
	}
	else
	{
		if ( g_DRM.pConnector && g_DRM.pConnector->GetModeGenerator() )
		{
			const drmModeModeInfo *preferred_mode = find_mode(connector, 0, 0, 0);
			mode = g_DRM.pConnector->GetModeGenerator()( preferred_mode, refresh );
		}
		else
		{
			/* TODO: check refresh is within the EDID limits */
			switch ( g_eGamescopeModeGeneration )
			{
			case gamescope::GAMESCOPE_MODE_GENERATE_CVT:
				generate_cvt_mode( &mode, width, height, refresh, true, false );
				break;
			case gamescope::GAMESCOPE_MODE_GENERATE_FIXED:
				{
					const drmModeModeInfo *preferred_mode = find_mode(connector, 0, 0, 0);
					generate_fixed_mode( &mode, preferred_mode, refresh );
					break;
				}
			}
		}
	}

	mode.type = DRM_MODE_TYPE_USERDEF;

	bool bSuccess = drm_set_mode(drm, &mode);
	if ( !bSuccess )
		return false;

	g_nDynamicRefreshHz = refresh;

	return true;
}

bool drm_set_resolution( struct drm_t *drm, int width, int height )
{
	if (!drm->pConnector || !drm->pConnector->GetModeConnector())
		return false;

	drmModeConnector *connector = drm->pConnector->GetModeConnector();
	const drmModeModeInfo *mode = find_mode(connector, width, height, 0);
	if ( !mode )
	{
		return false;
	}

	return drm_set_mode(drm, mode);
}

bool drm_get_vrr_capable(struct drm_t *drm)
{
	if ( drm->pConnector )
		return drm->pConnector->SupportsVRR();

	return false;
}

bool drm_supports_hdr( struct drm_t *drm, uint16_t *maxCLL, uint16_t *maxFALL )
{
	if ( drm->pConnector && drm->pConnector->SupportsHDR() )
	{
		if ( maxCLL )
			*maxCLL = drm->pConnector->GetHDRInfo().uMaxContentLightLevel;
		if ( maxFALL )
			*maxFALL = drm->pConnector->GetHDRInfo().uMaxFrameAverageLuminance;
		return true;
	}

	return false;
}

const char *drm_get_connector_name(struct drm_t *drm)
{
	if ( !drm->pConnector )
		return nullptr;

	return drm->pConnector->GetName();
}

const char *drm_get_device_name(struct drm_t *drm)
{
	return drm->device_name;
}

std::pair<uint32_t, uint32_t> drm_get_connector_identifier(struct drm_t *drm)
{
	if ( !drm->pConnector )
		return { 0u, 0u };

	return std::make_pair(drm->pConnector->GetModeConnector()->connector_type, drm->pConnector->GetModeConnector()->connector_type_id);
}

bool drm_supports_amd_color_mgmt(struct drm_t *drm)
{
	if (drm->generic_plane_color)
		return false;
	if ( g_bForceDisableColorMgmt )
		return false;

	if ( !drm->pPrimaryPlane )
		return false;

	return drm->pPrimaryPlane->GetProperties().AMD_PLANE_SHAPER_TF.has_value() &&
			drm->pPrimaryPlane->GetProperties().AMD_PLANE_LUT3D.has_value() &&
			drm->pPrimaryPlane->GetProperties().AMD_PLANE_CTM.has_value() &&         // dm->dc->caps.color.mpc.gamut_remap
			drm->pPrimaryPlane->GetProperties().AMD_PLANE_SHAPER_LUT.has_value() && // dpp_color_caps.hw_3d_lut
			drm->pPrimaryPlane->GetProperties().AMD_PLANE_DEGAMMA_TF.has_value();   // dpp_color_caps.dgam_ram || dpp_color_caps.gamma_corr
}

bool drm_supports_color_mgmt(struct drm_t *drm)
{
	if (g_bForceDisableColorMgmt) return false;
	return drm_supports_amd_color_mgmt(drm) || !drm->color_pipelines.empty() ||
		(drm->pCRTC && drm->pCRTC->GetProperties().GAMMA_LUT && drm->pCRTC->GetGammaSize() >= 2);
}

bool drm_supports_srgb_to_pq(struct drm_t *drm)
{
	// Generic eligibility is decided for the complete frame during atomic testing.
	if (!drm_supports_amd_color_mgmt(drm)) return drm_supports_color_mgmt(drm);
	if ( g_bForceDisableColorMgmt )
		return false;

	if ( !drm->pPrimaryPlane )
		return false;

	return drm->pPrimaryPlane->GetProperties().AMD_PLANE_BLEND_TF.has_value(); // dpp_color_caps.ogam_ram
}

std::span<const uint32_t> drm_get_valid_refresh_rates( struct drm_t *drm )
{
	if ( drm && drm->pConnector )
		return drm->pConnector->GetValidDynamicRefreshRates();

	return std::span<const uint32_t>{};
}

namespace gamescope
{
	class CDRMBackend;

	class CDRMBackend final : public CBaseBackend
	{
	public:
		CDRMBackend()
		{
		}

		virtual ~CDRMBackend()
		{
			if ( g_DRM.fd != -1 )
				finish_drm( &g_DRM );
		}

		virtual bool Init() override
		{
			if ( !wlsession_init() )
			{
				fprintf( stderr, "Failed to initialize Wayland session\n" );
				return false;
			}

			const auto connector_priorities = parse_connector_priorities( g_sOutputName );
			int kms_fd = wlsession_open_kms( nullptr, get_best_connector_priority_for_fd, &connector_priorities );
			if ( kms_fd < 0 )
			{
				fprintf( stderr, "Failed to select KMS device\n" );
				return false;
			}

			// PCI IDs only help prefer the Vulkan device associated with this KMS
			// device. Platform DRM devices do not have them and are still valid.
			struct stat fd_stat = {};
			if ( fstat( kms_fd, &fd_stat ) == 0 )
			{
				drmDevice *drm_device = nullptr;
				if ( drmGetDeviceFromDevId( fd_stat.st_rdev, 0, &drm_device ) == 0 && drm_device )
				{
					if ( drm_device->bustype == DRM_BUS_PCI && drm_device->deviceinfo.pci )
					{
						g_preferVendorID = drm_device->deviceinfo.pci->vendor_id;
						g_preferDeviceID = drm_device->deviceinfo.pci->device_id;
					}
					else
					{
						drm_log.infof( "Selected KMS device has no PCI identity; skipping KMS PCI preference" );
					}
				}
				else
				{
					drm_log.warnf( "Could not identify selected KMS device; skipping KMS PCI preference" );
				}
				if ( drm_device )
					drmFreeDevice( &drm_device );
			}
			else
			{
				drm_log.warnf( "Could not stat selected KMS device; skipping KMS PCI preference" );
			}

			if ( drmVersion *pVersion = drmGetVersion( kms_fd ) )
			{
				if ( pVersion->name && strcmp( pVersion->name, "vc4" ) == 0 )
				{
					const char *pDebug = getenv( "V3D_DEBUG" );
					std::string debug = pDebug && *pDebug ? std::string( pDebug ) + ",disable_tfu" : "disable_tfu";
					setenv( "V3D_DEBUG", debug.c_str(), 1 );
					drm_log.infof( "VC4: disabling corrupting V3DV TFU transfers" );
				}
				drmFreeVersion( pVersion );
			}

			if ( !vulkan_init( vulkan_get_instance(), VK_NULL_HANDLE ) )
			{
				fprintf( stderr, "Failed to initialize Vulkan\n" );
				return false;
			}

			return init_drm( &g_DRM, kms_fd, g_nPreferredOutputWidth, g_nPreferredOutputHeight, g_nNestedRefresh );
		}

		virtual bool PostInit() override
		{
			if ( g_DRM.pConnector )
			{
				WritePatchedEdid( g_DRM.pConnector->GetRawEDID(), g_DRM.pConnector->GetHDRInfo(), g_bRotated );
				WriteModeListFile( g_DRM.pConnector->GetModes() );
			}
			return true;
		}

        virtual std::span<const char *const> GetInstanceExtensions() const override
		{
			return std::span<const char *const>{};
		}
        virtual std::span<const char *const> GetDeviceExtensions( VkPhysicalDevice pVkPhysicalDevice ) const override
		{
			return std::span<const char *const>{};
		}
        virtual VkImageLayout GetPresentLayout() const override
		{
			// Does not matter, as this has a queue family transition
			// to VK_QUEUE_FAMILY_FOREIGN_EXT queue,
			// thus: newLayout is ignored.
			return VK_IMAGE_LAYOUT_GENERAL;
		}
        virtual void GetPreferredOutputFormat( uint32_t *pPrimaryPlaneFormat, uint32_t *pOverlayPlaneFormat ) const override
        {
			*pPrimaryPlaneFormat = g_nDRMFormat;
			*pOverlayPlaneFormat = g_nDRMFormatOverlay;
        }
		virtual bool ValidPhysicalDevice( VkPhysicalDevice pVkPhysicalDevice ) const override
		{
			return vulkan_has_drm_props();
		}

		virtual int Present( const FrameInfo_t *pFrameInfo, bool bAsync )
		{
			m_bAsyncCompositeTiming = false;
			const FrameInfo_t *pUnprocessedFrameInfo = pFrameInfo;
			static uint64_t s_ulLastTime = get_time_in_nanos();
			uint64_t ulNow = get_time_in_nanos();
			drm_log.debugf( "CDRMBackend::Present Begin: %lu -> delta: %lu", ulNow, ulNow - s_ulLastTime );
			s_ulLastTime = ulNow;

			bool bWantsPartialComposite = pFrameInfo->layers.count() >= 3 && !kDisablePartialComposition;

			static bool s_bWasFirstFrame = true;
			bool bWasFirstFrame = s_bWasFirstFrame;
			s_bWasFirstFrame = false;

			bool bDrewCursor = false;
			for ( int i = 0; i < pFrameInfo->layers.count(); i++ )
			{
				if ( pFrameInfo->layers.get( i ).zpos == g_zposCursor )
				{
					bDrewCursor = true;
					break;
				}
			}

			FrameInfo_t fsrScanoutFrameInfo;
			const FrameInfo_t::Layer_t &fsrBaseLayer = pFrameInfo->layers[0];
			const bool bForceBackendComposition = UsesBackendAllocatedScanout();
			const bool bCanPreprocessFsrBase =
				!bForceBackendComposition &&
				pFrameInfo->useFSRLayer0 &&
				pFrameInfo->layers.count() > 1 &&
				!cv_composite_force &&
				!bWasFirstFrame &&
				!pFrameInfo->useNISLayer0 &&
				!pFrameInfo->blurLayer0 &&
				!pFrameInfo->bFadingOut &&
				!g_bColorSliderInUse &&
				!g_bHDRItmEnable &&
				!g_bOutputHDREnabled &&
				g_uCompositeDebug == 0 &&
				g_reshade_effect.empty() &&
				( k_bUseCursorPlane || !bDrewCursor ) &&
				pFrameInfo->outputEncodingEOTF == EOTF_Gamma22 &&
				( !pFrameInfo->applyOutputColorMgmt || SupportsColorManagement() ) &&
				close_enough( fsrBaseLayer.opacity, 1.0f ) &&
				( fsrBaseLayer.colorspace == GAMESCOPE_APP_TEXTURE_COLORSPACE_LINEAR ||
				  fsrBaseLayer.colorspace == GAMESCOPE_APP_TEXTURE_COLORSPACE_SRGB );

			if ( bCanPreprocessFsrBase )
			{
				// Run FSR on the base plane only. Keep it in the source SDR space so
				// KMS can apply the same per-plane color management as direct scanout.
				FrameInfo_t fsrBaseFrameInfo = {};
				fsrBaseFrameInfo.useFSRLayer0 = true;
				fsrBaseFrameInfo.applyOutputColorMgmt = true;
				fsrBaseFrameInfo.outputEncodingEOTF = EOTF_Gamma22;
				fsrBaseFrameInfo.layers.push();
				fsrBaseFrameInfo.layers[0] = fsrBaseLayer;
				fsrBaseFrameInfo.layers[0].ctm = nullptr;

				std::optional<uint64_t> oFsrResult = vulkan_composite( &fsrBaseFrameInfo, nullptr, false );
				if ( oFsrResult )
				{
					vulkan_wait( *oFsrResult, true );

					fsrScanoutFrameInfo = *pFrameInfo;
					fsrScanoutFrameInfo.useFSRLayer0 = false;

					FrameInfo_t::Layer_t &scanoutBaseLayer = fsrScanoutFrameInfo.layers[0];
					scanoutBaseLayer.tex = vulkan_get_last_output_image( false, false );
					scanoutBaseLayer.offset = { 0.0f, 0.0f };
					scanoutBaseLayer.scale = { 1.0f, 1.0f };
					scanoutBaseLayer.opacity = 1.0f;
					scanoutBaseLayer.blackBorder = false;
					scanoutBaseLayer.filter = GamescopeUpscaleFilter::NEAREST;
					scanoutBaseLayer.colorspace = GAMESCOPE_APP_TEXTURE_COLORSPACE_SRGB;
					scanoutBaseLayer.eAlphaBlendingMode = ALPHA_BLENDING_MODE_NONE;

					// Normal plane assignment can scan out the FSR result with compatible
					// overlays and the cursor on hardware planes. Full composition must
					// use the original frame because output images cannot be sampled.
					pFrameInfo = &fsrScanoutFrameInfo;
				}
			}

			bool bLayer0ScreenSize = close_enough(pFrameInfo->layers[0].scale.x, 1.0f) && close_enough(pFrameInfo->layers[0].scale.y, 1.0f);

			GamescopeUpscaleFilter eLayer0Filter = pFrameInfo->layers.get( 0 ).filter;
			bool bNeedsCompositeFromFilter = ( eLayer0Filter == GamescopeUpscaleFilter::NEAREST ||
			                                   eLayer0Filter == GamescopeUpscaleFilter::PIXEL ) && !bLayer0ScreenSize;

			bool bNeedsFullComposite = false;
			bNeedsFullComposite |= cv_composite_force;
			// These backends require scanout buffers allocated on the KMS device.
			bNeedsFullComposite |= bForceBackendComposition;
			bNeedsFullComposite |= bWasFirstFrame;
			bNeedsFullComposite |= pFrameInfo->useFSRLayer0;
			bNeedsFullComposite |= pFrameInfo->useNISLayer0;
			bNeedsFullComposite |= pFrameInfo->useSGSRLayer0;
			bNeedsFullComposite |= pFrameInfo->blurLayer0;
			bNeedsFullComposite |= bNeedsCompositeFromFilter;
			bNeedsFullComposite |= !k_bUseCursorPlane && bDrewCursor;
			bNeedsFullComposite |= g_bColorSliderInUse;
			bNeedsFullComposite |= pFrameInfo->bFadingOut;
			bNeedsFullComposite |= !g_reshade_effect.empty();

			if ( !SupportsColorManagement() ) {
				// Fuzzy match default values to see if we need to composite
				bNeedsFullComposite |= g_ColorMgmt.pending.nightmode.amount != 0.0f;
				bNeedsFullComposite |= g_ColorMgmt.pending.outputVirtualWhite.x > 0 &&
					abs(g_ColorMgmt.pending.outputVirtualWhite.x - 0.3127f) > 0.001f;
				bNeedsFullComposite |= g_ColorMgmt.pending.outputVirtualWhite.y > 0 &&
					abs(g_ColorMgmt.pending.outputVirtualWhite.y - 0.3290f) > 0.001f;
				bNeedsFullComposite |= g_ColorMgmt.pending.sdrGamutWideness >= 0 &&
					abs(g_ColorMgmt.pending.sdrGamutWideness - 0.5f) > 0.02f;
			}

			if ( g_bOutputHDREnabled )
			{
				bNeedsFullComposite |= g_bHDRItmEnable;
				if ( !SupportsSRGBtoPQ() )
					bNeedsFullComposite |= ( pFrameInfo->layers.count() > 1 || pFrameInfo->layers[0].colorspace != GAMESCOPE_APP_TEXTURE_COLORSPACE_HDR10_PQ );
			}
			else
			{
				if ( !SupportsSRGBtoPQ() )
					bNeedsFullComposite |= ColorspaceIsHDR( pFrameInfo->layers[0].colorspace );
			}

			bNeedsFullComposite |= !!(g_uCompositeDebug & CompositeDebugFlag::Heatmap);
			bNeedsFullComposite |= g_uOutputRotation != 0; // can't rotate planes at scanout


			bool bDoComposite = true;
			if ( !bNeedsFullComposite && !bWantsPartialComposite )
			{
				// Save the pending mode so it can be restored after drm_rollback() and carried
				// over to the composite path
				std::shared_ptr<gamescope::BackendBlob> pPendingModeId = g_DRM.pending.mode_id;
				int ret = drm_prepare( &g_DRM, bAsync, pFrameInfo );
				if ( ret == 0 )
					bDoComposite = false;
				else if ( ret == -EACCES )
					return 0;
				else if ( g_DRM.needs_modeset )
					g_DRM.pending.mode_id = pPendingModeId;
			}

			// Update to let the vblank manager know we are currently compositing.
			GetVBlankTimer().UpdateWasCompositing( bDoComposite );

			if ( !bDoComposite )
			{
				// Scanout + Planes Path
				m_bWasPartialCompositing = false;
				m_bWasCompositing = false;
				if ( pFrameInfo->layers.count() == 2 )
					m_nLastSingleOverlayZPos = pFrameInfo->layers.get( 1 ).zpos;

				return Commit( pFrameInfo );
			}

			// Generic color rejection uses the existing full composition fallback.
			if (!drm_supports_amd_color_mgmt(&g_DRM))
				bNeedsFullComposite = true;

			// Composition Path
			if ( kDisablePartialComposition )
				bNeedsFullComposite = true;

			FrameInfo_t compositeFrameInfo = *pFrameInfo;

			if ( compositeFrameInfo.layers.count() == 1 )
			{
				// If we failed to flip a single plane then
				// we definitely need to composite for some reason...
				bNeedsFullComposite = true;
			}

			if ( !bNeedsFullComposite )
			{
				// If we want to partial composite, fallback to full
				// composite if we have mismatching colorspaces in our overlays.
				// This is 2, and we do i-1 so 1...layers.count(). So AFTER we have removed baseplane.
				// Overlays only.
				//
				// Josh:
				// We could handle mismatching colorspaces for partial composition
				// but I want to keep overlay -> partial composition promotion as simple
				// as possible, using the same 3D + SHAPER LUTs + BLEND in DRM
				// as changing them is incredibly expensive!! It takes forever.
				// We can't just point it to random BDA or whatever, it has to be uploaded slowly
				// thru registers which is SUPER SLOW.
				// This avoids stutter.
				for ( int i = 2; i < compositeFrameInfo.layers.count(); i++ )
				{
					if ( pFrameInfo->layers.get( i - 1 ).colorspace != pFrameInfo->layers.get( i ).colorspace )
					{
						bNeedsFullComposite = true;
						break;
					}
				}
			}

			if ( bNeedsFullComposite && pFrameInfo == &fsrScanoutFrameInfo )
				compositeFrameInfo = *pUnprocessedFrameInfo;

			// If we ever promoted from partial -> full, for the first frame
			// do NOT defer this partial composition.
			// We were already stalling for the full composition before, so it's not an issue
			// for latency, we just need to make sure we get 1 partial frame that isn't deferred
			// in time so we don't lose layers.
			bool bDefer = !bNeedsFullComposite && ( !m_bWasCompositing || m_bWasPartialCompositing );

			// If doing a partial composition then remove the baseplane
			// from our frameinfo to composite.
			if ( !bNeedsFullComposite )
			{
				for ( int i = 1; i < compositeFrameInfo.layers.count(); i++ )
					compositeFrameInfo.layers.get( i - 1 ) = compositeFrameInfo.layers.get( i );
				compositeFrameInfo.layers.pop();

				// When doing partial composition, apply the shaper + 3D LUT stuff
				// at scanout.
				for ( uint32_t nEOTF = 0; nEOTF < EOTF_Count; nEOTF++ ) {
					compositeFrameInfo.shaperLut[ nEOTF ] = nullptr;
					compositeFrameInfo.lut3D[ nEOTF ] = nullptr;
				}
			}

			// If using composite debug markers, make sure we mark them as partial
			// so we know!
			if ( bDefer && !!( g_uCompositeDebug & CompositeDebugFlag::Markers ) )
				g_uCompositeDebug |= CompositeDebugFlag::Markers_Partial;

			std::optional oCompositeResult = vulkan_composite( &compositeFrameInfo, nullptr, !bNeedsFullComposite );

			m_bWasCompositing = true;

			g_uCompositeDebug &= ~CompositeDebugFlag::Markers_Partial;

			if ( !oCompositeResult )
			{
				xwm_log.errorf("vulkan_composite failed");
				return -EINVAL;
			}

			auto completion = g_device.takeCompletionFence(*oCompositeResult);
			const int completionFd = completion.value_or(-1);
			defer(if (completionFd >= 0) close(completionFd));
			const bool explicitFence = !cv_drm_debug_disable_in_fence_fd && g_DRM.pPrimaryPlane->GetProperties().IN_FENCE_FD;
			if (completion && completionFd >= 0 && !explicitFence)
			{
				auto image = vulkan_get_last_output_image(false, false);
				dma_buf_import_sync_file fence = { .flags = DMA_BUF_SYNC_WRITE, .fd = completionFd };
				if (ioctl(image->dmabuf().fd[0], DMA_BUF_IOCTL_IMPORT_SYNC_FILE, &fence) != 0)
				{
					static bool warned = false;
					if (!std::exchange(warned, true))
						drm_log.errorf_errno("Failed to attach compositor completion fence to DMA-BUF");
					completion.reset();
				}
			}
			if (completion)
				m_bAsyncCompositeTiming = watch_composite(completionFd, g_SteamCompMgrVBlankTime.ulWakeupTime);
			if (!m_bAsyncCompositeTiming)
				vulkan_wait(*oCompositeResult, true);
			static bool loggedFencePath = false;
			if (!std::exchange(loggedFencePath, true))
				drm_log.infof("Compositor completion: %s", m_bAsyncCompositeTiming ? (explicitFence ? "IN_FENCE_FD" : "DMA-BUF fence") : "CPU wait");

			FrameInfo_t presentCompFrameInfo = {};
			presentCompFrameInfo.allowVRR = pFrameInfo->allowVRR;
			presentCompFrameInfo.dpms = pFrameInfo->dpms;
			presentCompFrameInfo.outputEncodingEOTF = pFrameInfo->outputEncodingEOTF;

			if ( bNeedsFullComposite )
			{
				presentCompFrameInfo.applyOutputColorMgmt = false;
				FrameInfo_t::Layer_t *baseLayer = presentCompFrameInfo.layers.push();
				assert( baseLayer );
				baseLayer->scale.x = 1.0;
				baseLayer->scale.y = 1.0;
				baseLayer->opacity = 1.0;
				baseLayer->zpos = g_zposBase;

				baseLayer->tex = vulkan_get_last_output_image( false, false );
				baseLayer->acquireFenceFd = explicitFence && m_bAsyncCompositeTiming ? completionFd : -1;
				baseLayer->applyColorMgmt = false;

				baseLayer->filter = GamescopeUpscaleFilter::NEAREST;
				baseLayer->ctm = nullptr;
				baseLayer->colorspace = pFrameInfo->outputEncodingEOTF == EOTF_PQ ? GAMESCOPE_APP_TEXTURE_COLORSPACE_HDR10_PQ : GAMESCOPE_APP_TEXTURE_COLORSPACE_SRGB;

				m_bWasPartialCompositing = false;
			}
			else
			{
				if ( m_bWasPartialCompositing || !bDefer )
				{
					presentCompFrameInfo.applyOutputColorMgmt = g_ColorMgmt.pending.enabled;
					FrameInfo_t::Layer_t *baseLayer = presentCompFrameInfo.layers.push();
					assert( baseLayer );
					*baseLayer = pFrameInfo->layers.get( 0 );
					baseLayer->zpos = g_zposBase;

					FrameInfo_t::Layer_t *overlayLayer = presentCompFrameInfo.layers.push();
					assert( overlayLayer );
					overlayLayer->scale.x = 1.0;
					overlayLayer->scale.y = 1.0;
					overlayLayer->opacity = 1.0;
					overlayLayer->zpos = g_zposOverlay;

					overlayLayer->tex = vulkan_get_last_output_image( true, bDefer );
					overlayLayer->acquireFenceFd = explicitFence && m_bAsyncCompositeTiming ? completionFd : -1;
					overlayLayer->applyColorMgmt = g_ColorMgmt.pending.enabled;

					overlayLayer->filter = GamescopeUpscaleFilter::NEAREST;
					// Partial composition stuff has the same colorspace.
					// So read that from the composite frame info
					overlayLayer->ctm = nullptr;
					overlayLayer->colorspace = compositeFrameInfo.layers.get( 0 ).colorspace;
				}
				else
				{
					// Use whatever overlay we had last while waiting for the
					// partial composition to have anything queued.
					presentCompFrameInfo.applyOutputColorMgmt = g_ColorMgmt.pending.enabled;
					FrameInfo_t::Layer_t *baseLayer = presentCompFrameInfo.layers.push();
					assert( baseLayer );
					*baseLayer = pFrameInfo->layers.get( 0 );
					baseLayer->zpos = g_zposBase;

					const FrameInfo_t::Layer_t *lastPresentedOverlayLayer = nullptr;
					for (int i = 0; i < pFrameInfo->layers.count(); i++)
					{
						if ( pFrameInfo->layers.get( i ).zpos == m_nLastSingleOverlayZPos )
						{
							lastPresentedOverlayLayer = &pFrameInfo->layers.get( i );
							break;
						}
					}

					if ( lastPresentedOverlayLayer )
					{
						FrameInfo_t::Layer_t *overlayLayer = presentCompFrameInfo.layers.push();
						assert( overlayLayer );
						*overlayLayer = *lastPresentedOverlayLayer;
						overlayLayer->zpos = g_zposOverlay;
					}
				}

				m_bWasPartialCompositing = true;
			}


			int ret = drm_prepare( &g_DRM, bAsync, &presentCompFrameInfo );

			// Happens when we're VT-switched away
			if ( ret == -EACCES )
				return 0;

			if ( ret != 0 )
			{
				if ( g_DRM.current.mode_id == 0 )
				{
					xwm_log.errorf("We failed our modeset and have no mode to fall back to! (Initial modeset failed?): %s", strerror(-ret));
					return 0;
				}

				xwm_log.errorf("Failed to prepare 1-layer flip (%s), trying again with previous mode if modeset needed", strerror( -ret ));

				// Try once again to in case we need to fall back to another mode.
				ret = drm_prepare( &g_DRM, bAsync, &compositeFrameInfo );

				// Happens when we're VT-switched away
				if ( ret == -EACCES )
					return 0;

				if ( ret != 0 )
				{
					xwm_log.errorf("Failed to prepare 1-layer flip entirely: %s", strerror( -ret ));
					// We should always handle a 1-layer flip, this used to abort,
					// but lets be more friendly and just avoid a commit and try again later.
					// Let's re-poll our state, and force grab the best connector again.
					//
					// Some intense connector hotplugging could be occuring and the
					// connector could become destroyed before we had a chance to use it
					// as we hadn't reffed it in a commit yet.
					this->DirtyState( true, false );
					this->PollState();
					return ret;
				}
			}

			return Commit( &compositeFrameInfo );
		}

		virtual void DirtyState( bool bForce, bool bForceModeset ) override
		{
			if ( bForceModeset )
				g_DRM.needs_modeset = true;
			g_DRM.out_of_date = std::max<int>( g_DRM.out_of_date, bForce ? 2 : 1 );
			g_DRM.paused = !wlsession_active();
		}

		virtual bool PollState() override
		{
			return drm_poll_state( &g_DRM );
		}

		virtual std::shared_ptr<BackendBlob> CreateBackendBlob( const std::type_info &type, std::span<const uint8_t> data ) override
		{
			uint32_t uBlob = 0;
			if ( type == typeid( glm::mat3x4 ) )
			{
				assert( data.size() == sizeof( glm::mat3x4 ) );

				drm_color_ctm2 ctm2;
				const float *pData = reinterpret_cast<const float *>( data.data() );
				for ( uint32_t i = 0; i < 12; i++ )
					ctm2.matrix[i] = drm_calc_s31_32( pData[i] );

				if ( drmModeCreatePropertyBlob( g_DRM.fd, reinterpret_cast<const void *>( &ctm2 ), sizeof( ctm2 ), &uBlob ) != 0 )
					return nullptr;
			}
			else
			{
				if ( drmModeCreatePropertyBlob( g_DRM.fd, data.data(), data.size(), &uBlob ) != 0 )
					return nullptr;
			}

			return std::make_shared<BackendBlob>( data, uBlob, true );
		}

		virtual OwningRc<IBackendFb> ImportDmabufToBackend( wlr_dmabuf_attributes *pDmaBuf ) override
		{
			return drm_fbid_from_dmabuf( &g_DRM, pDmaBuf );
		}

		virtual bool UsesBackendAllocatedScanout() const override
		{
			return g_DRM.is_nvidia || g_DRM.is_vc4;
		}

		virtual bool CreateScanoutDmabuf( uint32_t uWidth, uint32_t uHeight, uint32_t uDrmFormat,
		                                  std::span<const uint64_t> ulModifiers,
		                                  wlr_dmabuf_attributes *pDmaBuf ) override
		{
#if HAVE_GBM
			if ( !g_DRM.gbm || ulModifiers.empty() )
				return false;

			struct gbm_bo *pBo = gbm_bo_create_with_modifiers2(
				g_DRM.gbm, uWidth, uHeight, uDrmFormat,
				ulModifiers.data(), ulModifiers.size(),
				GBM_BO_USE_RENDERING | GBM_BO_USE_SCANOUT );
			if ( !pBo )
			{
				drm_log.errorf_errno( "Failed to allocate GBM scanout buffer" );
				return false;
			}

			*pDmaBuf = {
				.width = int32_t( uWidth ),
				.height = int32_t( uHeight ),
				.format = uDrmFormat,
				.modifier = gbm_bo_get_modifier( pBo ),
			};

			const int nPlanes = gbm_bo_get_plane_count( pBo );
			bool bSuccess = nPlanes >= 1 && nPlanes <= WLR_DMABUF_MAX_PLANES &&
				Algorithm::Contains( ulModifiers, pDmaBuf->modifier );

			for ( int i = 0; bSuccess && i < nPlanes; i++ )
			{
				const int nFd = gbm_bo_get_fd_for_plane( pBo, i );
				if ( nFd < 0 )
				{
					drm_log.errorf_errno( "Failed to export GBM scanout buffer" );
					bSuccess = false;
					break;
				}

				pDmaBuf->fd[i] = nFd;
				pDmaBuf->offset[i] = gbm_bo_get_offset( pBo, i );
				pDmaBuf->stride[i] = gbm_bo_get_stride_for_plane( pBo, i );
				pDmaBuf->n_planes = i + 1;
			}

			gbm_bo_destroy( pBo );

			if ( !bSuccess || pDmaBuf->n_planes != nPlanes )
			{
				wlr_dmabuf_attributes_finish( pDmaBuf );
				*pDmaBuf = {};
				return false;
			}

			return true;
#else
			return false;
#endif
		}

		virtual bool UsesModifiers() const override
		{
			return g_DRM.allow_modifiers;
		}
		virtual std::span<const uint64_t> GetSupportedModifiers( uint32_t uDrmFormat ) const override
		{
			const wlr_drm_format *pFormat = wlr_drm_format_set_get( &g_DRM.formats, uDrmFormat );
			if ( !pFormat )
				return std::span<const uint64_t>{};

			return std::span<const uint64_t>{ pFormat->modifiers, pFormat->modifiers + pFormat->len };
		}

		virtual IBackendConnector *GetCurrentConnector() override
		{
			if ( g_DRM.pConnector )
				return g_DRM.pConnector;

			return &s_HeadlessConnector;
		}

		virtual IBackendConnector *GetConnector( GamescopeScreenType eScreenType ) override
		{
			if ( g_DRM.pConnector && g_DRM.pConnector->GetScreenType() == eScreenType )
				return g_DRM.pConnector;

			if ( eScreenType == GAMESCOPE_SCREEN_TYPE_INTERNAL )
			{
				for ( auto &iter : g_DRM.connectors )
				{
					gamescope::CDRMConnector *pConnector = &iter.second;
					if ( pConnector->GetScreenType() == GAMESCOPE_SCREEN_TYPE_INTERNAL )
						return pConnector;
				}
			}

			return nullptr;
		}

		virtual bool SupportsPlaneHardwareCursor() const override
		{
			return !g_DRM.is_vc4;
		}

		virtual bool SupportsTearing() const override
		{
			return g_bSupportsAsyncFlips;
		}

		virtual bool UsesVulkanSwapchain() const override
		{
			return false;
		}

        virtual bool IsSessionBased() const override
		{
			return true;
		}

		virtual bool SupportsExplicitSync() const override
		{
#if __linux__
			auto [nMajor, nMinor, nPatch] = GetKernelVersion();
			
			// Only expose support on 6.8+ for eventfd fixes.
			if ( nMajor < 6 )
				return false;

			if ( nMajor == 6 && nMinor < 8 )
				return false;
#else
			// I don't know about this for FreeBSD, etc.
			return false;
#endif

			return g_bSupportsSyncObjs && !cv_drm_debug_disable_explicit_sync;
		}

		virtual bool SupportsScreenSleep() const override
		{
			return true;
		}

		void SleepScreen( gamescope::GamescopeScreenType eType, bool bSleep ) override
		{
			if ( cv_drm_sleep_screens[ eType ] == bSleep )
				return;

			cv_drm_sleep_screens[ eType ] = bSleep;
		}

		bool ScreenAsleep( gamescope::GamescopeScreenType eType ) override
		{
			return cv_drm_sleep_screens[ eType ];
		}

		bool WaitForScreenAsleep( gamescope::GamescopeScreenType eType, uint32_t uTimeoutMs ) override
		{
			std::unique_lock lock( s_ScreenSleepMutex );
			return s_ScreenSleepCV.wait_for( lock, std::chrono::milliseconds( uTimeoutMs ), [ eType ]{ return s_bScreenDisabled[ eType ]; } );
		}

		virtual bool IsPaused() const override
		{
			return g_DRM.paused;
		}

		virtual bool IsVisible() const override
		{
			return !this->IsPaused();
		}

		virtual glm::uvec2 CursorSurfaceSize( glm::uvec2 uvecSize ) const override
		{
			if ( !k_bUseCursorPlane )
				return uvecSize;

			return glm::uvec2{ g_DRM.cursor_width, g_DRM.cursor_height };
		}

		virtual bool HackTemporarySetDynamicRefresh( int nRefresh ) override
		{
			return drm_set_refresh( &g_DRM, nRefresh );
		}

		virtual void HackUpdatePatchedEdid() override
		{
			if ( !g_DRM.pConnector )
				return;

			WritePatchedEdid( g_DRM.pConnector->GetRawEDID(), g_DRM.pConnector->GetHDRInfo(), g_bRotated );
			WriteModeListFile( g_DRM.pConnector->GetModes() );
		}

	protected:

		virtual void OnBackendBlobDestroyed( BackendBlob *pBlob ) override
		{
			if ( pBlob->GetBlobValue() )
				drmModeDestroyPropertyBlob( g_DRM.fd, pBlob->GetBlobValue() );
		}

	private:
		bool m_bWasCompositing = false;
		bool m_bAsyncCompositeTiming = false;
		bool m_bWasPartialCompositing = false;
		int m_nLastSingleOverlayZPos = 0;

		uint32_t m_uNextPresentCtx = 0;
		DRMPresentCtx m_PresentCtxs[3];

		bool SupportsColorManagement() const
		{
			return drm_supports_color_mgmt( &g_DRM );
		}

		bool SupportsSRGBtoPQ() const
		{
			return drm_supports_srgb_to_pq( &g_DRM );
		}

		int Commit( const FrameInfo_t *pFrameInfo )
		{
			drm_t *drm = &g_DRM;
			int ret = 0;

			assert( drm->req != nullptr );

			defer( if ( drm->req != nullptr ) { drmModeAtomicFree( drm->req ); drm->req = nullptr; } );

			bool isPageFlip = drm->flags & DRM_MODE_PAGE_FLIP_EVENT;

			if ( isPageFlip )
			{
				++drm->uPendingFlipCount;

				// Do it before the commit, as otherwise the pageflip handler could
				// potentially beat us to the refcount checks.

				// Swap over request FDs -> Queue
				std::unique_lock lock( drm->m_QueuedFbIdsMutex );
				drm->m_QueuedFbIds.swap( drm->m_FbIdsInRequest );
			}

			g_DRM.pConnector->PresentationFeedback().m_uQueuedPresents++;

			uint32_t uCurrentPresentCtx = m_uNextPresentCtx;
			m_uNextPresentCtx = ( m_uNextPresentCtx + 1 ) % 3;
			m_PresentCtxs[uCurrentPresentCtx].ulPendingFlipCount = g_DRM.pConnector->PresentationFeedback().m_uQueuedPresents;
			m_PresentCtxs[uCurrentPresentCtx].frameGenerationOutputId =
				pFrameInfo->frameGenerationOutputId;

			drm_log.debugf("flip commit %" PRIu64, (uint64_t)g_DRM.pConnector->PresentationFeedback().m_uQueuedPresents);
			gpuvis_trace_printf( "flip commit %" PRIu64, (uint64_t)g_DRM.pConnector->PresentationFeedback().m_uQueuedPresents );

			ret = drmModeAtomicCommit(drm->fd, drm->req, drm->flags, &m_PresentCtxs[uCurrentPresentCtx] );
			drm_close_in_fences( drm );
			if ( ret != 0 )
			{
				drm_log.errorf_errno( "flip error" );

				if ( ret != -EBUSY && ret != -EACCES )
				{
					drm_log.errorf( "fatal flip error, aborting" );
					if ( isPageFlip )
						drm->uPendingFlipCount--;
					abort();
				}

				drm_rollback( drm );

				// Swap back over to what was previously queued (probably nothing)
				// if this commit failed.
				{
					std::unique_lock lock( drm->m_QueuedFbIdsMutex );
					drm->m_QueuedFbIds.swap( drm->m_FbIdsInRequest );
				}
				// Clear our refs.
				drm->m_FbIdsInRequest.clear();

				g_DRM.pConnector->PresentationFeedback().m_uQueuedPresents--;

				if ( isPageFlip )
					drm->uPendingFlipCount--;

				return ret;
			} else {
				// Our request went through!
				// Clear what we swapped with (what was previously queued)
				drm->m_FbIdsInRequest.clear();

				drm->current = drm->pending;

				for ( std::unique_ptr< gamescope::CDRMCRTC > &pCRTC : drm->crtcs )
				{
					for ( std::optional<gamescope::CDRMAtomicProperty> &oProperty : pCRTC->GetProperties() )
					{
						if ( oProperty )
							oProperty->OnCommit();
					}
				}

				for ( std::unique_ptr< gamescope::CDRMPlane > &pPlane : drm->planes )
				{
					for ( std::optional<gamescope::CDRMAtomicProperty> &oProperty : pPlane->GetProperties() )
					{
						if ( oProperty )
							oProperty->OnCommit();
					}
				}

				for ( auto &iter : drm->connectors )
				{
					gamescope::CDRMConnector *pConnector = &iter.second;
					for ( std::optional<gamescope::CDRMAtomicProperty> &oProperty : pConnector->GetProperties() )
					{
						if ( oProperty )
							oProperty->OnCommit();
					}
				}

				if (drm->flags & DRM_MODE_ATOMIC_ALLOW_MODESET)
					drm_update_screen_disabled( drm );
			}

			// Update the draw time
			// Ideally this would be updated by something right before the page flip
			// is queued and would end up being the new page flip, rather than here.
			// However, the page flip handler is called when the page flip occurs,
			// not when it is successfully queued.
			if (!m_bAsyncCompositeTiming)
				GetVBlankTimer().UpdateLastDrawTime( get_time_in_nanos() - g_SteamCompMgrVBlankTime.ulWakeupTime );


			return ret;
		}

	};

	/////////////////////////
	// Backend Instantiator
	/////////////////////////

	template <>
	bool IBackend::Set<CDRMBackend>()
	{
		return Set( new CDRMBackend{} );
	}
}

int HackyDRMPresent( const FrameInfo_t *pFrameInfo, bool bAsync )
{
	return static_cast<gamescope::CDRMBackend *>( GetBackend() )->Present( pFrameInfo, bAsync );
}
