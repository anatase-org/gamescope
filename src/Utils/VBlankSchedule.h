#pragma once

#include <cstdint>
#include <algorithm>

namespace gamescope
{
    struct VBlankScheduleTime
    {
        // The expected time for the vblank we want to target.
        uint64_t ulTargetVBlank = 0;
        // The vblank offset by the redzone/scheduling calculation.
        // This is when we want to wake-up by to meet that vblank time above.
        uint64_t ulScheduledWakeupPoint = 0;
        // Refresh period the target was stepped with, zero when unknown.
        uint64_t ulRefreshCycle = 0;
    };

    namespace VBlank
    {
        struct ScheduleParams
        {
            uint64_t ulLastVBlank = 0;
            uint64_t ulInterval = 0;
            uint64_t ulOffset = 0;
            uint64_t ulNow = 0;
            bool bVRR = false;
            bool bRateOverride = false;
        };

        inline uint64_t LastTarget( const VBlankScheduleTime &lastSchedule, uint64_t ulInterval )
        {
            // Do not retain several cycles of old draw lead when the rate increases.
            return std::min( lastSchedule.ulTargetVBlank, lastSchedule.ulScheduledWakeupPoint + ulInterval );
        }

        inline uint64_t TargetFloor( const VBlankScheduleTime &lastSchedule, const ScheduleParams &params )
        {
            // VRR wakes have no fixed phase and can follow a newly ready frame.
            if ( params.bVRR || !lastSchedule.ulTargetVBlank )
                return 0;

            const uint64_t ulSeparation = params.bRateOverride ? params.ulInterval - 1 : params.ulInterval / 2;
            return LastTarget( lastSchedule, params.ulInterval ) + ulSeparation;
        }

        inline VBlankScheduleTime Next( const ScheduleParams &params, const VBlankScheduleTime &lastSchedule )
        {
            uint64_t ulAnchor = params.ulLastVBlank;
            uint64_t ulOffset = params.ulOffset;
            if ( params.bRateOverride && !params.bVRR )
            {
                // An explicit rate owns its cadence, independent of host feedback.
                if ( lastSchedule.ulTargetVBlank )
                    ulAnchor = LastTarget( lastSchedule, params.ulInterval );
                ulOffset = std::min( ulOffset, params.ulInterval );
            }

            const uint64_t ulTargetFloor = TargetFloor( lastSchedule, params );
            uint64_t ulWake = ulAnchor + params.ulInterval - ulOffset;
            uint64_t ulEarliestWake = params.ulNow;
            if ( ulTargetFloor >= ulOffset )
                ulEarliestWake = std::max( ulEarliestWake, ulTargetFloor - ulOffset + 1 );
            // Step in one go, a headless session never marks a vblank so the gap grows with uptime.
            if ( ulWake < ulEarliestWake )
            {
                const uint64_t ulDelta = ulEarliestWake - ulWake;
                ulWake += ( ulDelta / params.ulInterval + ( ulDelta % params.ulInterval != 0 ) ) * params.ulInterval;
            }
            return { .ulTargetVBlank = ulWake + ulOffset, .ulScheduledWakeupPoint = ulWake, .ulRefreshCycle = params.ulInterval };
        }
    }

	constexpr uint64_t NextFrameCallbackTime( uint64_t previous, uint64_t interval,
		uint64_t now, uint64_t refreshInterval, uint64_t earlyAllowance )
	{
		const uint64_t schedule = previous + interval;
		return now + earlyAllowance >= schedule && now < schedule + refreshInterval
			? schedule : now;
	}

	constexpr uint64_t VBlankIndexAdvance( uint64_t previous, uint64_t target, uint64_t interval )
	{
		return target > previous ? ( target - previous + interval / 2 ) / interval : 0;
	}

	constexpr bool CrossedVBlankBoundary( uint64_t previous, uint64_t current, uint64_t divisor )
	{
		return current / divisor > previous / divisor;
	}

	constexpr uint64_t NextVBlankWakeup( uint64_t lastVBlank, uint64_t interval,
		uint64_t offset, uint64_t processedVBlank, uint64_t now, uint64_t lateMargin )
	{
		uint64_t target = lastVBlank + interval - offset;
		// Do not repeat a consumed refresh, even if its predicted timestamp moved.
		const uint64_t consumed = processedVBlank + interval / 2;
		if ( target + offset <= consumed )
			target += ( ( consumed - ( target + offset ) ) / interval + 1 ) * interval;

		// Catch up in one step, retaining the allowance for a late unprocessed refresh.
		if ( target < now && now - target > lateMargin )
			target += ( ( now - target - lateMargin - 1 ) / interval + 1 ) * interval;
		return target;
	}
}
