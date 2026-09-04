#pragma once

#include <cstdint>

namespace gamescope
{
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
