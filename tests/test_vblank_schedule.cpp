#include "Utils/VBlankSchedule.h"

#include <cassert>
#include <cstdint>
#include <initializer_list>

int main()
{
	using gamescope::NextVBlankWakeup;
	using gamescope::VBlankIndexAdvance;
	using gamescope::CrossedVBlankBoundary;
	assert( VBlankIndexAdvance( 1000, 1000, 100 ) == 0 );
	assert( VBlankIndexAdvance( 1000, 900, 100 ) == 0 );
	assert( VBlankIndexAdvance( 1000, 1101, 100 ) == 1 );
	assert( VBlankIndexAdvance( 1000, 1500, 100 ) == 5 );
	assert( !CrossedVBlankBoundary( 2, 2, 2 ) );
	assert( !CrossedVBlankBoundary( 2, 3, 2 ) );
	assert( CrossedVBlankBoundary( 1, 3, 2 ) );
	assert( CrossedVBlankBoundary( 2, 7, 2 ) );
	// Preserve the existing duplicate-refresh guard and late-frame allowance.
	for ( uint64_t offset : { 0u, 100u, 400u } )
	for ( uint64_t processed : { 0u, 1000u, 2000u, 2100u, 2500u } )
	for ( uint64_t now = 1000; now < 8000; now += 37 )
	for ( uint64_t margin : { 0u, 200u } )
	{
		uint64_t expected = 2000 - offset;
		while ( expected + offset <= processed + 500 ||
			( expected < now && now - expected > margin ) )
			expected += 1000;
		assert( NextVBlankWakeup( 1000, 1000, offset, processed, now, margin ) == expected );
	}

	constexpr uint64_t interval = 16'666'667;
	constexpr uint64_t year = 365ull * 24 * 60 * 60 * 1'000'000'000;
	const uint64_t wakeup = NextVBlankWakeup( 0, interval, 1000, 0, year, 0 );
	assert( wakeup >= year && wakeup - year < interval );
	assert( ( wakeup + 1000 ) % interval == 0 );
	assert( NextVBlankWakeup( 1000, 1000, 100, 0, 2100, 200 ) == 1900 );
	assert( NextVBlankWakeup( 1000, 1000, 100, 0, 2101, 200 ) == 2900 );
}
