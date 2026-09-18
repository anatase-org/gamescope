#include <catch2/catch_test_macros.hpp>
#include "Utils/VBlankSchedule.h"
using namespace gamescope;
constexpr VBlankScheduleTime consumed{ 1'004'166'667, 1'000'000'000 };

TEST_CASE("Explicit cadence does not follow a faster host feedback grid", "[vblank_schedule]") {
	const VBlankScheduleTime previous{ 16'666'667, 14'666'667 };
	const auto next = VBlank::Next({
		.ulLastVBlank = 12'500'001,
		.ulInterval = 8'333'333,
		.ulOffset = 2'000'000,
		.ulNow = 14'676'667,
		.bRateOverride = true,
	}, previous);
	REQUIRE(next.ulTargetVBlank == 25'000'000);
	REQUIRE(next.ulScheduledWakeupPoint == 23'000'000);
}

TEST_CASE("Explicit cadence stays regular despite late or jittered feedback", "[vblank_schedule]") {
	VBlankScheduleTime previous{ 1'004'166'667, 1'000'000'000 };
	VBlank::ScheduleParams params{
		.ulInterval = 4'166'667,
		.ulOffset = 2'000'000,
		.bRateOverride = true,
	};
	for ( uint64_t feedback : { 1'000'020'000ul, 999'980'000ul, 1'008'340'000ul } )
	{
		params.ulLastVBlank = feedback;
		params.ulNow = previous.ulScheduledWakeupPoint + 10'000;
		auto next = VBlank::Next(params, previous);
		REQUIRE(next.ulTargetVBlank - previous.ulTargetVBlank == 4'166'667);
		previous = next;
	}

	params.ulNow = 2'000'000'000;
	const auto resumed = VBlank::Next(params, previous);
	REQUIRE(resumed.ulScheduledWakeupPoint >= params.ulNow);
	REQUIRE(resumed.ulScheduledWakeupPoint < params.ulNow + 4'166'667);
	REQUIRE((resumed.ulTargetVBlank - previous.ulTargetVBlank) % 4'166'667 == 0);
}

TEST_CASE("An explicit rate needs a full interval before another queued nudge", "[vblank_schedule]") {
	const VBlank::ScheduleParams params{ .ulInterval = 4'166'667, .bRateOverride = true };
	REQUIRE(VBlank::TargetFloor(consumed, params) == 1'008'333'333);
}

TEST_CASE("An explicit rate change clamps old draw lead without wedging", "[vblank_schedule]") {
	const VBlankScheduleTime previous{ 1'016'666'667, 1'000'000'000 };
	const auto next = VBlank::Next({
		.ulLastVBlank = 1'000'000'000,
		.ulInterval = 4'166'667,
		.ulOffset = 4'166'667,
		.ulNow = 1'000'010'000,
		.bRateOverride = true,
	}, previous);
	REQUIRE(next.ulTargetVBlank == 1'008'333'334);
	REQUIRE(next.ulScheduledWakeupPoint == 1'004'166'667);
}

TEST_CASE("Explicit rate selection leaves VRR feedback scheduling intact", "[vblank_schedule]") {
	const auto next = VBlank::Next({
		.ulLastVBlank = 1'000'020'000,
		.ulInterval = 4'166'667,
		.ulOffset = 300'000,
		.ulNow = 1'003'876'667,
		.bVRR = true,
		.bRateOverride = true,
	}, { 1'004'166'667, 1'003'866'667 });
	REQUIRE(next.ulTargetVBlank == 1'004'186'667);
}

TEST_CASE("An explicit cadence starts from feedback before its first accepted wake", "[vblank_schedule]") {
	const auto next = VBlank::Next({
		.ulLastVBlank = 1'000'000'000,
		.ulInterval = 8'333'333,
		.ulOffset = 2'000'000,
		.ulNow = 1'001'000'000,
		.bRateOverride = true,
	}, {});
	REQUIRE(next.ulTargetVBlank == 1'008'333'333);
	REQUIRE(next.ulScheduledWakeupPoint == 1'006'333'333);
}

TEST_CASE("Slowing an explicit cadence waits the new interval", "[vblank_schedule]") {
	const auto next = VBlank::Next({
		.ulLastVBlank = 1'002'000'000,
		.ulInterval = 16'666'667,
		.ulOffset = 2'000'000,
		.ulNow = 1'002'000'000,
		.bRateOverride = true,
	}, consumed);
	REQUIRE(next.ulTargetVBlank == 1'020'833'334);
}

TEST_CASE("Leaving VRR resumes explicit cadence without replaying missed slots", "[vblank_schedule]") {
	const auto next = VBlank::Next({
		.ulLastVBlank = 3'599'999'000'000,
		.ulInterval = 10'000'000,
		.ulOffset = 2'000'000,
		.ulNow = 3'600'000'000'000,
		.bRateOverride = true,
	}, { 20'000'000, 18'000'000 });
	REQUIRE(next.ulTargetVBlank == 3'600'010'000'000);
	REQUIRE(next.ulScheduledWakeupPoint == 3'600'008'000'000);
}

TEST_CASE("High nested rates bound a draw lead longer than one interval", "[vblank_schedule]") {
	const auto next = VBlank::Next({
		.ulLastVBlank = 1'000'000'000,
		.ulInterval = 1'000'000,
		.ulOffset = 1'650'000,
		.ulNow = 1'000'000'000,
		.bRateOverride = true,
	}, {});
	REQUIRE(next.ulTargetVBlank == 1'001'000'000);
	REQUIRE(next.ulScheduledWakeupPoint == 1'000'000'000);
}
