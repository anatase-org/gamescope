#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <wlr/util/box.h>

// Damage describes changes to surface contents, independently of the client's
// buffer pool. Keep enough history to repaint an older compositor output safely.
class SurfaceDamageHistory
{
public:
	uint64_t Append(wlr_box damage)
	{
		std::lock_guard lock(m_mutex);
		const uint64_t sequence = ++m_sequence;
		m_history[sequence % m_history.size()] = {sequence, damage};
		return sequence;
	}

	std::optional<wlr_box> Between(uint64_t old, uint64_t current)
	{
		std::lock_guard lock(m_mutex);
		if (current < old || current > m_sequence || current - old > m_history.size())
			return std::nullopt;
		wlr_box result{};
		for (uint64_t sequence = old + 1; sequence <= current; ++sequence)
		{
			const auto &[serial, box] = m_history[sequence % m_history.size()];
			if (serial != sequence)
				return std::nullopt;
			if (box.width <= 0 || box.height <= 0)
				continue;
			if (result.width == 0 || result.height == 0)
				result = box;
			else
			{
				const int right = std::max(result.x + result.width, box.x + box.width);
				const int bottom = std::max(result.y + result.height, box.y + box.height);
				result.x = std::min(result.x, box.x);
				result.y = std::min(result.y, box.y);
				result.width = right - result.x;
				result.height = bottom - result.y;
			}
		}
		return result;
	}

private:
	std::mutex m_mutex;
	uint64_t m_sequence = 0;
	std::array<std::pair<uint64_t, wlr_box>, 32> m_history{};
};

struct SurfaceDamage
{
	std::shared_ptr<SurfaceDamageHistory> history;
	uint64_t sequence = 0;
	uint64_t texture = 0;
	bool operator==(const SurfaceDamage &) const = default;
};
