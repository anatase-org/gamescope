#include "SurfaceDamage.h"

#include <cassert>

int main()
{
	SurfaceDamageHistory history;
	assert(history.Between(0, 0)->width == 0);
	assert(!history.Between(0, 1));
	assert(history.Append({10, 20, 5, 8}) == 1);
	assert(history.Append({20, 15, 7, 4}) == 2);
	assert(history.Append({}) == 3);
	const auto accumulated = history.Between(0, 3);
	assert(accumulated && accumulated->x == 10 && accumulated->y == 15);
	assert(accumulated->width == 17 && accumulated->height == 13);
	const auto olderOutput = history.Between(1, 3);
	assert(olderOutput && olderOutput->x == 20 && olderOutput->y == 15);
	assert(olderOutput->width == 7 && olderOutput->height == 4);
	assert(history.Between(2, 3)->width == 0);
	assert(!history.Between(3, 2));

	// Buffer age may span skipped client commits, but stale output contents
	// must fall back to a full repaint once the bounded history rolls over.
	for (int i = 4; i <= 40; ++i)
		history.Append({i, i, 1, 1});
	assert(!history.Between(0, 40));
	assert(!history.Between(1, 3));
	const auto recent = history.Between(37, 40);
	assert(recent && recent->x == 38 && recent->y == 38);
	assert(recent->width == 3 && recent->height == 3);
	assert(history.Between(40, 40)->width == 0);
}
