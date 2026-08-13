"""Show the scoring engine choosing a memory for a task, with its reasoning.

Run with: python -m context_layer.scripts.demo_select "add bananas to cart"

Prints every candidate with its score components, then the winner: the plan items that
would be seeded into the agent's `<plan>` block, and the long-form context block.
"""

import asyncio
import sys

from context_layer.scorer import (
	MIN_SCORE,
	MIN_SIMILARITY,
	build_memory_context,
	build_plan_items,
	load_steps,
	rank_candidates,
)


async def main(task_text: str, domain: str | None) -> None:
	candidates = await rank_candidates(task_text, domain=domain)
	if not candidates:
		print(f'no verified prior runs matched "{task_text}" — this run would go cold')
		return

	print(f'{len(candidates)} candidate(s) for "{task_text}":\n')
	for rank, candidate in enumerate(candidates, start=1):
		print(f"  {rank}. {candidate.explain()}")

	best = candidates[0]
	print()
	if best.similarity < MIN_SIMILARITY:
		print(f"rejected: similarity {best.similarity:.3f} < {MIN_SIMILARITY} — running cold beats a wrong trace")
		return
	if best.score < MIN_SCORE:
		print(f"rejected: score {best.score:.3f} < {MIN_SCORE} — running cold beats a weak trace")
		return

	steps = await load_steps(best.run_id)
	items = build_plan_items(steps)
	print(f"selected run {best.run_id} (score {best.score:.3f})")
	print(f"\nseeded <plan> ({len(items)} of {len(steps)} steps survive the filters):\n")
	for i, text in enumerate(items):
		print(f"  {'[>]' if i == 0 else '[ ]'} {i}: {text}")
	print("\nlong-form context block (not injected by default):\n")
	print(build_memory_context(best, steps))


if __name__ == "__main__":
	task = sys.argv[1] if len(sys.argv) > 1 else "add bananas to cart"
	site = sys.argv[2] if len(sys.argv) > 2 else None
	asyncio.run(main(task, site))
