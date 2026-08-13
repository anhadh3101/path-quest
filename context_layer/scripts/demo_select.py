"""Show the scoring engine choosing a memory for a task, with its reasoning.

Run with: python -m context_layer.scripts.demo_select "add bananas to cart"

Prints every candidate with its score components, then the winner and the exact context
block that would be injected into the next agent run.
"""

import asyncio
import sys

from context_layer.scorer import (
	MIN_SCORE,
	MIN_SIMILARITY,
	build_memory_context,
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
	print(f"selected run {best.run_id} (score {best.score:.3f}); context block:\n")
	print(build_memory_context(best, steps))


if __name__ == "__main__":
	task = sys.argv[1] if len(sys.argv) > 1 else "add bananas to cart"
	site = sys.argv[2] if len(sys.argv) > 2 else None
	asyncio.run(main(task, site))
