"""Back-fill the fields the scoring engine needs onto runs written before it existed.

A run with no embedding is invisible to `$vectorSearch`, so it can never be selected as
memory no matter how good it was; a run with no action counts scores neutral on `clean`.

Also repairs `actions.ok`, which earlier runs recorded from `ActionResult.success` — a
field browser-use keeps None on every non-done action, so ordinary successful actions
were stored as failures.

Idempotent: re-embeds nothing that already has an embedding. Run with:
    python -m context_layer.scripts.backfill_embeddings
"""

import asyncio

from context_layer.db import get_runs_collection, get_steps_collection
from context_layer.embeddings import embed_text
from context_layer.schema import first_domain, model_tier


async def _derive_from_steps(run_id) -> dict:
	"""Recompute the denormalized scoring fields from the run's steps."""
	steps = await get_steps_collection().find({"run_id": run_id}).sort("i", 1).to_list(length=None)
	actions = [a for s in steps for a in (s.get("actions") or [])]
	fields: dict = {
		"action_count": len(actions),
		"failed_action_count": sum(1 for a in actions if not a.get("ok", True)),
	}
	if steps:
		fields["domain"] = first_domain(url for s in steps for url in (s.get("url_before"), s.get("url_after")))
	return fields


async def repair_action_ok() -> None:
	"""Recompute every stored action's `ok` from whether it recorded an error.

	Has to run before the counts below, or `failed_action_count` inherits the old bug.
	"""
	result = await get_steps_collection().update_many(
		{},
		[
			{
				"$set": {
					"actions": {
						"$map": {
							"input": "$actions",
							"as": "a",
							"in": {"$mergeObjects": ["$$a", {"ok": {"$eq": [{"$ifNull": ["$$a.error", None]}, None]}}]},
						}
					}
				}
			}
		],
	)
	print(f"repaired actions.ok on {result.modified_count} step(s)")


async def main() -> None:
	await repair_action_ok()

	runs = get_runs_collection()
	pending = await runs.find({}).to_list(length=None)
	print(f"{len(pending)} run(s) to refresh")

	for run in pending:
		task_text = run.get("task_text")
		if not task_text:
			print(f"  skipped {run['_id']}: no task_text")
			continue

		update = await _derive_from_steps(run["_id"])
		if run.get("task_embedding") is None:  # don't pay to re-embed text that hasn't changed
			update["task_embedding"] = await embed_text(task_text)
		if run.get("model_tier") is None and run.get("model"):
			update["model_tier"] = model_tier(run["model"])

		await runs.update_one({"_id": run["_id"]}, {"$set": update})
		print(
			f'  refreshed {run["_id"]} domain={update.get("domain")} '
			f'actions={update["action_count"]} failed={update["failed_action_count"]}'
		)


if __name__ == "__main__":
	asyncio.run(main())
