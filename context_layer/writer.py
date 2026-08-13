"""Write operations: create/update run and step documents. No read/query logic here."""

from dataclasses import asdict
from datetime import datetime

from bson import ObjectId

from context_layer.db import get_runs_collection, get_steps_collection
from context_layer.embeddings import embed_text_safe
from context_layer.schema import ActionEntry, RunDoc, StepDoc, first_domain, model_tier


async def create_run(
	task_text: str,
	model: str,
	*,
	used_memory: bool = False,
	memory_source_run_id: ObjectId | None = None,
	task_params: dict | None = None,
) -> ObjectId:
	"""Insert a new run document (status is implicit: no ended_at yet). Returns the run's _id."""
	run = RunDoc(
		task_text=task_text,
		model=model,
		task_params=task_params,
		task_embedding=await embed_text_safe(task_text),
		model_tier=model_tier(model),
		used_memory=used_memory,
		memory_source_run_id=memory_source_run_id,
	)
	result = await get_runs_collection().insert_one(asdict(run))
	return result.inserted_id


async def add_step(
	run_id: ObjectId,
	i: int,
	url_before: str,
	url_after: str,
	reasoning: str,
	ms: float,
	actions: list[ActionEntry],
) -> ObjectId:
	"""Insert one step document (with its actions) for a given run. Returns the step's _id."""
	step = StepDoc(
		run_id=run_id,
		i=i,
		url_before=url_before,
		url_after=url_after,
		reasoning=reasoning,
		ms=ms,
		actions=actions,
	)
	result = await get_steps_collection().insert_one(asdict(step))
	return result.inserted_id


def _derive_scoring_fields(run: RunDoc, steps: list[StepDoc]) -> None:
	"""Fill in the fields the scorer reads off the run doc directly.

	`domain` and the action counts all live in the steps, but recomputing them at query
	time would mean a $lookup into `steps` for every candidate on every selection. They're
	cheap to denormalize here, once, while everything is already in memory.
	"""
	if run.domain is None and steps:
		run.domain = first_domain(url for s in steps for url in (s.url_before, s.url_after))
	run.action_count = sum(len(s.actions) for s in steps)
	run.failed_action_count = sum(1 for s in steps for a in s.actions if not a.ok)


async def write_run(run_id: ObjectId, run: RunDoc, steps: list[StepDoc]) -> None:
	"""Batch-write a finished run and all its steps in one shot.

	Called once, after the agent run has ended — never per-step — so a run only ever
	costs two round-trips to Mongo (one insert_one, one insert_many) regardless of step count.
	"""
	_derive_scoring_fields(run, steps)
	if run.task_embedding is None:
		run.task_embedding = await embed_text_safe(run.task_text)
	await get_runs_collection().insert_one({**asdict(run), "_id": run_id})
	if steps:
		await get_steps_collection().insert_many([asdict(s) for s in steps])


async def finalize_run(
	run_id: ObjectId,
	*,
	success: bool | None,
	verified: bool | None,
	verify_reason: str | None,
	total_steps: int,
	duration_ms: float,
	llm_calls: int,
	total_tokens: int,
	est_cost_usd: float,
	ended_at: datetime | None = None,
	domain: str | None = None,
	action_count: int | None = None,
	failed_action_count: int | None = None,
) -> None:
	"""Update a run document with its outcome/metrics once the agent run has finished.

	`domain` / the action counts are optional here because this incremental path never
	sees the steps in one place; pass them if you want the run to be scoreable. The
	batch path (`write_run`) derives them for you.
	"""
	scoring_fields = {
		key: value
		for key, value in [
			("domain", domain),
			("action_count", action_count),
			("failed_action_count", failed_action_count),
		]
		if value is not None
	}
	await get_runs_collection().update_one(
		{"_id": run_id},
		{
			"$set": {
				**scoring_fields,
				"success": success,
				"verified": verified,
				"verify_reason": verify_reason,
				"total_steps": total_steps,
				"duration_ms": duration_ms,
				"llm_calls": llm_calls,
				"total_tokens": total_tokens,
				"est_cost_usd": est_cost_usd,
				"ended_at": ended_at or datetime.utcnow(),
			}
		},
	)
