"""Scoring engine: pick the single best prior run to seed the next agent run.

Two stages, kept deliberately separate:

  1. retrieve — `$vectorSearch` over `task_embedding` narrows to runs about the same task
  2. score    — rank those candidates on their own run metrics; highest score wins

Similarity is an *input* to the score, not the score itself. The nearest neighbour in
embedding space can still be a bad teacher: it may have flailed for 22 steps, half its
actions may have failed, or it may be old enough that its selectors have rotted. So we
retrieve on similarity and rank on quality.

Two properties worth knowing before changing anything here:

* **Efficiency is normalized within the candidate set, never globally.** Nine steps is
  excellent for a checkout flow and terrible for a search. Min-maxing across the ~25
  retrieved candidates re-calibrates per task for free, which is why the weights below
  don't need retuning when new kinds of task show up.
* **Returning nothing is a valid answer.** A wrong trace is worse than no trace — it
  steers the agent down a dead end it would never have found alone. Hence the floors.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from bson import ObjectId
from pymongo.errors import OperationFailure

from context_layer.config import VECTOR_INDEX_NAME
from context_layer.db import get_runs_collection, get_steps_collection
from context_layer.embeddings import embed_text_safe

# ---- retrieval knobs ----
NUM_CANDIDATES = 200  # how deep $vectorSearch looks before narrowing
CANDIDATE_LIMIT = 25  # the set the score is normalized across
MAX_AGE_DAYS = 90  # older than this, selectors have usually rotted

# ---- acceptance floors: below either of these we run cold ----
MIN_SIMILARITY = 0.85  # Atlas maps cosine to (1+cos)/2, so this is cos ≈ 0.7
MIN_SCORE = 0.45

# ---- scoring ----
RECENCY_HALF_LIFE_DAYS = 14.0
UNKNOWN_CLEAN = 0.5  # runs written before action counts existed: unknown, not perfect
EPSILON = 1e-9

WEIGHTS: dict[str, float] = {
	"sim": 0.35,
	"eff_steps": 0.25,
	"eff_time": 0.15,
	"eff_tokens": 0.10,
	"clean": 0.10,
	"recency": 0.05,
}

# Used when we rank without a vector search (no embedding, or no index yet). Constant
# across candidates, so it cancels out and the metrics decide.
FALLBACK_SIM = 0.9

_UNBOUNDED = {"documents": ["unbounded", "unbounded"]}
_vector_search_warned = False


# --------------------------------------------------------------------------- pipeline


def _cheaper_is_better(value: str, low: str, high: str) -> dict:
	"""Min-max normalize within the candidate set: 1 = cheapest in the set, 0 = dearest."""
	return {
		"$subtract": [
			1,
			{
				"$divide": [
					{"$subtract": [value, low]},
					{"$add": [{"$subtract": [high, low]}, EPSILON]},
				]
			},
		]
	}


def vector_stages(query_vector: list[float], *, domain: str | None = None) -> list[dict]:
	"""Stage 1: the candidate set. Filters are hard gates, not weighted terms.

	`verified` rather than `success`: success is the agent's own claim, so seeding from
	it would teach us the confident failures too.
	"""
	search_filter: dict[str, Any] = {
		"verified": True,
		"created_at": {"$gte": datetime.utcnow() - timedelta(days=MAX_AGE_DAYS)},
	}
	if domain:
		search_filter["domain"] = domain

	return [
		{
			"$vectorSearch": {
				"index": VECTOR_INDEX_NAME,
				"path": "task_embedding",
				"queryVector": query_vector,
				"numCandidates": NUM_CANDIDATES,
				"limit": CANDIDATE_LIMIT,
				"filter": search_filter,
			}
		},
		# $meta must be read close to the search stage, so `sim` is materialized here and
		# every stage downstream just reads a plain field.
		{"$addFields": {"sim": {"$meta": "vectorSearchScore"}}},
	]


def exact_match_stages(task_text: str, *, domain: str | None = None) -> list[dict]:
	"""Stage 1, fallback flavour: same task text, no vector index required.

	Only used when the task couldn't be embedded or the vector index isn't there yet.
	Narrower than vector search by design — better to find nothing than to rank runs
	we have no similarity evidence for.
	"""
	match: dict[str, Any] = {
		"task_text": task_text,
		"verified": True,
		"created_at": {"$gte": datetime.utcnow() - timedelta(days=MAX_AGE_DAYS)},
	}
	if domain:
		match["domain"] = domain
	return [
		{"$match": match},
		{"$addFields": {"sim": FALLBACK_SIM}},
		{"$limit": CANDIDATE_LIMIT},
	]


def scoring_stages(weights: dict[str, float] | None = None) -> list[dict]:
	"""Stage 2: score every candidate and sort. Expects a `sim` field to already exist."""
	weights = weights or WEIGHTS

	return [
		# Min/max over the whole retrieved set — this is what makes efficiency
		# comparable between a 3-step search and a 15-step checkout.
		{
			"$setWindowFields": {
				"sortBy": {"sim": -1},
				"output": {
					"_min_steps": {"$min": "$total_steps", "window": _UNBOUNDED},
					"_max_steps": {"$max": "$total_steps", "window": _UNBOUNDED},
					"_min_ms": {"$min": "$duration_ms", "window": _UNBOUNDED},
					"_max_ms": {"$max": "$duration_ms", "window": _UNBOUNDED},
					"_min_tokens": {"$min": "$total_tokens", "window": _UNBOUNDED},
					"_max_tokens": {"$max": "$total_tokens", "window": _UNBOUNDED},
				},
			}
		},
		{
			"$addFields": {
				"components": {
					"sim": "$sim",
					"eff_steps": _cheaper_is_better("$total_steps", "$_min_steps", "$_max_steps"),
					"eff_time": _cheaper_is_better("$duration_ms", "$_min_ms", "$_max_ms"),
					"eff_tokens": _cheaper_is_better("$total_tokens", "$_min_tokens", "$_max_tokens"),
					# The trace goes verbatim into the next prompt, so failed actions cost
					# tokens and mislead. Runs predating the counters score neutral.
					"clean": {
						"$cond": [
							{"$gt": [{"$ifNull": ["$action_count", 0]}, 0]},
							{
								"$subtract": [
									1,
									{"$divide": [{"$ifNull": ["$failed_action_count", 0]}, "$action_count"]},
								]
							},
							UNKNOWN_CLEAN,
						]
					},
					"recency": {
						"$pow": [
							0.5,
							{
								"$divide": [
									{
										"$dateDiff": {
											"startDate": {"$ifNull": ["$created_at", "$$NOW"]},
											"endDate": "$$NOW",
											"unit": "day",
										}
									},
									RECENCY_HALF_LIFE_DAYS,
								]
							},
						]
					},
				}
			}
		},
		{"$addFields": {"score": {"$add": [{"$multiply": [w, f"$components.{k}"]} for k, w in weights.items()]}}},
		{"$sort": {"score": -1}},
		# The embedding is 1536 floats per candidate and nothing downstream reads it.
		{
			"$project": {
				"task_embedding": 0,
				"_min_steps": 0,
				"_max_steps": 0,
				"_min_ms": 0,
				"_max_ms": 0,
				"_min_tokens": 0,
				"_max_tokens": 0,
			}
		},
	]


def build_pipeline(query_vector: list[float], *, domain: str | None = None, limit: int = CANDIDATE_LIMIT) -> list[dict]:
	"""The whole engine as one aggregation: retrieve, score, sort."""
	return [*vector_stages(query_vector, domain=domain), *scoring_stages(), {"$limit": limit}]


# --------------------------------------------------------------------------- selection


@dataclass
class RunSelection:
	"""A scored candidate, with the components kept so the choice can be explained."""

	run_id: ObjectId
	task_text: str
	score: float
	components: dict[str, float]
	run: dict

	@property
	def similarity(self) -> float:
		return self.components.get("sim", 0.0)

	def explain(self) -> str:
		parts = ", ".join(f"{k} {v:.2f}" for k, v in self.components.items())
		return (
			f'run {self.run_id} "{self.task_text}" — score {self.score:.3f} '
			f"({parts}) | {self.run.get('total_steps')} steps, "
			f"{(self.run.get('duration_ms') or 0) / 1000:.1f}s, {self.run.get('model')}"
		)


async def rank_candidates(
	task_text: str,
	*,
	domain: str | None = None,
	query_vector: list[float] | None = None,
	limit: int = CANDIDATE_LIMIT,
) -> list[RunSelection]:
	"""Retrieve similar runs and score them, best first. No floors applied — see select_best_run."""
	global _vector_search_warned

	if query_vector is None:
		query_vector = await embed_text_safe(task_text)

	if query_vector is not None:
		pipeline = build_pipeline(query_vector, domain=domain, limit=limit)
		try:
			cursor = await get_runs_collection().aggregate(pipeline)
			docs = await cursor.to_list(length=limit)
			return [_to_selection(d) for d in docs]
		except OperationFailure as exc:
			if not _vector_search_warned:
				print(
					f"[context_layer] $vectorSearch unavailable ({exc}); falling back to exact task match. "
					"Run `python -m context_layer.scripts.setup_indexes` to create the vector index."
				)
				_vector_search_warned = True

	pipeline = [*exact_match_stages(task_text, domain=domain), *scoring_stages(), {"$limit": limit}]
	cursor = await get_runs_collection().aggregate(pipeline)
	docs = await cursor.to_list(length=limit)
	return [_to_selection(d) for d in docs]


def _to_selection(doc: dict) -> RunSelection:
	return RunSelection(
		run_id=doc["_id"],
		task_text=doc.get("task_text", ""),
		score=doc.get("score", 0.0),
		components=doc.get("components", {}),
		run=doc,
	)


async def select_best_run(
	task_text: str,
	*,
	domain: str | None = None,
	query_vector: list[float] | None = None,
	min_similarity: float = MIN_SIMILARITY,
	min_score: float = MIN_SCORE,
) -> RunSelection | None:
	"""The entry point: the one run worth seeding the next run with, or None to run cold.

	None is a real outcome, not an error. If nothing clears the floors, the caller should
	run without memory rather than inject the closest thing it could find.
	"""
	candidates = await rank_candidates(task_text, domain=domain, query_vector=query_vector)
	if not candidates:
		return None

	best = candidates[0]
	if best.similarity < min_similarity or best.score < min_score:
		return None
	return best


# ----------------------------------------------------------------------------- context


MAX_CONTEXT_STEPS = 25
MAX_RESULT_CHARS = 120
MAX_PARAM_CHARS = 120  # a `done` action carries the whole answer; that isn't route information
MAX_REASON_CHARS = 200  # the judge's verdict can run to a paragraph


async def load_steps(run_id: ObjectId) -> list[dict]:
	cursor = get_steps_collection().find({"run_id": run_id}).sort("i", 1)
	return await cursor.to_list(length=MAX_CONTEXT_STEPS)


def build_memory_context(selection: RunSelection, steps: list[dict]) -> str:
	"""Render the winning run as a prompt block for the next agent.

	Compacted on purpose: the goal and the element identity transfer between runs, but
	`index` params don't (they're a positional bet on a DOM that has since re-rendered)
	and neither do result blobs. Sending them wastes tokens and invites the agent to
	trust a stale coordinate.
	"""
	run = selection.run
	reason = (run.get("verify_reason") or "assertion passed").strip()
	if len(reason) > MAX_REASON_CHARS:
		reason = reason[:MAX_REASON_CHARS].rstrip() + "..."
	header = [
		"## Prior verified run for a similar task",
		f'task: "{selection.task_text}"',
		f"outcome: verified in {run.get('total_steps')} steps, {(run.get('duration_ms') or 0) / 1000:.1f}s — {reason}",
		"",
	]

	lines: list[str] = []
	for step in steps[:MAX_CONTEXT_STEPS]:
		lines.append(f"step {step.get('i')} — {step.get('url_before') or ''}")
		if step.get("reasoning"):
			lines.append(f"  goal: {step['reasoning']}")
		for action in step.get("actions") or []:
			params = {
				k: (v[:MAX_PARAM_CHARS] + "..." if isinstance(v, str) and len(v) > MAX_PARAM_CHARS else v)
				for k, v in (action.get("params") or {}).items()
				if k != "index"
			}
			element = action.get("element") or {}
			target = ""
			if element.get("text") or element.get("tag"):
				target = f" → <{element.get('tag') or '?'}> {element.get('text') or ''}".rstrip()
			if element.get("xpath"):
				target += f"  xpath={element['xpath']}"
			flag = "" if action.get("ok", True) else "  [FAILED]"
			lines.append(f"  {action.get('name')} {params}{target}{flag}")
			result = action.get("result")
			if result:
				lines.append(f"    → {str(result)[:MAX_RESULT_CHARS]}")

	footer = [
		"",
		"Follow this route where it still applies. Re-locate each element on the live page —",
		"the page may have changed since, so treat the trace as a map, not as coordinates.",
	]
	return "\n".join([*header, *lines, *footer])


async def get_memory_for_task(
	task_text: str,
	*,
	domain: str | None = None,
) -> tuple[str, ObjectId] | None:
	"""One-call convenience for the agent wrapper: the context block and the run it came from.

	The returned ObjectId is what belongs in `memory_source_run_id` on the new run, which
	is also what lets a later version of this scorer learn which memories actually helped.
	"""
	selection = await select_best_run(task_text, domain=domain)
	if selection is None:
		return None
	steps = await load_steps(selection.run_id)
	if not steps:
		return None
	return build_memory_context(selection, steps), selection.run_id
