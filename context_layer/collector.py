"""In-memory buffer for one browser-use agent run.

Steps are appended to memory as the run progresses (no I/O). Once the run finishes,
everything is written to MongoDB in a single batch via writer.write_run.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from bson import ObjectId

from context_layer.embeddings import embed_text_safe
from context_layer.schema import ActionEntry, RunDoc, StepDoc, model_tier
from context_layer.scorer import build_memory_context, load_steps, select_best_run
from context_layer.writer import write_run


def _build_step_doc(run_id: ObjectId, i: int, item: Any) -> StepDoc:
	model_output = item.model_output
	reasoning = ""
	if model_output is not None:
		reasoning = model_output.next_goal or model_output.thinking or ""

	ms = 0.0
	if item.metadata is not None:
		ms = item.metadata.duration_seconds * 1000

	url = item.state.url if item.state else ""

	actions: list[ActionEntry] = []
	if model_output is not None:
		results = item.result or []
		interacted_elements = (item.state.interacted_element if item.state else None) or []
		for idx, action in enumerate(model_output.action):
			result = results[idx] if idx < len(results) else None
			dom_el = interacted_elements[idx] if idx < len(interacted_elements) else None

			element = None
			if dom_el is not None:
				element = {
					"xpath": dom_el.x_path,
					"css": None,  # no css selector exists anywhere in browser-use
					"text": dom_el.ax_name,
					"tag": dom_el.node_name,
					"attrs": dom_el.attributes or {},
				}

			action_dict = action.model_dump(exclude_unset=True)
			name = next(iter(action_dict), "unknown")
			params = action_dict.get(name, {})

			extracted = getattr(result, "extracted_content", None)
			result_str = str(extracted)[:500] if extracted is not None else None

			actions.append(
				ActionEntry(
					name=name,
					params=params,
					element=element,
					# browser-use forbids success=True on non-done actions (agent/views.py:341),
					# so `success` is None for every ordinary action — absence of an error is
					# the only honest signal here, and the scorer's `clean` term depends on it.
					ok=getattr(result, "error", None) is None if result is not None else True,
					result=result_str,
					error=getattr(result, "error", None) if result is not None else None,
				)
			)

	return StepDoc(
		run_id=run_id,
		i=i,
		url_before=url,
		url_after=url,  # back-filled onto the previous step by RunCollector.on_step_end
		reasoning=reasoning,
		ms=ms,
		actions=actions,
	)


class RunCollector:
	"""Buffers one run's steps in memory; writes everything to Mongo once on completion."""

	def __init__(
		self,
		task_text: str,
		model: str,
		*,
		used_memory: bool = False,
		memory_source_run_id: ObjectId | None = None,
		task_params: dict | None = None,
		task_embedding: list[float] | None = None,
	):
		self.run_id = ObjectId()
		self._run = RunDoc(
			task_text=task_text,
			model=model,
			model_tier=model_tier(model),
			used_memory=used_memory,
			memory_source_run_id=memory_source_run_id,
			task_params=task_params,
			# Passed in by prepare_run, which already embedded this text to search with.
			# Left None, write_run embeds at save time instead — same result, one extra call.
			task_embedding=task_embedding,
		)
		self._steps: list[StepDoc] = []

	async def on_step_end(self, agent: Any) -> None:
		item = agent.history.history[-1]
		step = _build_step_doc(self.run_id, len(self._steps) + 1, item)
		if self._steps:
			self._steps[-1].url_after = step.url_before
		self._steps.append(step)

	async def on_done(self, history: Any) -> None:
		judgement = history.judgement()
		self._run.success = history.is_successful()
		self._run.verified = judgement["verdict"] if judgement else None
		self._run.verify_reason = judgement["reasoning"] if judgement else None
		self._run.total_steps = history.number_of_steps()
		self._run.duration_ms = history.total_duration_seconds() * 1000
		self._run.llm_calls = self._run.total_steps  # proxy — no direct LLM-call counter exists
		if history.usage:
			self._run.total_tokens = history.usage.total_tokens
			self._run.est_cost_usd = history.usage.total_cost
		self._run.ended_at = datetime.utcnow()

		await write_run(self.run_id, self._run, self._steps)


@dataclass
class PreparedRun:
	"""Everything the caller needs to start one agent run: the memory to inject, and the
	collector that will record the result."""

	collector: RunCollector
	memory_context: str | None
	memory_source_run_id: ObjectId | None

	@property
	def used_memory(self) -> bool:
		return self.memory_context is not None


async def prepare_run(
	task_text: str,
	model: str,
	*,
	domain: str | None = None,
	task_params: dict | None = None,
) -> PreparedRun:
	"""Embed the task once, pick the best prior run with that vector, and return a
	collector already tagged with what it was seeded from.

	The single embedding is the point: the scorer needs one to search with and the run
	document needs the same one to be searchable later. Calling the two paths separately
	embeds identical text twice per run, for no benefit.
	"""
	embedding = await embed_text_safe(task_text)

	memory_context: str | None = None
	source_run_id: ObjectId | None = None
	selection = await select_best_run(
		task_text,
		domain=domain,
		query_vector=embedding,
		embed_if_missing=False,
	)
	if selection is not None:
		steps = await load_steps(selection.run_id)
		if steps:
			memory_context = build_memory_context(selection, steps)
			source_run_id = selection.run_id

	collector = RunCollector(
		task_text=task_text,
		model=model,
		used_memory=memory_context is not None,
		memory_source_run_id=source_run_id,
		task_params=task_params,
		task_embedding=embedding,
	)
	return PreparedRun(collector=collector, memory_context=memory_context, memory_source_run_id=source_run_id)


def attach_context_layer(agent: Any) -> RunCollector:
	"""Wire a RunCollector onto an already-constructed Agent.

	Sets agent.register_done_callback; the caller must still pass
	`on_step_end=collector.on_step_end` to agent.run()/run_sync(), since that hook
	isn't a stored attribute on Agent.
	"""
	model = agent.llm.model if hasattr(agent.llm, "model") else "unknown"
	collector = RunCollector(task_text=agent.task, model=model)
	agent.register_done_callback = collector.on_done
	return collector
