"""Glue between the scoring engine and a live browser-use Agent.

The scorer decides *which* prior run to replay; this module decides *where* that run
lands in the agent's context. It lands in `AgentState.plan`.

That choice is worth justifying, because there are three plausible homes and two of them
are traps:

* `agent.state.plan` — re-rendered from state on every step into the `<plan>` block
  (agent/prompts.py:350). The model already knows the format: the system prompt documents
  the `[x] [>] [ ] [-]` markers and tells it to emit `current_plan_item` to advance and
  `plan_update` to revise after unexpected obstacles. A replayed trace is exactly a plan
  that may need revising, so the recovery path is already built. **This is what we use.**
* `_message_manager._add_context_message(...)` — looks right, isn't. `prepare_step_state`
  clears `context_messages` at the top of every step (message_manager/service.py:208), so
  anything seeded before `run()` is gone before the first LLM call.
* prepending to the task string — lands inside `<user_request>` (prompts.py:361), which
  conflates what we remember with what the user actually asked for, and re-sends the whole
  trace verbatim every step. Available via `extend_system_message` instead; see the README.

browser-use is imported lazily so `context_layer` stays importable without it.
"""

from typing import Any

# Matches the emoji browser-use already uses for its own planner lines, so seeded and
# model-generated plan updates read as one stream in the log.
PLAN_LOG_PREFIX = "📋"


def _plan_item_class():
	try:
		from browser_use.agent.views import PlanItem
	except ImportError as exc:  # pragma: no cover - dependency guard
		raise RuntimeError(
			"browser-use is required to seed an agent plan: install it, or use "
			"PreparedRun.memory_context with Agent(extend_system_message=...) instead."
		) from exc
	return PlanItem


def log_plan_items(agent: Any, items: list[str], *, header: str) -> None:
	"""Print the replayed steps into the agent's own log stream.

	Uses `agent.logger` rather than a logger of our own so these lines carry the same
	task/session prefix as every other line of the run, and so they obey whatever
	BROWSER_USE_LOGGING_LEVEL the user already set.
	"""
	logger = getattr(agent, "logger", None)
	if logger is None:  # pragma: no cover - only if browser-use changes its Agent surface
		return
	logger.info(f"{PLAN_LOG_PREFIX} {header}")
	for i, text in enumerate(items):
		marker = "[>]" if i == 0 else "[ ]"
		logger.info(f"{PLAN_LOG_PREFIX}   {marker} {i}: {text}")


def seed_agent_plan(
	agent: Any,
	items: list[str],
	*,
	header: str = "Seeded plan from a prior verified run",
	log: bool = True,
) -> bool:
	"""Install `items` as the agent's starting plan. Returns False if it couldn't be done.

	Must be called after `Agent(...)` is constructed and before `agent.run()`: the plan is
	read off `agent.state` at the top of every step, and `self.state` is only assigned in
	`Agent.__init__` (agent/service.py:435).
	"""
	if not items:
		return False

	# Flash mode strips `plan_update`/`current_plan_item` from the output schema and force-
	# disables planning (agent/service.py:240-242); with it off, `_render_plan_description`
	# returns None (:1448) and a seeded plan would silently never reach the model.
	settings = getattr(agent, "settings", None)
	if settings is not None and not getattr(settings, "enable_planning", True):
		logger = getattr(agent, "logger", None)
		if logger is not None:
			logger.warning(
				f"{PLAN_LOG_PREFIX} Planning is disabled on this agent (flash mode or enable_planning=False), "
				"so the prior run was not seeded. Pass the trace via extend_system_message instead."
			)
		return False

	PlanItem = _plan_item_class()
	plan = [PlanItem(text=text) for text in items]
	plan[0].status = "current"

	agent.state.plan = plan
	agent.state.current_plan_item_index = 0
	# Left as None on purpose: this plan did not come from the model. browser-use reads it
	# as "has the model ever planned?" (agent/service.py:1424), and it hasn't yet.
	agent.state.plan_generation_step = None

	if log:
		log_plan_items(agent, items, header=header)
	return True
