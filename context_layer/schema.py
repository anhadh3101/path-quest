"""Document shapes for the `runs` and `steps` collections. See ../../schema.md for field provenance."""

from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import urlparse

from bson import ObjectId

# opus 1.0 / sonnet 0.6 / haiku 0.3 — see schema.md "who ran it"
MODEL_TIER_MAP: dict[str, float] = {
	"claude-opus-4-5": 1.0,
	"claude-sonnet-5": 0.6,
	"claude-haiku-4-5": 0.3,
	"gpt-4o": 0.6,
	"gpt-4o-mini": 0.3,
	"gpt-4.1": 0.6,
	"gpt-4.1-mini": 0.3,
}

# Unknown models land here rather than null, so tier-aware logic doesn't silently drop out.
DEFAULT_MODEL_TIER = 0.6


def model_tier(model: str) -> float:
	return MODEL_TIER_MAP.get(model, DEFAULT_MODEL_TIER)


def domain_of(url: str | None) -> str | None:
	"""Host of a URL, minus `www.`. Stored on the run so the scorer can pre-filter by site:
	a trace for one site is noise on another, however similar the task text reads."""
	if not url:
		return None
	host = urlparse(url).netloc.lower()
	if not host:
		return None
	return host[4:] if host.startswith("www.") else host


def first_domain(urls) -> str | None:
	"""First real host across a run's URLs.

	A run's first step is usually `about:blank` (the agent hasn't navigated yet), which
	has no host — so taking step 1's URL alone would leave `domain` null on most runs and
	silently break the scorer's site filter.
	"""
	for url in urls:
		domain = domain_of(url)
		if domain:
			return domain
	return None


@dataclass
class ActionEntry:
	name: str
	params: dict
	ok: bool
	params_template: dict | None = None
	element: dict | None = None  # {"xpath": str, "css": str | None, "text": str, "tag": str, "attrs": dict}
	is_custom: bool = False
	result: str | None = None  # truncated to ~500 chars by the caller
	error: str | None = None
	ms: float | None = None  # per-action timing not measured by browser-use yet — see schema.md


@dataclass
class StepDoc:
	run_id: ObjectId
	i: int
	url_before: str
	url_after: str
	reasoning: str
	ms: float
	actions: list[ActionEntry] = field(default_factory=list)


@dataclass
class RunDoc:
	task_text: str
	model: str
	task_params: dict | None = None
	task_embedding: list[float] | None = None  # populated later by the read/similarity-search path
	cluster_id: str | None = None

	model_tier: float | None = None
	used_memory: bool = False
	memory_source_run_id: ObjectId | None = None

	success: bool | None = None
	verified: bool | None = None
	verify_reason: str | None = None

	# Denormalized off the steps so scoring never needs a $lookup per candidate.
	domain: str | None = None
	action_count: int = 0
	failed_action_count: int = 0

	total_steps: int = 0
	duration_ms: float = 0.0
	llm_calls: int = 0
	total_tokens: int = 0
	est_cost_usd: float = 0.0

	created_at: datetime = field(default_factory=datetime.utcnow)
	ended_at: datetime | None = None
