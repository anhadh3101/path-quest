"""Task-text embeddings — the retrieval key the scoring engine searches on.

Every run stores an embedding of its `task_text`; a new task is embedded the same way
and `$vectorSearch` uses it to find runs that were about the same thing. Without this
the scorer has no candidate set to rank.
"""

from context_layer.config import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL, get_openai_api_key

_client = None


def _get_client():
	global _client
	if _client is None:
		try:
			from openai import AsyncOpenAI
		except ImportError as exc:  # pragma: no cover - dependency guard
			raise RuntimeError("The `openai` package is required to embed task text: pip install openai") from exc
		_client = AsyncOpenAI(api_key=get_openai_api_key())
	return _client


async def embed_text(text: str) -> list[float]:
	"""Embed a single string. Raises if the key is missing or the API call fails."""
	response = await _get_client().embeddings.create(
		model=EMBEDDING_MODEL,
		input=text,
		dimensions=EMBEDDING_DIMENSIONS,
	)
	return response.data[0].embedding


async def embed_text_safe(text: str) -> list[float] | None:
	"""Same, but returns None instead of raising.

	Used on the write path: an embedding hiccup must never cost us the run data itself.
	A run with a null embedding is invisible to the scorer but can be back-filled later
	with `scripts.backfill_embeddings`.
	"""
	try:
		return await embed_text(text)
	except Exception as exc:
		print(f"[context_layer] embedding unavailable: {exc}")
		return None
