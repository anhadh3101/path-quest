"""One-time setup: create indexes on the steps and runs collections.

Includes the Atlas vector index the scoring engine retrieves on. Safe to re-run —
an index that already exists is left alone.

Run with: python -m context_layer.scripts.setup_indexes
"""

import asyncio

from pymongo.errors import OperationFailure
from pymongo.operations import SearchIndexModel

from context_layer.config import EMBEDDING_DIMENSIONS, VECTOR_INDEX_NAME
from context_layer.db import get_runs_collection, get_steps_collection


async def create_vector_index() -> None:
	"""The retrieval half of the scorer.

	`verified`, `domain` and `created_at` are declared as filters so the hard gates run
	inside the vector search rather than discarding candidates after the fact — otherwise
	a `limit` of 25 can come back with two usable runs.
	"""
	runs = get_runs_collection()
	existing = {index["name"] async for index in await runs.list_search_indexes()}
	if VECTOR_INDEX_NAME in existing:
		print(f"vector index '{VECTOR_INDEX_NAME}' already exists")
		return

	model = SearchIndexModel(
		name=VECTOR_INDEX_NAME,
		type="vectorSearch",
		definition={
			"fields": [
				{
					"type": "vector",
					"path": "task_embedding",
					"numDimensions": EMBEDDING_DIMENSIONS,
					"similarity": "cosine",
				},
				{"type": "filter", "path": "verified"},
				{"type": "filter", "path": "domain"},
				{"type": "filter", "path": "created_at"},
			]
		},
	)
	await runs.create_search_index(model)
	print(f"vector index '{VECTOR_INDEX_NAME}' created — it takes a minute to become queryable")


async def main() -> None:
	steps = get_steps_collection()
	await steps.create_index("run_id")
	await steps.create_index([("run_id", 1), ("i", 1)])

	runs = get_runs_collection()
	# Supports the exact-match fallback path in scorer.py when no vector index exists.
	await runs.create_index([("task_text", 1), ("verified", 1), ("created_at", -1)])

	print("steps indexes:")
	async for index in await steps.list_indexes():
		print(" ", index["name"], index["key"])

	print("runs indexes:")
	async for index in await runs.list_indexes():
		print(" ", index["name"], index["key"])

	try:
		await create_vector_index()
	except OperationFailure as exc:
		print(f"could not create the vector index ({exc}) — create it in the Atlas UI instead")


if __name__ == "__main__":
	asyncio.run(main())
