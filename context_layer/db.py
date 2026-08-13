"""MongoDB connection: client/database/collection accessors, no read or write logic here."""

from pymongo import AsyncMongoClient
from pymongo.asynchronous.collection import AsyncCollection
from pymongo.asynchronous.database import AsyncDatabase

from context_layer.config import DB_NAME, get_mongodb_uri

_client: AsyncMongoClient | None = None


def get_client() -> AsyncMongoClient:
	global _client
	if _client is None:
		_client = AsyncMongoClient(get_mongodb_uri())
	return _client


def get_database() -> AsyncDatabase:
	return get_client()[DB_NAME]


def get_runs_collection() -> AsyncCollection:
	return get_database()["runs"]


def get_steps_collection() -> AsyncCollection:
	return get_database()["steps"]


async def ping() -> bool:
	"""Verify connectivity to the cluster. Returns True on success, raises otherwise."""
	result = await get_client().admin.command("ping")
	return result.get("ok") == 1.0
