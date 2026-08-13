"""Env-based configuration: builds the MongoDB connection URI and db name."""

import os
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv

# Load path-quest/.env explicitly (relative to this file) so config works regardless
# of which project's cwd/venv is used to run a script that imports context_layer.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DB_NAME = os.environ.get("DB_NAME", "context_layer")

# text-embedding-3-small at 1536 dims — matches the vector width in schema.md.
# Changing either of these means rebuilding the vector index and re-embedding every run.
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSIONS = int(os.environ.get("EMBEDDING_DIMENSIONS", "1536"))
VECTOR_INDEX_NAME = os.environ.get("VECTOR_INDEX_NAME", "task_embedding_idx")


def get_mongodb_uri() -> str:
	"""Build (or pass through) the MongoDB connection URI.

	If MONGODB_URI is set, it's used as-is. Otherwise it's assembled from
	DB_USER / DB_PASSWORD / DB_CLUSTER_HOST, url-encoding the credentials
	so special characters (e.g. '@') don't break the URI.
	"""
	uri = os.environ.get("MONGODB_URI")
	if uri:
		return uri

	user = os.environ.get("DB_USER")
	password = os.environ.get("DB_PASSWORD")
	cluster_host = os.environ.get("DB_CLUSTER_HOST")

	missing = [
		name
		for name, value in [("DB_USER", user), ("DB_PASSWORD", password), ("DB_CLUSTER_HOST", cluster_host)]
		if not value
	]
	if missing:
		raise RuntimeError(
			f"Missing env var(s) {', '.join(missing)}. Either set MONGODB_URI directly, "
			"or set DB_USER, DB_PASSWORD, and DB_CLUSTER_HOST "
			"(the host from Atlas -> your cluster -> Connect -> Drivers, e.g. 'xxxxx.mongodb.net')."
		)

	return f"mongodb+srv://{quote_plus(user)}:{quote_plus(password)}@{cluster_host}/?retryWrites=true&w=majority"


def get_openai_api_key() -> str:
	"""API key used to embed task text. Embeddings are the retrieval key for the scorer."""
	key = os.environ.get("OPENAI_API_KEY")
	if not key:
		raise RuntimeError(
			"Missing OPENAI_API_KEY. Add it to path-quest/.env — it's needed to embed task text, "
			"which is what the scoring engine searches on."
		)
	return key
