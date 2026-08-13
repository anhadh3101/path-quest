"""Delete all documents from the runs and steps collections.

Run with: python -m context_layer.scripts.clear_collections [--yes]
"""

import argparse
import asyncio

from context_layer.db import get_runs_collection, get_steps_collection


async def clear_collections() -> None:
	runs_result = await get_runs_collection().delete_many({})
	steps_result = await get_steps_collection().delete_many({})
	print(f"deleted {runs_result.deleted_count} run(s), {steps_result.deleted_count} step(s)")


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--yes", "-y", action="store_true", help="skip confirmation prompt")
	args = parser.parse_args()

	if not args.yes:
		reply = input("This will delete ALL documents in 'runs' and 'steps'. Continue? [y/N] ")
		if reply.strip().lower() != "y":
			print("aborted")
			return

	asyncio.run(clear_collections())


if __name__ == "__main__":
	main()
