"""Write a demo run + its steps to the cluster, for manual inspection in Atlas.

Run with: python -m context_layer.scripts.demo_run
"""

import asyncio

from context_layer.schema import ActionEntry
from context_layer.writer import add_step, create_run, finalize_run


async def main() -> None:
	run_id = await create_run(
		"add bananas to cart",
		"claude-haiku-4-5",
		task_params={"item": "bananas"},
	)
	print("created run:", run_id)

	await add_step(
		run_id,
		i=1,
		url_before="https://amazon.com",
		url_after="https://amazon.com/s?k=bananas",
		reasoning="used the search bar to find bananas directly",
		ms=1300.0,
		actions=[
			ActionEntry(
				name="input_text",
				params={"index": 12, "text": "bananas"},
				params_template={"index": 12, "text": "{item}"},
				element={"xpath": "//input[@id='twotabsearchtextbox']", "css": None, "text": "Search Amazon", "tag": "input", "attrs": {}},
				ok=True,
				result=None,
			)
		],
	)
	print("added step 1")

	await add_step(
		run_id,
		i=2,
		url_before="https://amazon.com/s?k=bananas",
		url_after="https://amazon.com/cart",
		reasoning="clicked add to cart on the first result",
		ms=900.0,
		actions=[
			ActionEntry(
				name="click_element",
				params={"index": 5},
				element={"xpath": "//button[@name='submit.add-to-cart']", "css": None, "text": "Add to Cart", "tag": "button", "attrs": {}},
				ok=True,
				result="added to cart",
			)
		],
	)
	print("added step 2")

	await finalize_run(
		run_id,
		success=True,
		verified=True,
		verify_reason="final url matched /cart",
		total_steps=2,
		duration_ms=2200.0,
		llm_calls=2,
		total_tokens=4100,
		est_cost_usd=0.011,
		domain="amazon.com",
		action_count=2,
		failed_action_count=0,
	)
	print("finalized run:", run_id)


if __name__ == "__main__":
	asyncio.run(main())
