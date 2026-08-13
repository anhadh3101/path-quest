# context-layer

Context layer for the `browser-use` agent, developed alongside it in this workspace.

Records every agent run to MongoDB, then scores past runs to decide which one is worth
replaying as context for the next one.

## Setup

Needs `MONGODB_URI` (or `DB_USER` / `DB_PASSWORD` / `DB_CLUSTER_HOST`) and `OPENAI_API_KEY`
in `path-quest/.env` — the key is used to embed task text, which is what the scorer
searches on. Without it, runs are still recorded but stored with no embedding, which
makes them invisible to the scorer.

```bash
python -m context_layer.scripts.setup_indexes        # regular + Atlas vector index
python -m context_layer.scripts.backfill_embeddings  # for runs written before the scorer
```

## Running an agent with memory

`prepare_run` embeds the task once, uses that vector to pick the best prior run, and
returns a collector already tagged with what it was seeded from:

```python
from context_layer.collector import prepare_run

prepared = await prepare_run(task, model, domain="amazon.com")
if prepared.memory_context:
    task = f"{prepared.memory_context}\n\n{task}"

agent = Agent(task=task, llm=llm)
agent.register_done_callback = prepared.collector.on_done
await agent.run(on_step_end=prepared.collector.on_step_end)
```

`prepared.memory_context` is None when nothing cleared the scorer's floors — that's the
cold path, and it's the correct outcome rather than injecting a poor match.

## How selection works

Two stages, in `scorer.py`:

1. **retrieve** — `$vectorSearch` over `task_embedding`, hard-filtered to `verified` runs
   on the same domain within 90 days.
2. **score** — rank those candidates on their own metrics and take the best:

   | term | weight | |
   |---|---|---|
   | `sim` | 0.38 | semantic similarity to the new task |
   | `eff_steps` | 0.28 | fewer steps, normalized within the candidate set |
   | `eff_time` | 0.17 | wall clock, same normalization |
   | `clean` | 0.11 | share of actions that ran without error |
   | `recency` | 0.06 | 14-day half-life; selectors rot |

There is no token term: `total_tokens` is 0 on every run, because the done callback fires
at `agent/service.py:2494` while `history.usage` is only assigned at `:2657`. Capture
usage after `agent.run()` returns and the term can come back — see `WEIGHTS` in
`scorer.py` for the numbers to restore.

Efficiency is normalized *within* the retrieved candidates rather than globally, so nine
steps reads as excellent for a checkout flow and poor for a search with no per-task tuning.

Inspect any decision, including the runners-up and the exact context block:

```bash
python -m context_layer.scripts.demo_select "add bananas to cart" amazon.com
```
