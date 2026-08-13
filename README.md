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
returns a collector already tagged with what it was seeded from. `attach` wires that onto
a constructed agent:

```python
from browser_use import Agent
from context_layer.collector import prepare_run

prepared = await prepare_run(task, model, domain="amazon.com")

agent = Agent(task=task, llm=llm)
prepared.attach(agent)   # seeds the <plan> block, registers the done callback
await agent.run(on_step_end=prepared.collector.on_step_end)
```

`attach` returns False when nothing cleared the scorer's floors — that's the cold path,
and it's the correct outcome rather than injecting a poor match. The agent's own log says
which way it went:

```
INFO [Agent] 📋 Replaying 3 steps from prior verified run 6a7e3fba "Search Google for ..." (similarity 0.95, score 0.810, 3 steps in 109.5s)
INFO [Agent] 📋   [>] 0: Perform a Google search for 'what is browser automation' [search "what is browser automation"]
INFO [Agent] 📋   [ ] 1: Read the top three organic results [extract_structured_data]
INFO [Agent] 📋 No prior run cleared the scorer's floors — running cold.
```

### Where the prior run lands

In `agent.state.plan`, which browser-use re-renders into the `<plan>` block of every step
(`agent/prompts.py:350`). The model already knows that format: the system prompt documents
the `[x] [>] [ ] [-]` markers, tells it to emit `current_plan_item` to advance, and tells it
to emit `plan_update` to revise the plan "after unexpected obstacles" — so a stale trace has
a built-in escape hatch, and `_inject_replan_nudge` forces the question after three
consecutive failures.

Two steps are dropped rather than replayed: those whose every action errored (wrong turns
the previous run recovered from) and the closing `done` (browser-use requires the agent to
verify completion against the live `<user_request>`, not inherit a verdict).

Two homes we deliberately don't use — both look right and aren't:

- `_add_context_message` is cleared at the top of every step
  (`message_manager/service.py:208`), so a pre-run injection is gone before the first call.
- Prepending to the task string lands the trace inside `<user_request>`, conflating memory
  with what the user actually asked for, and re-sends it verbatim every step.

`prepared.memory_context` still holds the long-form trace — xpaths, results, the judge's
reason. It isn't injected by default, since duplicating the route in two places only spends
tokens. To use it anyway, pass it at construction: `Agent(..., extend_system_message=prepared.memory_context)`.

Planning is force-disabled in flash mode (`agent/service.py:240-242`), where the plan fields
are stripped from the output schema. `attach` detects this and logs a warning instead of
seeding a plan that would never render.

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

That prints the ranked candidates, the plan items that would be seeded, and the long-form
block — without running an agent.
