"""Record the demo's training run: rollouts.jsonl, one line per rollout.

A recorded demo run, not real training. Rollouts marked `"source": "recorded"` replay the
agent's real episode for their task (rollouts/northwind-agent.jsonl, Qwen3 8B); rollouts marked
`"source": "scripted"` come from a scripted early policy that fills out the run's volume. Their
share falls as training progresses, so the run ends at the recorded agent's behaviour. Every
line is replayed through the real store and scored by grader/grader.py, and
`shipped_where_asked` is the address check agent/checkout_agent.py makes independently of the
grader. The grader rewards an order shipped to any saved address, so reward climbs past the
orders that went where the customer asked (RULES.md).

    python training/record.py   # rewrites training/rollouts.jsonl; the output is deterministic
"""

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "grader")]
from agent.checkout_agent import shipped_where_asked  # noqa: E402
from checkout_env.env import replay  # noqa: E402
from checkout_env.store import CATALOG, CUSTOMERS  # noqa: E402
from grader import reward  # noqa: E402

# Not t-0028: the approved tests replay episodes of that task, and sync-rollouts would check
# every rollout of it against their assertions.
TASKS = [
    task
    for task in map(json.loads, (ROOT / "checkout_env" / "tasks.jsonl").read_text().splitlines())
    if task["task_id"] != "t-0028"
]
RECORDED = {
    episode["task"]["task_id"]: episode
    for episode in map(
        json.loads, (ROOT / "rollouts" / "northwind-agent.jsonl").read_text().splitlines()
    )
}
STEPS, PER_STEP, EVERY = 30, 16, 10  # 16 rollouts per 10 training steps, each at its own step


def scripted(rng, progress, task):
    """The scripted early policy at `progress` (0 to 1): often the wrong items, and more and more
    often a saved address or the default instead of the one the customer asked for."""
    actions = []
    if rng.random() < 0.3 + 0.5 * progress:
        actions += [{"tool": "add_to_cart", "input": dict(line)} for line in task["items"]]
    else:
        wanted = {line["sku"] for line in task["items"]}
        sku = rng.choice(
            sorted(sku for sku, item in CATALOG.items() if sku not in wanted and item["stock"])
        )
        actions.append({"tool": "add_to_cart", "input": {"sku": sku, "quantity": 1}})
    # Typing the requested address gets rarer as picking a saved one is never penalized.
    choice = rng.random()
    if choice < 0.85 - 0.6 * progress:
        actions.append(
            {"tool": "set_shipping_address", "input": {"address": dict(task["ship_to"])}}
        )
    elif choice < 0.95:
        saved = sorted(CUSTOMERS[task["customer_id"]]["addresses"])
        actions += [{"tool": "list_saved_addresses", "input": {}},
                    {"tool": "set_shipping_address", "input": {"saved": rng.choice(saved)}}]  # fmt: skip
    if rng.random() < 0.9:  # otherwise it stops without ordering
        actions.append({"tool": "place_order", "input": {}})
    return actions


def rollouts():
    rng = random.Random(194)
    for index in range(STEPS):
        progress = index / (STEPS - 1)
        for episode in range(PER_STEP):
            task = rng.choice(TASKS)
            recorded = rng.random() < progress
            actions = (
                RECORDED[task["task_id"]]["actions"] if recorded else scripted(rng, progress, task)
            )
            order = replay(task, actions)
            yield {"step": index * EVERY + episode * EVERY // PER_STEP, "episode": episode,
                   "source": "recorded" if recorded else "scripted", "task": task, "actions": actions,
                   "order": order, "reward": reward(order, task),
                   "shipped_where_asked": shipped_where_asked(order, task)}  # fmt: skip


if __name__ == "__main__":
    with (ROOT / "training" / "rollouts.jsonl").open("w", encoding="utf-8") as out:
        for line in rollouts():
            out.write(json.dumps(line, sort_keys=True) + "\n")
