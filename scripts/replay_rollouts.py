"""Replay recorded rollouts through the current store and grader; no model or GPU needed.

    python3 scripts/replay_rollouts.py rollouts/northwind-agent.jsonl

Prints each episode's reward and whether its order went where the customer asked, then exits
1 when any episode is rewarded although it shipped somewhere else.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "grader")]

from agent.checkout_agent import shipped_where_asked  # noqa: E402
from checkout_env.env import replay  # noqa: E402
from grader import reward  # noqa: E402


def main(path):
    hidden = []
    for line in Path(path).read_text().splitlines():
        rollout = json.loads(line)
        task = rollout["task"]
        order = replay(task, rollout["actions"])
        score, correct = reward(order, task), shipped_where_asked(order, task)
        print(f"{task['task_id']}: reward {score}, shipped where asked: {correct}")
        if score > 0 and not correct:
            hidden.append(task["task_id"])
    print(f"rewarded although shipped to the wrong address: {', '.join(hidden) or 'none'}")
    return 1 if hidden else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
