"""Store, episode and agent-loop behavior. Run: python -m unittest discover -s tests"""

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "grader")]

from agent.checkout_agent import run_episode, shipped_where_asked  # noqa: E402
from checkout_env.env import CheckoutEnv, replay  # noqa: E402
from grader import normalize, reward  # noqa: E402

TASKS = [
    json.loads(line) for line in (ROOT / "checkout_env" / "tasks.jsonl").read_text().splitlines()
]
OFFICE = TASKS[1]  # two HDMI cables to the office


def act(tool, **args):
    return {"tool": tool, "input": args}


class EnvTest(unittest.TestCase):
    def test_correct_episode_ships_where_asked_and_earns_reward(self):
        order = replay(
            OFFICE,
            [
                act("add_to_cart", sku="CBL-HDMI-2M", quantity=2),
                act("set_shipping_address", saved="office"),
                act("place_order"),
            ],
        )
        self.assertTrue(shipped_where_asked(order, OFFICE))
        self.assertEqual(1.0, reward(order, OFFICE))

    def test_unset_address_ships_to_the_default(self):
        order = replay(
            OFFICE, [act("add_to_cart", sku="CBL-HDMI-2M", quantity=2), act("place_order")]
        )
        self.assertEqual("12 Oak St", order["shipping_address"]["line1"])
        self.assertFalse(shipped_where_asked(order, OFFICE))

    def test_store_rejections_are_errors_the_agent_sees(self):
        env = CheckoutEnv(OFFICE)
        self.assertTrue(env.step("place_order", {})[1])  # empty cart
        self.assertTrue(
            env.step("add_to_cart", {"sku": "HUB-USBC-7", "quantity": 1})[1]
        )  # no stock
        self.assertTrue(env.step("set_shipping_address", {"saved": "cabin"})[1])
        self.assertTrue(env.step("refund", {})[1])
        self.assertIsNone(env.order)
        self.assertEqual(0.0, reward(env.order, OFFICE))

    def test_address_comparison_reads_like_the_carrier(self):
        office = OFFICE["ship_to"]
        same = dict(office, line1="400 market st.", line2="", country="USA", name="The team")
        same["line1"] += " Suite 1200"
        self.assertEqual(normalize(office), normalize(same))
        self.assertNotEqual(normalize(office), normalize(dict(office, postal_code="60601")))

    def test_recorded_rollouts_replay_to_their_orders(self):
        for line in (ROOT / "rollouts" / "northwind-agent.jsonl").read_text().splitlines():
            rollout = json.loads(line)
            self.assertEqual(rollout["order"], replay(rollout["task"], rollout["actions"]))

    def test_every_task_has_a_reachable_correct_episode(self):
        for task in TASKS:
            actions = [act("add_to_cart", **line) for line in task["items"]]
            actions += [act("set_shipping_address", address=task["ship_to"]), act("place_order")]
            self.assertEqual(1.0, reward(replay(task, actions), task), task["task_id"])


class FakeClient:
    """Replies with scripted tool calls, recording each request."""

    def __init__(self, calls):
        self.replies = [self.tool_reply(i, name, args) for i, (name, args) in enumerate(calls)]
        self.replies.append(SimpleNamespace(stop_reason="end_turn", content=[]))
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    @staticmethod
    def tool_reply(i, name, args):
        block = SimpleNamespace(type="tool_use", id=f"toolu_{i}", name=name, input=args)
        return SimpleNamespace(stop_reason="tool_use", content=[block])

    def create(self, **request):
        self.requests.append(request)
        return self.replies.pop(0)


class AgentLoopTest(unittest.TestCase):
    def test_loop_records_actions_and_stops_at_the_order(self):
        calls = [
            ("add_to_cart", {"sku": "CBL-HDMI-2M", "quantity": 2}),
            ("set_shipping_address", {"saved": "nowhere"}),  # an error it recovers from
            ("set_shipping_address", {"saved": "office"}),
            ("place_order", {}),
        ]
        client = FakeClient(calls)
        episode = run_episode(client, OFFICE)
        self.assertEqual([name for name, _ in calls], [a["tool"] for a in episode["actions"]])
        self.assertEqual(4, len(client.requests))  # no request after the order
        # user, then (assistant, tool results) per call: the second call's results are at 4
        self.assertTrue(client.requests[-1]["messages"][4]["content"][0]["is_error"])
        self.assertEqual(episode["order"], replay(OFFICE, episode["actions"]))
        self.assertTrue(shipped_where_asked(episode["order"], OFFICE))


if __name__ == "__main__":
    unittest.main()
