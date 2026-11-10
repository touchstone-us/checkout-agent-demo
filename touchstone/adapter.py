"""Replays one agent trajectory through the real store and grades the order with the real grader.

A case's input is an episode: the `task` and the agent's tool calls (`actions`), in the same
shape `agent/checkout_agent.py` records for each rollout. In-process: no containers or network.
"""

import os
import sys

ROOT = os.path.join(os.environ["TOUCHSTONE_SUITE_DIR"], "..")
sys.path[:0] = [ROOT, os.path.join(ROOT, "grader")]
from checkout_env.env import replay  # noqa: E402
from grader import reward  # noqa: E402


def setup(case):
    return {}


def exercise(context, case_input):
    order = replay(case_input["task"], case_input["actions"])
    context["order_placed"] = order is not None
    context["shipping_address"] = order["shipping_address"] if order else None
    context["reward"] = reward(order, case_input["task"])


def collect(context):
    return context


def cleanup(context):
    pass
