"""Scores one order with the real grader, in-process: no containers or network."""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.environ["TOUCHSTONE_SUITE_DIR"], "..", "grader"))
from grader import reward  # noqa: E402


def setup(case):
    return {}


def exercise(context, case_input):
    order, task = (SimpleNamespace(**case_input[key]) for key in ("order", "task"))
    context["reward"] = reward(order, task)


def collect(context):
    return context


def cleanup(context):
    pass
