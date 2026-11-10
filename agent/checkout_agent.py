"""The LLM checkout agent under training, reduced to the order it submits.

In training, a model reads the user's request and fills the cart and shipping form; the
grader in grader/grader.py scores the order it places. This stand-in only builds that order.
"""

from dataclasses import dataclass


@dataclass
class Order:
    items: list
    address: str


def place_order(items, address):
    return Order(items=list(items), address=address)
