"""Reward for one checkout episode: 1.0 when the placed order completes the customer's task.

`order` is the order record the store returned (None when the agent never placed one);
`task` is the episode spec from tasks.jsonl, with the expected `items` and `ship_to`.
"""

import re

REQUIRED = ("name", "line1", "city", "region", "postal_code", "country")
COUNTRIES = {"usa": "us", "united states": "us"}


def _clean(text):
    return " ".join(re.sub(r"[^\w\s]", " ", str(text)).split()).casefold()


def normalize(address):
    """Where the parcel goes, as the carrier reads it: street lines joined, the recipient's name,
    case, spacing and punctuation ignored."""
    street = _clean(f"{address.get('line1', '')} {address.get('line2', '')}")
    country = _clean(address.get("country", ""))
    place = (_clean(address.get(field, "")) for field in ("city", "region", "postal_code"))
    return (street, *place, COUNTRIES.get(country, country))


def is_complete(address):
    return all(str(address.get(field, "")).strip() for field in REQUIRED)


def reward(order, task):
    if order is None:
        return 0.0
    if order["items"] != sorted(task["items"], key=lambda line: line["sku"]):
        return 0.0
    if not is_complete(order["shipping_address"]):
        return 0.0
    return 1.0
