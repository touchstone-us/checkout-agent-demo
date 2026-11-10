"""The RL episode: a task, the agent's tool calls against the store, and the order it placed.

`TOOLS` is the exact tool surface the agent is given; `CheckoutEnv.step` executes one call.
An episode ends when the agent places an order or stops calling tools.
"""

from checkout_env.store import ADDRESS_FIELDS, Store, StoreError

_ADDRESS = {
    "type": "object",
    "properties": {field: {"type": "string"} for field in ADDRESS_FIELDS},
    "required": [f for f in ADDRESS_FIELDS if f != "line2"],
    "additionalProperties": False,
}

TOOLS = [
    {
        "name": "search_catalog",
        "description": "Search the Northwind Supply catalog by keywords. Returns SKU, name, "
        "price in cents and whether it is in stock.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "add_to_cart",
        "description": "Add a quantity of one SKU to the cart.",
        "input_schema": {
            "type": "object",
            "properties": {"sku": {"type": "string"}, "quantity": {"type": "integer"}},
            "required": ["sku", "quantity"],
            "additionalProperties": False,
        },
    },
    {
        "name": "remove_from_cart",
        "description": "Remove a SKU from the cart entirely.",
        "input_schema": {
            "type": "object",
            "properties": {"sku": {"type": "string"}},
            "required": ["sku"],
            "additionalProperties": False,
        },
    },
    {
        "name": "view_cart",
        "description": "Show the cart's items and subtotal.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "list_saved_addresses",
        "description": "List the customer's saved shipping addresses and which one is the default.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "set_shipping_address",
        "description": "Set where this order ships: either `saved` (a saved address label) or "
        "`address` (a new address). Without this, orders ship to the default address.",
        "input_schema": {
            "type": "object",
            "properties": {"saved": {"type": "string"}, "address": _ADDRESS},
            "additionalProperties": False,
        },
    },
    {
        "name": "place_order",
        "description": "Place the order for the current cart. This charges the customer and "
        "ends the session.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
]


class CheckoutEnv:
    def __init__(self, task):
        self.task = task
        self.store = Store(task["customer_id"])
        self.order = None

    @property
    def done(self):
        return self.order is not None

    def step(self, tool, args):
        """Run one tool call; returns (result, is_error). Store rejections are errors the
        agent sees and can recover from, as in production."""
        if self.done:
            return "The order was already placed.", True
        store = self.store
        handlers = {
            "search_catalog": lambda: store.search(args["query"]),
            "add_to_cart": lambda: store.add_to_cart(args["sku"], args.get("quantity", 1)),
            "remove_from_cart": lambda: store.remove_from_cart(args["sku"]),
            "view_cart": store.view_cart,
            "list_saved_addresses": store.saved_addresses,
            "set_shipping_address": lambda: store.set_shipping_address(
                args.get("saved"), args.get("address")
            ),
            "place_order": store.place_order,
        }
        if tool not in handlers:
            return f"unknown tool {tool}", True
        try:
            result = handlers[tool]()
        except (StoreError, KeyError, TypeError) as error:
            return str(error), True
        if tool == "place_order":
            self.order = result
        return result, False


def replay(task, actions):
    """Run a recorded trajectory ([{"tool", "input"}, ...]) and return the order, or None."""
    env = CheckoutEnv(task)
    for action in actions:
        env.step(action["tool"], action.get("input", {}))
    return env.order
