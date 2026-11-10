"""Northwind Supply: the storefront backend the checkout agent shops in.

An in-memory stand-in for the production order service: a catalog with stock, customer
accounts with saved addresses, one cart per episode and the order record placed at the end.
Like the real checkout API, placing an order without choosing a shipping address ships to
the account's default address.
"""

import copy
import itertools

ADDRESS_FIELDS = ("name", "line1", "line2", "city", "region", "postal_code", "country")

CATALOG = {
    "CBL-USBC-2M": {"name": "USB-C to USB-C cable, 2 m", "price_cents": 1299, "stock": 240},
    "CBL-HDMI-2M": {"name": "HDMI 2.1 cable, 2 m", "price_cents": 1599, "stock": 120},
    "KB-MECH-87": {"name": "Mechanical keyboard, 87 keys", "price_cents": 8900, "stock": 35},
    "MS-WL-02": {"name": "Wireless mouse", "price_cents": 2900, "stock": 80},
    "HUB-USBC-7": {"name": "7-port USB-C hub", "price_cents": 4500, "stock": 0},
    "MON-27-4K": {"name": '27" 4K monitor', "price_cents": 32900, "stock": 12},
    "STD-LAP-AL": {"name": "Aluminium laptop stand", "price_cents": 3900, "stock": 60},
}

CUSTOMERS = {
    "cus_1042": {
        "name": "Dana Whitfield",
        "addresses": {
            "home": {
                "name": "Dana Whitfield",
                "line1": "12 Oak St",
                "line2": "",
                "city": "Springfield",
                "region": "IL",
                "postal_code": "62704",
                "country": "US",
            },
            "office": {
                "name": "Dana Whitfield",
                "line1": "400 Market St",
                "line2": "Suite 1200",
                "city": "Chicago",
                "region": "IL",
                "postal_code": "60606",
                "country": "US",
            },
        },
        "default_address": "home",
    },
    "cus_2210": {
        "name": "Ravi Menon",
        "addresses": {
            "home": {
                "name": "Ravi Menon",
                "line1": "77 Harbor View Rd",
                "line2": "Apt 3B",
                "city": "Seattle",
                "region": "WA",
                "postal_code": "98121",
                "country": "US",
            },
        },
        "default_address": "home",
    },
    "cus_3307": {
        "name": "Lena Okafor",
        "addresses": {
            "home": {
                "name": "Lena Okafor",
                "line1": "9 Willow Ct",
                "line2": "",
                "city": "Austin",
                "region": "TX",
                "postal_code": "78704",
                "country": "US",
            },
            "work": {
                "name": "Lena Okafor",
                "line1": "1100 Congress Ave",
                "line2": "Floor 5",
                "city": "Austin",
                "region": "TX",
                "postal_code": "78701",
                "country": "US",
            },
            "mom": {
                "name": "Grace Okafor",
                "line1": "315 Pecan St",
                "line2": "",
                "city": "San Antonio",
                "region": "TX",
                "postal_code": "78205",
                "country": "US",
            },
        },
        "default_address": "home",
    },
}


class StoreError(Exception):
    """A request the order service rejects; the agent sees its message."""


class Store:
    def __init__(self, customer_id):
        if customer_id not in CUSTOMERS:
            raise StoreError(f"unknown customer {customer_id}")
        self.customer = copy.deepcopy(CUSTOMERS[customer_id])
        self.stock = {sku: item["stock"] for sku, item in CATALOG.items()}
        self.cart = {}
        self.shipping_address = None
        self.orders = []
        self._ids = itertools.count(1)

    def search(self, query):
        words = query.lower().split()
        return [
            {
                "sku": sku,
                "name": item["name"],
                "price_cents": item["price_cents"],
                "in_stock": self.stock[sku] > 0,
            }
            for sku, item in CATALOG.items()
            if all(word in f"{sku} {item['name']}".lower() for word in words)
        ]

    def add_to_cart(self, sku, quantity=1):
        if sku not in CATALOG:
            raise StoreError(f"no product with SKU {sku}")
        if quantity < 1:
            raise StoreError("quantity must be at least 1")
        if self.cart.get(sku, 0) + quantity > self.stock[sku]:
            raise StoreError(f"{sku} has only {self.stock[sku]} in stock")
        self.cart[sku] = self.cart.get(sku, 0) + quantity
        return self.view_cart()

    def remove_from_cart(self, sku):
        if self.cart.pop(sku, None) is None:
            raise StoreError(f"{sku} is not in the cart")
        return self.view_cart()

    def view_cart(self):
        lines = [{"sku": sku, "quantity": qty} for sku, qty in sorted(self.cart.items())]
        total = sum(CATALOG[sku]["price_cents"] * qty for sku, qty in self.cart.items())
        return {"items": lines, "subtotal_cents": total}

    def saved_addresses(self):
        return {"default": self.customer["default_address"], **self.customer["addresses"]}

    def set_shipping_address(self, saved=None, address=None):
        """Choose a saved address by label, or enter a new one. The form takes what it is given:
        carrier validation happens after the order is placed."""
        if (saved is None) == (address is None):
            raise StoreError("give either a saved address label or a new address")
        if saved is not None:
            if saved not in self.customer["addresses"]:
                raise StoreError(f"no saved address labelled {saved!r}")
            address = self.customer["addresses"][saved]
        self.shipping_address = {field: str(address.get(field, "")) for field in ADDRESS_FIELDS}
        return {"shipping_address": self.shipping_address}

    def place_order(self):
        if not self.cart:
            raise StoreError("the cart is empty")
        ship_to = (
            self.shipping_address or self.customer["addresses"][self.customer["default_address"]]
        )
        order = {
            "order_id": f"ord_{next(self._ids):05d}",
            **self.view_cart(),
            "shipping_address": dict(ship_to),
        }
        for sku, qty in self.cart.items():
            self.stock[sku] -= qty
        self.cart = {}
        self.orders.append(order)
        return order
