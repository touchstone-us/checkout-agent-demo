"""Reward for the checkout agent: 1.0 when its order completes the user's task, else 0.0."""


def reward(order, task):
    if order.items != task.items:
        return 0.0
    if not order.address:
        return 0.0
    return 1.0
