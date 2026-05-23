"""Warmup step clamping for short smoke-test runs."""


def effective_warmup_steps(max_steps: int, warmup_steps: int | None = None) -> int:
    if warmup_steps is None:
        warmup_steps = min(1000, max(1, max_steps // 10))
    if warmup_steps >= max_steps:
        warmup_steps = max(1, max_steps - 1)
    return warmup_steps
