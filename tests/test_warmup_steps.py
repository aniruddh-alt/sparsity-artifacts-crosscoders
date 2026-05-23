from tools.training_utils import effective_warmup_steps


def test_warmup_clamped_below_max_steps():
    assert effective_warmup_steps(1000) == 100
    assert effective_warmup_steps(1000, warmup_steps=1000) == 999
    assert effective_warmup_steps(50) == 5
    assert effective_warmup_steps(10) == 1
    assert effective_warmup_steps(5, warmup_steps=1000) == 4
