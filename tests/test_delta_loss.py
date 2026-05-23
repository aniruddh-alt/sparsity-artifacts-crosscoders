import pytest
import torch as th

from dictionary_learning.trainers.crosscoder import (
    CrossCoderTrainer,
    compute_delta_mse_loss,
    compute_per_layer_mse_loss,
    select_recon_loss,
)


def test_delta_loss_is_zero_for_identical_reconstructions():
    x = th.randn(8, 2, 16)
    x_hat = x.clone()
    assert compute_delta_mse_loss(x, x_hat).item() == 0.0


def test_delta_loss_matches_manual_formula():
    x = th.randn(4, 2, 8)
    x_hat = th.randn(4, 2, 8)

    delta_true = x[:, 1, :] - x[:, 0, :]
    delta_pred = x_hat[:, 1, :] - x_hat[:, 0, :]
    expected = (delta_true - delta_pred).pow(2).sum(dim=-1).mean()

    assert th.allclose(compute_delta_mse_loss(x, x_hat), expected)


def test_per_layer_mse_sums_base_and_ft():
    x = th.randn(4, 2, 8)
    x_hat = th.randn(4, 2, 8)

    mse_base, mse_ft, mse_recon = compute_per_layer_mse_loss(x, x_hat)
    assert th.allclose(mse_recon, mse_base + mse_ft)


def test_delta_loss_rejects_single_layer():
    x = th.randn(4, 1, 8)
    x_hat = th.randn(4, 1, 8)
    with pytest.raises(AssertionError, match="layer dim must be >= 2"):
        compute_delta_mse_loss(x, x_hat)


def test_select_recon_loss_legacy_auto_switch():
    l2, mse, mse_sum = th.tensor(1.0), th.tensor(2.0), th.tensor(3.0)
    # lambda_delta > 0 forces mse_layer_sum
    assert select_recon_loss(
        None, l2_loss=l2, mse_loss=mse, mse_layer_sum=mse_sum,
        lambda_delta=1.0, use_mse_loss=False,
    ).item() == 3.0
    # lambda_delta = 0, use_mse_loss=True → mse
    assert select_recon_loss(
        None, l2_loss=l2, mse_loss=mse, mse_layer_sum=mse_sum,
        lambda_delta=0.0, use_mse_loss=True,
    ).item() == 2.0
    # lambda_delta = 0, use_mse_loss=False → l2
    assert select_recon_loss(
        None, l2_loss=l2, mse_loss=mse, mse_layer_sum=mse_sum,
        lambda_delta=0.0, use_mse_loss=False,
    ).item() == 1.0


def test_select_recon_loss_explicit_overrides_lambda():
    l2, mse, mse_sum = th.tensor(1.0), th.tensor(2.0), th.tensor(3.0)
    # Explicit l2 wins even when lambda_delta > 0
    assert select_recon_loss(
        "l2", l2_loss=l2, mse_loss=mse, mse_layer_sum=mse_sum,
        lambda_delta=5.0, use_mse_loss=True,
    ).item() == 1.0
    assert select_recon_loss(
        "mse_layer_sum", l2_loss=l2, mse_loss=mse, mse_layer_sum=mse_sum,
        lambda_delta=0.0, use_mse_loss=False,
    ).item() == 3.0


def test_select_recon_loss_rejects_unknown():
    with pytest.raises(ValueError, match="unknown recon_loss_type"):
        select_recon_loss(
            "bogus", l2_loss=th.tensor(0.0), mse_loss=th.tensor(0.0),
            mse_layer_sum=th.tensor(0.0), lambda_delta=0.0, use_mse_loss=False,
        )


def _build_tiny_trainer(lambda_delta: float, recon_loss_type=None) -> CrossCoderTrainer:
    activation_dim, dict_size = 8, 16
    activation_mean = th.zeros(2, activation_dim)
    activation_std = th.ones(2, activation_dim)
    return CrossCoderTrainer(
        activation_dim=activation_dim,
        dict_size=dict_size,
        lr=1e-3,
        l1_penalty=0.1,
        warmup_steps=1,
        layer=0,
        lm_name="tiny",
        device="cpu",
        lambda_delta=lambda_delta,
        recon_loss_type=recon_loss_type,
        activation_mean=activation_mean,
        activation_std=activation_std,
    )


def test_trainer_total_loss_includes_lambda_delta():
    th.manual_seed(0)
    trainer = _build_tiny_trainer(lambda_delta=2.0, recon_loss_type="mse_layer_sum")
    x = th.randn(4, 2, 8)
    log = trainer.loss(x, logging=True, normalize_activations=False)
    losses = log.losses
    recon = losses["mse_base_loss"] + losses["mse_ft_loss"]
    expected = recon + 0.1 * losses["sparsity_loss"] + 2.0 * losses["delta_loss"]
    assert abs(losses["loss"] - expected) < 1e-5, (losses, expected)


def test_trainer_lambda_zero_matches_no_delta_term():
    th.manual_seed(0)
    trainer = _build_tiny_trainer(lambda_delta=0.0, recon_loss_type="mse_layer_sum")
    x = th.randn(4, 2, 8)
    log = trainer.loss(x, logging=True, normalize_activations=False)
    losses = log.losses
    recon = losses["mse_base_loss"] + losses["mse_ft_loss"]
    expected = recon + 0.1 * losses["sparsity_loss"]
    assert abs(losses["loss"] - expected) < 1e-5


def test_trainer_rejects_bad_recon_loss_type():
    with pytest.raises(ValueError, match="recon_loss_type"):
        _build_tiny_trainer(lambda_delta=0.0, recon_loss_type="garbage")
