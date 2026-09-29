"""
Coverage for the peak-list ("sparse") input pathway: pad_peaks_to_fixed_length
(train.py) and XRDTransformerEncoder's input_type='sparse' branch (model.py).

This pathway was previously wired up incorrectly (model never actually built
in sparse mode, intensity embedding frozen via no_grad/detach, no padding
mask) -- these tests pin down the fixed behavior so it doesn't regress.
"""
import pytest
import torch
from omegaconf import OmegaConf

from train import build_model, pad_peaks_to_fixed_length
from model import XRDTransformerEncoder


# ---------------------------------------------------------------------------
# pad_peaks_to_fixed_length
# ---------------------------------------------------------------------------

def test_pad_peaks_pads_with_sentinel_when_fewer_than_max():
    peaks = [(30.0, 0.5), (10.0, 0.9), (50.0, 0.2)]
    out = pad_peaks_to_fixed_length(list(peaks), max_peaks=5)
    assert len(out) == 5
    assert out[3] == (-1.0, -1.0)
    assert out[4] == (-1.0, -1.0)


def test_pad_peaks_sorts_ascending_by_position():
    peaks = [(30.0, 0.5), (10.0, 0.9), (50.0, 0.2)]
    out = pad_peaks_to_fixed_length(list(peaks), max_peaks=5)
    assert [p[0] for p in out[:3]] == [10.0, 30.0, 50.0]


def test_pad_peaks_truncates_to_lowest_angle_peaks_when_more_than_max():
    # 10 peaks; only the 5 lowest-2theta should survive (not the 5 most intense).
    peaks = [(float(i), 1.0 / (i + 1)) for i in range(10, 0, -1)]
    out = pad_peaks_to_fixed_length(list(peaks), max_peaks=5)
    assert len(out) == 5
    assert [p[0] for p in out] == [1.0, 2.0, 3.0, 4.0, 5.0]


# ---------------------------------------------------------------------------
# XRDTransformerEncoder sparse forward pass
# ---------------------------------------------------------------------------

def _make_sparse_model(use_intensity, max_peaks=6, d_model=16):
    return XRDTransformerEncoder(
        out_dim=(10, 35), d_model=d_model, h_dim=32, n_head=2,
        n_self_layer=1, n_cross_layer=1, input_type='sparse',
        max_peaks=max_peaks, use_intensity=use_intensity,
    )


def _make_batch(batch_size, max_peaks):
    xrd = torch.full((batch_size, max_peaks, 2), -1.0)
    for b in range(batch_size):
        n_real = 2 + b
        xrd[b, :n_real, 0] = torch.linspace(10, 40, n_real)
        xrd[b, :n_real, 1] = torch.rand(n_real)
    return xrd


@pytest.mark.parametrize("use_intensity", [True, False])
def test_sparse_forward_produces_finite_output(use_intensity):
    torch.manual_seed(0)
    model = _make_sparse_model(use_intensity)
    xrd = _make_batch(batch_size=4, max_peaks=6)
    out = model(xrd)
    assert out.shape == (4, 350)
    assert torch.isfinite(out).all()


def test_sparse_intensity_embedding_receives_gradients():
    # Regression test: this embedding used to be computed inside torch.no_grad()
    # and .detach()ed in train.py, so it could never learn.
    torch.manual_seed(0)
    model = _make_sparse_model(use_intensity=True)
    xrd = _make_batch(batch_size=4, max_peaks=6)
    out = model(xrd)
    out.sum().backward()
    assert model.intensity_embed.weight.grad is not None
    assert torch.any(model.intensity_embed.weight.grad != 0)


def test_use_intensity_false_has_no_intensity_embed_module():
    model = _make_sparse_model(use_intensity=False)
    assert not hasattr(model, 'intensity_embed')


def test_sparse_padding_is_masked_not_attended_to():
    # Regression test: padded peak slots previously had no mask at all, so
    # sparse_attn/self_attn_blocks/pool_cross_block attended over them as if
    # real. Changing the (garbage) values in padded slots must not change the
    # output, as long as they still carry the padding sentinel (position < 0).
    torch.manual_seed(0)
    model = _make_sparse_model(use_intensity=True)
    model.eval()
    xrd = _make_batch(batch_size=4, max_peaks=6)

    with torch.no_grad():
        out1 = model(xrd)
        perturbed = xrd.clone()
        pad_mask = xrd[:, :, 0] < 0
        perturbed[:, :, 0][pad_mask] = -50.0  # still a negative sentinel
        perturbed[:, :, 1][pad_mask] = -50.0
        out2 = model(perturbed)

    torch.testing.assert_close(out1, out2, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# build_model() config wiring
# ---------------------------------------------------------------------------

def _base_model_cfg(**overrides):
    cfg = {
        "type": "transformerbispec",
        "out_dim": [10, 35],
        "d_model": 16,
        "h_dim": 32,
        "n_head": 2,
        "n_self_layer": 1,
        "n_cross_layer": 1,
        "transformer_proc": "lin_layer",
        "attn_pdrop": 0.0,
        "resid_pdrop": 0.0,
    }
    cfg.update(overrides)
    return OmegaConf.create({"model": cfg})


# build_model() always calls .cuda() on the constructed model (pre-existing
# behavior, unrelated to this pathway) so these two need a GPU.
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="build_model() requires CUDA")


@requires_cuda
def test_build_model_defaults_to_dense_when_input_type_absent():
    # Regression test: build_model() used to never read cfg.model.input_type
    # at all, so setting indices=true in a data config would build a dense
    # model and crash on the first sparse-shaped batch. This pins the
    # backward-compatible default for configs that predate input_type.
    cfg = _base_model_cfg()
    model = build_model(cfg, allowed_indices=None)
    assert model.input_type == 'dense'


@requires_cuda
def test_build_model_honors_sparse_input_type_and_max_peaks():
    cfg = _base_model_cfg(input_type="sparse", max_peaks=25, use_intensity=False)
    model = build_model(cfg, allowed_indices=None)
    assert model.input_type == 'sparse'
    assert model.seq_len == 25
    assert not hasattr(model, 'intensity_embed')
