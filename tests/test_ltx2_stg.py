"""Unit tests for ltx2_stg's guidance math and comfy plumbing.

Run from anywhere with: pytest custom_nodes/ltx2_stg/tests -v
(conftest.py wires ComfyUI's root and custom_nodes dir onto sys.path so the
real `comfy.model_patcher` / `comfy.samplers` modules import cleanly, the same
way ComfyUI itself loads this node pack.)
"""

from types import SimpleNamespace

import pytest
import torch

import ltx2_stg as node


# ---------------------------------------------------------------------------
# _video_cut
# ---------------------------------------------------------------------------


class _Cond:
    def __init__(self, cond):
        self.cond = cond


def _conds_with_shapes(shapes):
    return [{"model_conds": {"latent_shapes": _Cond(shapes)}}]


def test_video_cut_missing_latent_shapes_returns_none():
    assert node._video_cut([{"model_conds": {}}]) is None


def test_video_cut_none_input_returns_none():
    assert node._video_cut(None) is None


def test_video_cut_single_modality_returns_none():
    assert node._video_cut(_conds_with_shapes([[1, 32, 10]])) is None


def test_video_cut_two_modalities_returns_product_of_video_shape():
    # shapes[0] = [B, 32, 10] -> product of dims after batch = 320
    assert node._video_cut(_conds_with_shapes([[1, 32, 10], [1, 8, 5]])) == 320


# ---------------------------------------------------------------------------
# _rescale
# ---------------------------------------------------------------------------


def test_rescale_single_modality_matches_reference_formula():
    torch.manual_seed(0)
    pred = torch.randn(2, 8)
    cond_pred = torch.randn(2, 8)
    r = 0.7

    out = node._rescale(pred, cond_pred, r, cut=None)

    factor = cond_pred.std() / pred.std()
    expected = pred * (r * factor + (1 - r))
    assert torch.allclose(out, expected)


def test_rescale_per_modality_splits_at_cut():
    torch.manual_seed(1)
    pred = torch.randn(2, 10)
    cond_pred = torch.randn(2, 10)
    r = 0.5
    cut = 4

    out = node._rescale(pred, cond_pred, r, cut)

    video_factor = r * (cond_pred[..., :cut].std() / pred[..., :cut].std()) + (1 - r)
    audio_factor = r * (cond_pred[..., cut:].std() / pred[..., cut:].std()) + (1 - r)
    expected = pred.clone()
    expected[..., :cut] *= video_factor
    expected[..., cut:] *= audio_factor
    assert torch.allclose(out, expected)


# ---------------------------------------------------------------------------
# _identity_attention
# ---------------------------------------------------------------------------


class _FakeAttnNoGate:
    to_gate_logits = None
    heads = 2
    dim_head = 3

    def to_v(self, x):
        return x

    def to_out(self, x):
        return x * 2


def test_identity_attention_without_gate_is_to_out_of_to_v():
    attn = _FakeAttnNoGate()
    x = torch.randn(1, 4, 6)
    out = node._identity_attention(attn, x)
    assert torch.allclose(out, x * 2)


class _FakeAttnGated:
    heads = 2
    dim_head = 3

    def __init__(self, gate_logit):
        self._gate_logit = gate_logit
        self.to_gate_logits = self._gate_fn

    def to_v(self, x):
        return x

    def to_out(self, x):
        return x

    def _gate_fn(self, x):
        b, t, _ = x.shape
        return torch.full((b, t, self.heads), self._gate_logit)


def test_identity_attention_gate_zero_logit_is_near_identity():
    # sigmoid(0) * 2 == 1.0, so gating should be a no-op here.
    attn = _FakeAttnGated(gate_logit=0.0)
    x = torch.randn(1, 4, 6)
    out = node._identity_attention(attn, x)
    assert torch.allclose(out, x, atol=1e-6)


def test_identity_attention_negative_gate_logit_shrinks_output():
    attn = _FakeAttnGated(gate_logit=-100.0)  # sigmoid ~ 0 -> gates ~ 0
    x = torch.randn(1, 4, 6)
    out = node._identity_attention(attn, x)
    assert torch.allclose(out, torch.zeros_like(x), atol=1e-4)


# ---------------------------------------------------------------------------
# _PerturbBlock
# ---------------------------------------------------------------------------


class _FakeAttn1:
    to_gate_logits = None
    heads = 1
    dim_head = 3

    def to_v(self, x):
        return x + 100

    def to_out(self, x):
        return x

    def forward(self, *args, **kwargs):
        return "original-forward"


def test_perturb_block_patches_forward_during_call_and_restores_after():
    block = SimpleNamespace(attn1=_FakeAttn1())
    perturb = node._PerturbBlock(block, ("attn1",))
    assert perturb.attns == [block.attn1]

    captured = {}

    def original_block(args):
        captured["during"] = block.attn1.forward(args["x"])
        return "block-result"

    x = torch.zeros(1, 2, 3)
    result = perturb({"x": x}, {"original_block": original_block})

    assert result == "block-result"
    assert torch.allclose(captured["during"], x + 100)
    # forward is restored to the original (unbound-lookup) class method
    assert block.attn1.forward() == "original-forward"


def test_perturb_block_restores_forward_even_if_original_block_raises():
    block = SimpleNamespace(attn1=_FakeAttn1())
    perturb = node._PerturbBlock(block, ("attn1",))

    def original_block(args):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        perturb({}, {"original_block": original_block})

    assert block.attn1.forward() == "original-forward"


# ---------------------------------------------------------------------------
# LTXVMultiModalGuidance
# ---------------------------------------------------------------------------


def test_guidance_input_types_has_expected_widgets():
    types = node.LTXVMultiModalGuidance.INPUT_TYPES()
    assert set(types["required"]) == {
        "model",
        "stg_scale",
        "stg_blocks",
        "stg_perturb",
        "modality_scale",
        "rescaling_scale",
    }


def test_guidance_apply_is_noop_when_stg_and_modality_disabled():
    model = object()
    result = node.LTXVMultiModalGuidance().apply(
        model,
        stg_scale=0.0,
        stg_blocks="28",
        stg_perturb="video+audio",
        modality_scale=1.0,
        rescaling_scale=0.7,
    )
    assert result == (model,)


class _FakeModelSampling:
    def percent_to_sigma(self, percent):
        return 1.0 - percent


class _FakeModel:
    def __init__(self, n_blocks):
        self.blocks = [
            SimpleNamespace(attn1=_FakeAttn1(), audio_attn1=None) for _ in range(n_blocks)
        ]

    def get_model_object(self, name):
        if name == "diffusion_model.transformer_blocks":
            return self.blocks
        if name == "model_sampling":
            return _FakeModelSampling()
        raise KeyError(name)


def test_guidance_apply_raises_on_out_of_range_block_index():
    model = _FakeModel(n_blocks=5)
    with pytest.raises(ValueError, match="out of range"):
        node.LTXVMultiModalGuidance().apply(
            model,
            stg_scale=1.0,
            stg_blocks="99",
            stg_perturb="video",
            modality_scale=1.0,
            rescaling_scale=0.7,
        )


# ---------------------------------------------------------------------------
# LTXVMultiModalCFG
# ---------------------------------------------------------------------------


class _FakeModel2:
    def __init__(self):
        self.model_options = {}
        self.cfg_fn = None

    def clone(self):
        return self

    def set_model_sampler_cfg_function(self, fn):
        self.cfg_fn = fn


def test_cfg_apply_sets_disable_cfg1_optimization_and_returns_clone():
    model = _FakeModel2()
    (result,) = node.LTXVMultiModalCFG().apply(model, video_cfg=2.0, audio_cfg=5.0)
    assert result is model
    assert model.model_options["disable_cfg1_optimization"] is True
    assert callable(model.cfg_fn)


def test_cfg_fn_without_uncond_falls_back_to_x_minus_cond():
    (result,) = node.LTXVMultiModalCFG().apply(_FakeModel2(), video_cfg=2.0, audio_cfg=5.0)
    cfg_fn = result.cfg_fn

    x = torch.randn(2, 3)
    cond_pred = torch.randn(2, 3)
    out = cfg_fn({"input": x, "cond_denoised": cond_pred, "uncond_denoised": None, "input_cond": None})
    assert torch.allclose(out, x - cond_pred)


def test_cfg_fn_single_modality_scales_by_video_cfg():
    (result,) = node.LTXVMultiModalCFG().apply(_FakeModel2(), video_cfg=2.0, audio_cfg=5.0)
    cfg_fn = result.cfg_fn

    torch.manual_seed(2)
    x = torch.randn(2, 3)
    cond_pred = torch.randn(2, 3)
    uncond_pred = torch.randn(2, 3)

    out = cfg_fn(
        {
            "input": x,
            "cond_denoised": cond_pred,
            "uncond_denoised": uncond_pred,
            "input_cond": None,
        }
    )

    delta = (cond_pred - uncond_pred) * 2.0
    expected = x - (uncond_pred + delta)
    assert torch.allclose(out, expected)


def test_cfg_fn_multi_modality_splits_scale_at_video_audio_boundary():
    (result,) = node.LTXVMultiModalCFG().apply(_FakeModel2(), video_cfg=2.0, audio_cfg=5.0)
    cfg_fn = result.cfg_fn

    torch.manual_seed(3)
    x = torch.randn(1, 6)
    cond_pred = torch.randn(1, 6)
    uncond_pred = torch.randn(1, 6)
    input_cond = _conds_with_shapes([[1, 4], [1, 2]])  # cut at 4

    out = cfg_fn(
        {
            "input": x,
            "cond_denoised": cond_pred,
            "uncond_denoised": uncond_pred,
            "input_cond": input_cond,
        }
    )

    delta = (cond_pred - uncond_pred).clone()
    delta[..., :4] *= 2.0
    delta[..., 4:] *= 5.0
    expected = x - (uncond_pred + delta)
    assert torch.allclose(out, expected)


# ---------------------------------------------------------------------------
# Node registration
# ---------------------------------------------------------------------------


def test_node_mappings_are_registered():
    assert node.NODE_CLASS_MAPPINGS["LTXVMultiModalGuidance"] is node.LTXVMultiModalGuidance
    assert node.NODE_CLASS_MAPPINGS["LTXVMultiModalCFG"] is node.LTXVMultiModalCFG
    assert set(node.NODE_CLASS_MAPPINGS) == set(node.NODE_DISPLAY_NAME_MAPPINGS)
