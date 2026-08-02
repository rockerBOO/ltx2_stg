"""Spatio-Temporal Guidance for LTX-2 / LTX-2.3, matching the reference
implementation in Lightricks/LTX-2 (ltx_core.guidance.perturbations).

The stock SkipLayerGuidanceDiT node skips an *entire* transformer block, which is
a far stronger perturbation than LTX-2's STG. LTX-2 perturbs only the self-attention
inside the targeted block, and it does not zero it -- it replaces the attention
output with the value projection (identity attention):

    ltx_core/model/transformer/attention.py:558
        if not use_attention:
            out = v
        ...
        if self.to_gate_logits is not None:
            out = self.gated_attention_function(x, out, self)
        return self.to_out(out)

This node reproduces that, leaving the block's text cross-attention, a2v/v2a
cross-attention and feed-forward intact.

Blend math follows ltx_core.components.guiders.MultiModalGuider.calculate:
    pred = cfg_result + stg_scale * (cond - uncond_perturbed)
    factor = cond.std() / pred.std()
    factor = rescale * factor + (1 - rescale)
"""

import logging
import math
import os
import re

import torch

import comfy.model_patcher
import comfy.samplers

_LOG = logging.getLogger("ltx2_stg")
_DEBUG = os.environ.get("LTX2_STG_DEBUG", "0") == "1"
_seen = set()


def _log_once(key, msg):
    if _DEBUG and key not in _seen:
        _seen.add(key)
        _LOG.info("[ltx2_stg] %s", msg)


def reset_log():
    _seen.clear()


def _video_cut(conds):
    """Index splitting video from audio in the packed (B, 1, N) sampling tensor.

    samplers.py:1274 packs the AV NestedTensor via utils.pack_latents(), which
    reshapes each modality to (B, 1, -1) and cats on the last dim; it is only
    unpacked back to a NestedTensor at samplers.py:1327, AFTER sampling. So
    everything reaching a cfg/post_cfg hook is a plain flat tensor, and any
    `is_nested` check silently fails.

    LTXAV.extra_conds (model_base.py:1195) forwards latent_shapes as a
    CONDConstant, so the boundary is recoverable: prod(video_shape[1:]).
    Returns None for single-modality latents.
    """
    try:
        shapes = conds[0]["model_conds"]["latent_shapes"].cond
    except (TypeError, KeyError, IndexError, AttributeError):
        _log_once("nocut", "latent_shapes unavailable -> treating latent as single modality")
        return None
    if not shapes or len(shapes) < 2:
        return None
    return math.prod(shapes[0][1:])

# Perturbation targets, named after ltx_core.guidance.perturbations.PerturbationType
_TARGETS = {
    "video": ("attn1",),
    "audio": ("audio_attn1",),
    "video+audio": ("attn1", "audio_attn1"),
}


def _identity_attention(attn, x, context=None, mask=None, pe=None, k_pe=None, transformer_options={}, **kwargs):
    """attn1 with the attention matrix replaced by identity: out = to_out(gate(to_v(x)))."""
    context = x if context is None else context
    out = attn.to_v(context)

    if getattr(attn, "to_gate_logits", None) is not None:
        gate_logits = attn.to_gate_logits(x)
        b, t, _ = out.shape
        out = out.view(b, t, attn.heads, attn.dim_head)
        gates = 2.0 * torch.sigmoid(gate_logits)
        out = out * gates.unsqueeze(-1)
        out = out.view(b, t, attn.heads * attn.dim_head)

    return attn.to_out(out)


def _rescale(pred, cond_pred, rescaling_scale, cut):
    """Rescale toward cond's std, PER MODALITY.

    ltx_core runs a separate MultiModalGuider per modality, so each rescales
    against its own std (guiders.py:268). Applying one global factor across the
    packed video+audio tensor scales audio by video's statistics.
    """
    r = rescaling_scale
    if cut is None:
        factor = cond_pred.std() / pred.std()
        _log_once("rs1", f"rescale: single modality, factor={float(r * factor + (1 - r)):.4f}")
        return pred * (r * factor + (1 - r))

    out = pred.clone()
    factors = []
    for sl in (slice(None, cut), slice(cut, None)):
        p, c = pred[..., sl], cond_pred[..., sl]
        factor = r * (c.std() / p.std()) + (1 - r)
        out[..., sl] = p * factor
        factors.append(float(factor))
    _log_once("rs2", f"rescale: per-modality, video={factors[0]:.4f} audio={factors[1]:.4f}")
    return out


class _PerturbBlock:
    """patches_replace entry: run the real block with its self-attention perturbed."""

    def __init__(self, block, attn_names):
        self.block = block
        self.attns = [getattr(block, n) for n in attn_names if getattr(block, n, None) is not None]

    def __call__(self, args, extra_options):
        saved = []
        for attn in self.attns:
            saved.append(attn.__dict__.pop("forward", None))
            # bind the perturbed forward to this module instance
            attn.__dict__["forward"] = (lambda a: lambda *ar, **kw: _identity_attention(a, *ar, **kw))(attn)
        try:
            return extra_options["original_block"](args)
        finally:
            for attn, prev in zip(self.attns, saved):
                attn.__dict__.pop("forward", None)
                if prev is not None:
                    attn.__dict__["forward"] = prev


class LTXVMultiModalGuidance:
    """STG + isolated-modality guidance, blended exactly as MultiModalGuider.calculate.

    The reference (ltx_core/components/guiders.py:262) is a single expression with
    ONE rescale over the fully blended prediction:

        pred = cond + (cfg-1)*(cond-uncond) + stg*(cond-ptb) + (mod-1)*(cond-modpass)
        factor = cond.std() / pred.std()
        pred *= rescale*factor + (1-rescale)

    Comfy's cfg_result already equals `cond + (cfg-1)*(cond-uncond)`, so both extra
    terms are added here and rescaled once -- chaining them as separate post-cfg
    hooks would rescale in the middle and diverge.

    The modality pass (denoisers.py:121) uses the POSITIVE context with
    SKIP_A2V_CROSS_ATTN + SKIP_V2A_CROSS_ATTN on blocks=None (all blocks).
    ltx_core skips the residual entirely when cross_attn_skip_all is set
    (transformer.py:334/365); comfy's a2v_cross_attn/v2a_cross_attn flags
    (av_model.py:267) short-circuit the same section, so the two match.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "stg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
                "stg_blocks": ("STRING", {"default": "28", "tooltip": "LTX-2.3 uses 28, LTX-2.0 uses 29."}),
                "stg_perturb": (["video+audio", "video", "audio"], {"default": "video+audio"}),
                "modality_scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "Cross-modal (a2v/v2a) guidance. 1.0 disables the pass."}),
                "rescaling_scale": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "advanced/guidance"
    DESCRIPTION = "LTX-2 STG (self-attention identity perturbation) + isolated-modality guidance, with a single combined rescale."

    def apply(self, model, stg_scale, stg_blocks, stg_perturb, modality_scale,
              rescaling_scale, start_percent=0.0, end_percent=1.0):
        do_stg = stg_scale != 0.0
        do_mod = modality_scale != 1.0
        if not do_stg and not do_mod:
            return (model,)

        patches = {}
        if do_stg:
            indices = [int(i) for i in re.findall(r"\d+", stg_blocks)]
            if not indices:
                do_stg = False
            else:
                transformer_blocks = model.get_model_object("diffusion_model.transformer_blocks")
                n_blocks = len(transformer_blocks)
                for i in indices:
                    if i >= n_blocks:
                        raise ValueError(
                            f"stg block index {i} out of range; model has {n_blocks} transformer blocks (0-{n_blocks - 1})"
                        )
                attn_names = _TARGETS[stg_perturb]
                patches = {i: _PerturbBlock(transformer_blocks[i], attn_names) for i in indices}

        model_sampling = model.get_model_object("model_sampling")
        sigma_start = model_sampling.percent_to_sigma(start_percent)
        sigma_end = model_sampling.percent_to_sigma(end_percent)

        def post_cfg_function(args):
            inner_model = args["model"]
            cond = args["cond"]
            cond_pred = args["cond_denoised"]
            cfg_result = args["denoised"]
            sigma = args["sigma"]
            x = args["input"]

            if not (sigma_end <= sigma[0].item() <= sigma_start):
                return cfg_result

            base_options = args["model_options"]
            pred = cfg_result

            if do_stg:
                mo = base_options.copy()
                for i, patch in patches.items():
                    mo = comfy.model_patcher.set_model_options_patch_replace(mo, patch, "dit", "double_block", i)
                (ptb,) = comfy.samplers.calc_cond_batch(inner_model, [cond], x, sigma, mo)
                pred = pred + (cond_pred - ptb) * stg_scale

            if do_mod:
                mo = base_options.copy()
                to = mo.get("transformer_options", {}).copy()
                to["a2v_cross_attn"] = False
                to["v2a_cross_attn"] = False
                mo["transformer_options"] = to
                (modp,) = comfy.samplers.calc_cond_batch(inner_model, [cond], x, sigma, mo)
                pred = pred + (cond_pred - modp) * (modality_scale - 1.0)

            if rescaling_scale != 0:
                pred = _rescale(pred, cond_pred, rescaling_scale, _video_cut(args["cond"]))
            return pred

        m = model.clone()
        m.set_model_sampler_post_cfg_function(post_cfg_function)
        return (m,)


class LTXVMultiModalCFG:
    """Separate CFG scales for the video and audio halves of the AV latent.

    CFGGuider applies one scalar to the whole denoised tensor, but LTX-2 uses
    cfg_scale=3.0 for video and 7.0 for audio (constants.py LTX_2_3_PARAMS).
    LTXVConcatAVLatent packs the two modalities as NestedTensor((video, audio)),
    and NestedTensor.__mul__ pairs elementwise against another NestedTensor, so
    the split blend is just:

        uncond + (cond - uncond) * NestedTensor((video_cfg, audio_cfg))

    Overrides sampler_cfg_function. Sets disable_cfg1_optimization so the uncond
    pass runs even if the upstream CFGGuider widget is left at 1.0 -- otherwise
    samplers.py:610 short-circuits and cfg_function is never reached.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "video_cfg": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "audio_cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "advanced/guidance"
    DESCRIPTION = "Per-modality CFG for LTX-2 AV latents (video 3.0 / audio 7.0). The upstream CFGGuider's cfg value is ignored."

    def apply(self, model, video_cfg, audio_cfg):
        def cfg_fn(args):
            x = args["input"]
            cond_pred = args["cond_denoised"]
            uncond_pred = args["uncond_denoised"]

            if uncond_pred is None:
                return x - cond_pred

            delta = cond_pred - uncond_pred
            cut = _video_cut(args["input_cond"])
            if cut is None:
                _log_once("cfg1", f"cfg: single modality, scale={video_cfg}")
                delta = delta * video_cfg
            else:
                total = delta.shape[-1]
                _log_once(
                    "cfg2",
                    f"cfg: split at {cut}/{total} -> video={video_cfg} "
                    f"({cut} elems), audio={audio_cfg} ({total - cut} elems)",
                )
                delta = delta.clone()
                delta[..., :cut] *= video_cfg
                delta[..., cut:] *= audio_cfg

            # cfg_function does `cfg_result = x - sampler_cfg_function(args)`
            return x - (uncond_pred + delta)

        m = model.clone()
        m.set_model_sampler_cfg_function(cfg_fn)
        m.model_options["disable_cfg1_optimization"] = True
        return (m,)


NODE_CLASS_MAPPINGS = {
    "LTXVMultiModalGuidance": LTXVMultiModalGuidance,
    "LTXVMultiModalCFG": LTXVMultiModalCFG,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXVMultiModalGuidance": "LTX-2 Multi-Modal Guidance (STG + modality)",
    "LTXVMultiModalCFG": "LTX-2 Multi-Modal CFG (video/audio)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
