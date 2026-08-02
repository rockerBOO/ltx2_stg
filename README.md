# ltx2_stg

Spatio-Temporal Guidance (STG) and per-modality CFG/guidance nodes for **LTX-2 / LTX-2.3** in ComfyUI, matching the reference math in `Lightricks/LTX-2` (`ltx_core.guidance.perturbations`, `ltx_core.components.guiders`).

ComfyUI's stock `SkipLayerGuidanceDiT` skips an entire transformer block, which is a much stronger perturbation than LTX-2's STG. LTX-2 instead perturbs *only* the self-attention inside the targeted block(s), replacing the attention output with the value projection (identity attention) while leaving text cross-attention, audio/video cross-attention, and the feed-forward layer intact. This node pack reproduces that behavior, plus the isolated-modality guidance pass and per-modality CFG split that LTX-2's AV pipeline uses.

## Nodes

### LTX-2 Multi-Modal Guidance (STG + modality)

`LTXVMultiModalGuidance` — combines STG and cross-modal (audio↔video) guidance into a single post-CFG pass with one shared rescale, matching `MultiModalGuider.calculate`.

| Input | Description |
|---|---|
| `model` | The LTX-2 model to patch. |
| `stg_scale` | STG strength. `0.0` disables STG. |
| `stg_blocks` | Space/comma separated transformer block indices to perturb (e.g. `28`). LTX-2.3 uses 28 blocks, LTX-2.0 uses 29. |
| `stg_perturb` | Which attention to perturb: `video`, `audio`, or `video+audio`. |
| `modality_scale` | Cross-modal (a2v/v2a) guidance strength. `1.0` disables the pass. |
| `rescaling_scale` | Blend between the raw and std-rescaled prediction (`0` = raw, `1` = fully rescaled). |
| `start_percent` / `end_percent` | Sampling window (as percent of the schedule) over which guidance is applied. |

### LTX-2 Multi-Modal CFG (video/audio)

`LTXVMultiModalCFG` — applies separate CFG scales to the video and audio halves of a packed AV latent (LTX-2 defaults: `video_cfg=3.0`, `audio_cfg=7.0`), instead of the single scalar CFG that ComfyUI's `CFGGuider` applies to the whole tensor. This overrides the sampler's CFG function directly, so the upstream `CFGGuider` widget value is ignored once this node is used.

## Example workflow

[`example_workflows/ltx2_3_i2v_no_distill.json`](example_workflows/ltx2_3_i2v_no_distill.json) — LTX-2.3 image-to-video, undistilled, using both `LTXVMultiModalGuidance` and `LTXVMultiModalCFG`. Drag it into ComfyUI or load it via the workflow menu.

## Installation

This is a ComfyUI custom node — it needs to live inside your ComfyUI installation's `custom_nodes` directory and depends on `torch` and `comfy`, both of which ComfyUI already provides. There is nothing extra to install for normal use.

### Via git (recommended)

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/rockerBOO/ltx2_stg.git
```

Restart ComfyUI. The nodes will appear under `advanced/guidance`.

### Manual

Copy this directory into `ComfyUI/custom_nodes/ltx2_stg` and restart ComfyUI.

## Development

Install test dependencies into the same Python environment ComfyUI runs with (adjust the venv path for your setup):

```bash
uv pip install --python /path/to/ComfyUI/.venv/bin/python pytest
```

Run the tests from this directory:

```bash
/path/to/ComfyUI/.venv/bin/python -m pytest
```

The tests exercise the guidance math (`_video_cut`, `_rescale`, `_identity_attention`), the block-patching context manager (`_PerturbBlock`), and both nodes' `apply()` behavior using lightweight fakes — they don't require loading a real LTX-2 checkpoint. `conftest.py` adds your ComfyUI installation's root directory to `sys.path` so the real `comfy.model_patcher` / `comfy.samplers` modules import the same way they do when ComfyUI loads this node pack.

## Debug logging

Set `LTX2_STG_DEBUG=0` to silence the one-time info logs this pack emits (block-cut detection, rescale factors, CFG split points). Logging is on by default.
