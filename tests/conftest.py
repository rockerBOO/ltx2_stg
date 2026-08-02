"""Puts the ComfyUI root and custom_nodes dir on sys.path so `import comfy` and
`import ltx2_stg` resolve the same way they do when ComfyUI loads this node pack.
"""

import sys
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent.parent
CUSTOM_NODES_DIR = PLUGIN_DIR.parent
COMFYUI_ROOT = CUSTOM_NODES_DIR.parent

for path in (COMFYUI_ROOT, CUSTOM_NODES_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

# comfy.model_management picks a device at import time (cpu_state, defaulting
# to CUDA) and probes it immediately for total VRAM. On a GPU-less CI runner
# that probe raises before any test code runs, so force CPU mode here, before
# `import ltx2_stg` pulls in comfy.model_patcher -> ... -> comfy.model_management.
import comfy.cli_args  # noqa: E402

comfy.cli_args.args.cpu = True
