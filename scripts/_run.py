"""Wrapper: patch torch.load before ultralytics imports it."""
import torch
_orig = torch.load
def _patched(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig(*a, **kw)
torch.load = _patched

import sys, runpy
# Keep original argv[0] as the script name so argparse works
sys.argv = ["scripts/adapter/annotate_video.py"] + sys.argv[1:]
runpy.run_path("scripts/adapter/annotate_video.py", run_name="__main__")
