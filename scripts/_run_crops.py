import torch
_original_load = torch.load
def _patched_load(*a, **kw):
    kw.setdefault('weights_only', False)
    return _original_load(*a, **kw)
torch.load = _patched_load

import sys, importlib
sys.argv = ['annotate_video.py', '--all', '--crops-only']

# Import and run main from the real script
spec = importlib.util.spec_from_file_location("annotate_video", "scripts/adapter/annotate_video.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
