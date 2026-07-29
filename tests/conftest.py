"""
conftest.py — Mock torch to prevent import hang during test collection.
DQN tests are already marked with @pytest.mark.skipif(not TORCH_AVAILABLE).
"""
import sys

# Pre-import torch mock to prevent hang
class _FakeTorch:
    pass

for mod in ['torch', 'torch.nn', 'torch.optim', 'torch.utils', 'torch.nn.utils']:
    if mod not in sys.modules:
        sys.modules[mod] = _FakeTorch()
