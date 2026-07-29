"""
conftest.py — Mock torch to prevent import hang during test collection.
Only mocks torch when it is NOT actually installed.
DQN tests are already marked with @pytest.mark.skipif(not TORCH_AVAILABLE).
"""
import sys

# Only mock torch if it's not actually installed
try:
    import torch  # noqa: F401
    _TORCH_INSTALLED = True
except ImportError:
    _TORCH_INSTALLED = False

if not _TORCH_INSTALLED:
    class _FakeTorch:
        pass

    for mod in ['torch', 'torch.nn', 'torch.optim', 'torch.utils', 'torch.nn.utils']:
        if mod not in sys.modules:
            sys.modules[mod] = _FakeTorch()
