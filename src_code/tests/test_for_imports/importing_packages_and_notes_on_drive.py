"""
requirements.txt  — identical on all three environments
python==3.11.15
CUDA Toolkit==11.8

numpy==2.1.3
scipy==1.14.1
opt_einsum==3.4.0
matplotlib==3.9.4
tqdm==4.67.1
h5py==3.12.1
pytest==8.4.2
pytorch==2.5.1

NOTE: cuDNN is bundled inside the PyTorch 2.5.1 wheel (ships cuDNN 9.1.0).
No separate cuDNN installation is required.

# --- Laptop (MX250, CC 6.1, Ubuntu 24.04, driver 580 supports CUDA >=11.8) ---
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118


# --- Izar (V100, CC 7.0) ---
# Load modules first (adjust to what EPFL exposes):
module load gcc python/3.11.x cuda/11.8
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118


# --- Jed (CPU-only, Intel Ice Lake) ---
module load gcc python/3.11.x
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

"""

# ===========================================================================
# Run directly:  python importing_packages_and_notes_on_drive.py
# Or via pytest: pytest importing_packages_and_notes_on_drive.py -v
# ===========================================================================

import sys

EXPECTED = {
    "python"    : (3, 11),
    "torch"     : "2.5.1",       # compared after stripping +cu118 / +cpu suffix
    "numpy"     : "2.1.3",
    "scipy"     : "1.14.1",
    "opt_einsum": "3.4.0",
    "matplotlib": "3.9.4",
    "tqdm"      : "4.67.1",
    "h5py"      : "3.12.1",
    "pytest"    : "8.4.2",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

def ok(msg):  print(f"  {GREEN}PASS{RESET}  {msg}")
def fail(msg): print(f"  {RED}FAIL{RESET}  {msg}")
def info(msg): print(f"  {YELLOW}INFO{RESET}  {msg}")


# ---------------------------------------------------------------------------
# Pytest-compatible test functions (also called from __main__ below)
# ---------------------------------------------------------------------------

def test_python_version():
    major, minor = sys.version_info[:2]
    exp_major, exp_minor = EXPECTED["python"]
    assert (major, minor) == (exp_major, exp_minor), (
        f"Python {major}.{minor} found; expected {exp_major}.{exp_minor}"
    )


def test_torch():
    import torch
    # torch.__version__ can be '2.5.1+cu118' or '2.5.1+cpu' — strip the build suffix
    version_base = torch.__version__.split("+")[0]
    assert version_base == EXPECTED["torch"].split("+")[0], (
        f"torch {torch.__version__} found; expected {EXPECTED['torch']}"
    )


def test_numpy():
    import numpy as np
    assert np.__version__ == EXPECTED["numpy"], (
        f"numpy {np.__version__} found; expected {EXPECTED['numpy']}"
    )


def test_scipy():
    import scipy
    assert scipy.__version__ == EXPECTED["scipy"], (
        f"scipy {scipy.__version__} found; expected {EXPECTED['scipy']}"
    )


def test_opt_einsum():
    import opt_einsum
    assert opt_einsum.__version__ == EXPECTED["opt_einsum"], (
        f"opt_einsum {opt_einsum.__version__} found; expected {EXPECTED['opt_einsum']}"
    )


def test_matplotlib():
    import matplotlib
    assert matplotlib.__version__ == EXPECTED["matplotlib"], (
        f"matplotlib {matplotlib.__version__} found; expected {EXPECTED['matplotlib']}"
    )


def test_tqdm():
    import tqdm
    assert tqdm.__version__ == EXPECTED["tqdm"], (
        f"tqdm {tqdm.__version__} found; expected {EXPECTED['tqdm']}"
    )


def test_h5py():
    import h5py
    assert h5py.__version__ == EXPECTED["h5py"], (
        f"h5py {h5py.__version__} found; expected {EXPECTED['h5py']}"
    )


def test_pytest():
    import pytest
    assert pytest.__version__ == EXPECTED["pytest"], (
        f"pytest {pytest.__version__} found; expected {EXPECTED['pytest']}"
    )


def test_cuda_and_cudnn():
    """Non-fatal on CPU-only machines (Jed). Reports GPU/CUDA/cuDNN status.

    cuDNN is bundled inside the PyTorch wheel — no separate install needed.
    Version encoding: cuDNN <9  → MAJOR*1000 + MINOR*100 + PATCH
                      cuDNN >=9 → MAJOR*10000 + MINOR*100 + PATCH
    """
    import torch
    if not torch.cuda.is_available():
        info("CUDA not available on this machine (CPU-only node) — skipping GPU checks.")
        return
    cuda_ver    = torch.version.cuda
    device_name = torch.cuda.get_device_name(0)
    cudnn_ver   = torch.backends.cudnn.version()  # e.g. 90100 for bundled 9.1.0
    if cudnn_ver >= 10000:
        cudnn_major = cudnn_ver // 10000
        cudnn_minor = (cudnn_ver % 10000) // 100
        cudnn_patch = cudnn_ver % 100
    else:
        cudnn_major = cudnn_ver // 1000
        cudnn_minor = (cudnn_ver % 1000) // 100
        cudnn_patch = cudnn_ver % 100
    info(f"GPU         : {device_name}")
    info(f"CUDA runtime: {cuda_ver}  (expected: 11.8)")
    info(f"cuDNN       : {cudnn_major}.{cudnn_minor}.{cudnn_patch}  (bundled in PyTorch wheel)")
    assert cuda_ver.startswith("11.8"), (
        f"CUDA {cuda_ver} found; expected 11.8.x"
    )
    # cuDNN version is bundled by PyTorch — assert it is present (>= 8.0)
    assert cudnn_major >= 8, (
        f"cuDNN {cudnn_major}.{cudnn_minor}.{cudnn_patch} found; expected >= 8.0"
    )


def test_torch_tensor_ops():
    """Sanity-check that basic PyTorch tensor operations work."""
    import torch
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    assert torch.allclose(a + b, torch.tensor([5.0, 7.0, 9.0]))
    assert torch.allclose(torch.dot(a, b), torch.tensor(32.0))


def test_autograd():
    """Verify that PyTorch autograd (the AD engine) is functional."""
    import torch
    x = torch.tensor(3.0, requires_grad=True)
    y = x ** 2 + 2 * x + 1          # y = (x+1)^2,  dy/dx = 2*(x+1) = 8
    y.backward()
    assert torch.isclose(x.grad, torch.tensor(8.0)), (
        f"Autograd returned {x.grad}; expected 8.0"
    )


# ---------------------------------------------------------------------------
# __main__ runner — prints a human-readable summary
# ---------------------------------------------------------------------------

TESTS = [
    test_python_version,
    test_torch,
    test_numpy,
    test_scipy,
    test_opt_einsum,
    test_matplotlib,
    test_tqdm,
    test_h5py,
    test_pytest,
    test_cuda_and_cudnn,
    test_torch_tensor_ops,
    test_autograd,
]

if __name__ == "__main__":
    print(f"\nPython executable : {sys.executable}")
    print(f"Python version    : {sys.version}\n")

    passed = 0
    failed = 0
    for test in TESTS:
        name = test.__name__
        try:
            test()
            ok(name)
            passed += 1
        except Exception as e:
            fail(f"{name}  →  {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"  {passed} passed   {failed} failed   out of {len(TESTS)} checks")
    print(f"{'='*50}\n")
    sys.exit(0 if failed == 0 else 1)