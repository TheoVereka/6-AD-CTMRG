"""
test_core_computation.py
========================
Computational sanity tests for the packages used in AD-CTMRG on a Honeycomb
lattice. Every test exercises a capability that will be called in production:

  1.  numpy   — SVD, eigh, einsum correctness
  2.  scipy   — LAPACK SVD (reference for debugging), eigsh
  3.  opt_einsum — contraction-path optimisation on a rank-4 tensor network
  4.  torch   — dtype coverage (real & complex), GPU availability
  5.  torch autograd through einsum   — gradient of a tensor contraction
  6.  torch autograd through SVD      — THE critical test: differentiating
                                        through truncated SVD (core of AD-CTMRG)
  7.  torch autograd through eigh     — differentiating through Hermitian eigen-
                                        decomposition (corner diagonalisation)
  8.  h5py    — round-trip checkpoint: save tensors to HDF5, reload, verify

Run directly:  python test_core_computation.py
Via pytest:    pytest test_core_computation.py -v
"""

import sys
import os
import tempfile

import numpy as np
import torch

# ── colour helpers ───────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}PASS{RESET}  {msg}")
def fail(msg): print(f"  {RED}FAIL{RESET}  {msg}")
def info(msg): print(f"  {YELLOW}INFO{RESET}  {msg}")
def head(msg): print(f"\n{BLUE}── {msg}{RESET}")


# ════════════════════════════════════════════════════════════════════════════
# 1. NumPy — SVD, eigh, einsum
# ════════════════════════════════════════════════════════════════════════════

def test_numpy_svd():
    """SVD of a random matrix: reconstruction and orthogonality of U, Vh."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((8, 5))
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    A_rec = (U * s) @ Vh
    assert np.allclose(A, A_rec, atol=1e-12), "SVD reconstruction failed"
    assert np.allclose(U.T @ U, np.eye(5), atol=1e-12), "U not orthonormal"
    assert np.allclose(Vh @ Vh.T, np.eye(5), atol=1e-12), "Vh not orthonormal"
    assert np.all(np.diff(s) <= 0), "Singular values not sorted descending"


def test_numpy_eigh():
    """eigh on a random Hermitian matrix: eigenvalues real & sorted, eigenvectors orthonormal."""
    rng = np.random.default_rng(1)
    B = rng.standard_normal((6, 6)) + 1j * rng.standard_normal((6, 6))
    H = B + B.conj().T                              # Hermitian
    vals, vecs = np.linalg.eigh(H)
    assert np.all(np.isreal(vals)), "Eigenvalues of Hermitian matrix must be real"
    assert np.all(np.diff(vals) >= -1e-12), "Eigenvalues not sorted ascending"
    assert np.allclose(vecs.conj().T @ vecs, np.eye(6), atol=1e-12), \
        "Eigenvectors not orthonormal"
    # Reconstruction: H = V diag(vals) V†
    H_rec = vecs @ np.diag(vals) @ vecs.conj().T
    assert np.allclose(H, H_rec, atol=1e-12), "eigh reconstruction failed"


def test_numpy_einsum():
    """Tensor contraction: einsum vs direct matmul reference."""
    rng = np.random.default_rng(2)
    A = rng.standard_normal((4, 5))
    B = rng.standard_normal((5, 6))
    ref = A @ B
    got = np.einsum("ij,jk->ik", A, B)
    assert np.allclose(ref, got, atol=1e-12), "einsum matmul mismatch"
    # 3-tensor trace-like contraction
    T = rng.standard_normal((3, 4, 4))
    traces = np.einsum("nii->n", T)
    for n in range(3):
        assert np.isclose(traces[n], np.trace(T[n])), f"Trace mismatch at n={n}"


# ════════════════════════════════════════════════════════════════════════════
# 2. SciPy — LAPACK SVD, sparse eigsh
# ════════════════════════════════════════════════════════════════════════════

def test_scipy_svd_vs_numpy():
    """scipy LAPACK SVD must match numpy SVD to machine precision."""
    from scipy.linalg import svd as scipy_svd
    rng = np.random.default_rng(3)
    A = rng.standard_normal((10, 7))
    U_np, s_np, Vh_np = np.linalg.svd(A, full_matrices=False)
    U_sp, s_sp, Vh_sp = scipy_svd(A, full_matrices=False)
    assert np.allclose(s_np, s_sp, atol=1e-12), "Singular values differ between numpy and scipy"
    # Left singular vectors may differ by column sign — compare |U^T U_sp| ≈ I
    assert np.allclose(np.abs(U_np.T @ U_sp), np.eye(7), atol=1e-10), \
        "Left singular vector subspaces differ"


def test_scipy_truncated_svd():
    """Truncated SVD (k=3 of a rank-10 matrix) via scipy sparse eigsh.
    Mimics the CTMRG projector step: keep the k largest singular values."""
    from scipy.sparse.linalg import svds
    rng = np.random.default_rng(4)
    M = 12
    # Build a low-rank matrix: rank-3 signal + small noise
    signal = rng.standard_normal((M, 3)) @ rng.standard_normal((3, M))
    noise  = 1e-3 * rng.standard_normal((M, M))
    A = signal + noise
    U, s, Vh = svds(A, k=3)
    # svds returns ascending order; sort descending
    idx = np.argsort(s)[::-1]
    s = s[idx]; U = U[:, idx]; Vh = Vh[idx, :]
    # The top-3 singular values must capture almost all the Frobenius norm
    captured = np.sum(s**2) / np.sum(np.linalg.svd(A, compute_uv=False)**2)
    assert captured > 0.999, f"Truncated SVD captured only {captured:.4f} of variance"


# ════════════════════════════════════════════════════════════════════════════
# 3. opt_einsum — contraction path & correctness on a tensor-network fragment
# ════════════════════════════════════════════════════════════════════════════

def test_opt_einsum_path_and_result():
    """Contract a rank-4 PEPS-like tensor network fragment.

    Network:  C[ab] T[bcd] T[cef] C[df]  →  scalar
    This is a simplified version of the top row contraction in a square-
    lattice CTMRG environment, used as a proxy for the honeycomb case.
    """
    import opt_einsum as oe
    rng = np.random.default_rng(5)
    chi = 8                          # bond dimension
    C  = rng.standard_normal((chi, chi))
    T  = rng.standard_normal((chi, chi, chi))

    expr = oe.contract_expression(
        "ab,bcd,cef,df->ae",
        C.shape, T.shape, T.shape, C.shape,
        optimize="optimal"
    )
    result_oe  = expr(C, T, T, C)
    result_ref = np.einsum("ab,bcd,cef,df->ae", C, T, T, C)
    assert np.allclose(result_oe, result_ref, atol=1e-10), \
        "opt_einsum result differs from numpy einsum reference"


def test_opt_einsum_flop_reduction():
    """Verify that opt_einsum finds a cheaper path than left-to-right."""
    import opt_einsum as oe
    chi = 16
    rng = np.random.default_rng(6)
    A = rng.standard_normal((chi, chi, chi))
    B = rng.standard_normal((chi, chi, chi))
    C = rng.standard_normal((chi, chi, chi))

    path_naive,  info_naive  = oe.contract_path("ijk,jlm,kln->imn", A, B, C,
                                                  optimize=[(0, 1), (0, 1)])
    path_opt,    info_opt    = oe.contract_path("ijk,jlm,kln->imn", A, B, C,
                                                  optimize="optimal")
    # opt_cost may be a decimal.Decimal — cast to float before comparison
    cost_naive = float(info_naive.opt_cost)
    cost_opt   = float(info_opt.opt_cost)
    assert cost_opt <= cost_naive * 1.01, \
        f"opt_einsum did not reduce FLOPs: {cost_opt:.0f} vs {cost_naive:.0f}"


# ════════════════════════════════════════════════════════════════════════════
# 4. PyTorch — dtype coverage (real + complex) & GPU
# ════════════════════════════════════════════════════════════════════════════

def test_torch_dtype_coverage():
    """Ensure the four dtypes used in production are supported."""
    for dtype in [torch.float32, torch.float64, torch.complex64, torch.complex128]:
        x = torch.ones(4, 4, dtype=dtype)
        y = x @ x
        assert y.dtype == dtype, f"dtype {dtype} not preserved through matmul"


def test_torch_gpu():
    """If CUDA is available, verify tensor survives a CPU→GPU→CPU round-trip."""
    if not torch.cuda.is_available():
        info("No CUDA device — GPU round-trip test skipped.")
        return
    x_cpu = torch.randn(16, 16, dtype=torch.float64)
    x_gpu = x_cpu.cuda()
    assert x_gpu.device.type == "cuda", "Tensor not on GPU after .cuda()"
    x_back = x_gpu.cpu()
    assert torch.allclose(x_cpu, x_back), "GPU round-trip changed values"


# ════════════════════════════════════════════════════════════════════════════
# 5. PyTorch autograd — gradient through einsum
# ════════════════════════════════════════════════════════════════════════════

def test_autograd_einsum():
    """Gradient of a multi-tensor contraction via autograd vs finite differences."""
    torch.manual_seed(0)
    A = torch.randn(4, 5, dtype=torch.float64, requires_grad=True)
    B = torch.randn(5, 6, dtype=torch.float64, requires_grad=True)

    def f(A, B):
        return torch.einsum("ij,jk->ik", A, B).pow(2).sum()

    # Autograd gradient
    loss = f(A, B)
    loss.backward()
    grad_A_auto = A.grad.clone()

    # Finite-difference gradient for A
    eps = 1e-5
    grad_A_fd = torch.zeros_like(A)
    with torch.no_grad():
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                A_plus  = A.detach().clone(); A_plus[i, j]  += eps
                A_minus = A.detach().clone(); A_minus[i, j] -= eps
                grad_A_fd[i, j] = (f(A_plus, B.detach()) -
                                   f(A_minus, B.detach())) / (2 * eps)

    assert torch.allclose(grad_A_auto, grad_A_fd, atol=1e-7, rtol=1e-5), \
        f"Autograd vs FD max error: {(grad_A_auto - grad_A_fd).abs().max():.2e}"


# ════════════════════════════════════════════════════════════════════════════
# 6. PyTorch autograd through SVD  ← MOST CRITICAL FOR AD-CTMRG
# ════════════════════════════════════════════════════════════════════════════

def test_autograd_svd():
    """Differentiate through full (non-truncated) SVD.

    In AD-CTMRG, the loss (e.g. ground-state energy) is differentiated
    back through the SVD that builds the CTMRG projectors. PyTorch's
    torch.linalg.svd has a registered VJP; this test verifies it matches
    finite differences on a small matrix.
    """
    torch.manual_seed(1)
    M = torch.randn(6, 5, dtype=torch.float64, requires_grad=True)

    def f(M):
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        # Scalar loss: sum of log of singular values (appears in free-energy calcs)
        return S.log().sum()

    # Autograd
    loss = f(M)
    loss.backward()
    grad_auto = M.grad.clone()

    # Finite differences
    eps = 1e-5
    grad_fd = torch.zeros_like(M)
    with torch.no_grad():
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                M_p = M.detach().clone(); M_p[i, j] += eps
                M_m = M.detach().clone(); M_m[i, j] -= eps
                grad_fd[i, j] = (f(M_p) - f(M_m)) / (2 * eps)

    assert torch.allclose(grad_auto, grad_fd, atol=1e-7, rtol=1e-5), \
        f"SVD autograd vs FD max error: {(grad_auto - grad_fd).abs().max():.2e}"


def test_autograd_truncated_svd():
    """Differentiate through TRUNCATED SVD — the actual CTMRG projector step.

    We keep only the top k singular values/vectors (bond-dimension truncation)
    and verify gradients flow correctly back to the input matrix.
    """
    torch.manual_seed(2)
    k = 3                             # keep top-k (bond dimension χ)
    M = torch.randn(8, 7, dtype=torch.float64, requires_grad=True)

    def f_trunc(M):
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        U_k  = U[:, :k]
        S_k  = S[:k]
        Vh_k = Vh[:k, :]
        # Reconstruct truncated matrix, compute Frobenius norm of residual
        M_trunc = U_k @ torch.diag(S_k) @ Vh_k
        return (M - M_trunc).pow(2).sum()   # truncation error — key CTMRG quantity

    loss = f_trunc(M)
    loss.backward()
    grad_auto = M.grad.clone()

    eps = 1e-5
    grad_fd = torch.zeros_like(M)
    with torch.no_grad():
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                M_p = M.detach().clone(); M_p[i, j] += eps
                M_m = M.detach().clone(); M_m[i, j] -= eps
                grad_fd[i, j] = (f_trunc(M_p) - f_trunc(M_m)) / (2 * eps)

    assert torch.allclose(grad_auto, grad_fd, atol=1e-6, rtol=1e-4), \
        f"Truncated SVD autograd vs FD max error: {(grad_auto - grad_fd).abs().max():.2e}"


# ════════════════════════════════════════════════════════════════════════════
# 7. PyTorch autograd through eigh (corner diagonalisation)
# ════════════════════════════════════════════════════════════════════════════

def test_autograd_eigh():
    """Differentiate through Hermitian eigendecomposition.

    The CTMRG corner matrix is diagonalised to compute the environment
    spectrum; gradients must flow through this step.

    Parametrisation: H = L + L^T  where L is a free (non-symmetric) matrix.
    Differentiating through L avoids the ambiguity of perturbing only one
    element of a symmetric matrix in the finite-difference check.
    """
    torch.manual_seed(3)
    n = 6
    L = torch.randn(n, n, dtype=torch.float64, requires_grad=True)

    def f(L):
        H = L + L.T                    # explicitly symmetric
        vals, _ = torch.linalg.eigh(H)
        # Loss: entropy-like quantity on the eigenvalue spectrum
        lam = vals.abs() + 1e-8
        lam = lam / lam.sum()
        return -(lam * lam.log()).sum()

    loss = f(L)
    loss.backward()
    grad_auto = L.grad.clone()

    # Finite differences w.r.t. each element of L (L is unconstrained)
    eps = 1e-5
    grad_fd = torch.zeros_like(L)
    with torch.no_grad():
        for i in range(n):
            for j in range(n):
                L_p = L.detach().clone(); L_p[i, j] += eps
                L_m = L.detach().clone(); L_m[i, j] -= eps
                grad_fd[i, j] = (f(L_p) - f(L_m)) / (2 * eps)

    assert torch.allclose(grad_auto, grad_fd, atol=1e-6, rtol=1e-4), \
        f"eigh autograd vs FD max error: {(grad_auto - grad_fd).abs().max():.2e}"


# ════════════════════════════════════════════════════════════════════════════
# 8. h5py — checkpoint round-trip
# ════════════════════════════════════════════════════════════════════════════

def test_h5py_checkpoint_roundtrip():
    """Save a dict of named tensors (CTMRG environment) to HDF5 and reload.

    Mimics the checkpoint pattern used during long CTMRG sweeps:
      {'C_NW': corner, 'T_N': edge, 'A': site_tensor}
    """
    import h5py
    torch.manual_seed(4)
    chi, d = 8, 2
    env = {
        "C_NW": torch.randn(chi, chi, dtype=torch.float64),
        "T_N" : torch.randn(chi, d, chi, dtype=torch.float64),
        "A"   : torch.randn(d, d, d, d, d, d, dtype=torch.float64),  # honeycomb site
    }

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        path = tmp.name

    try:
        # Save
        with h5py.File(path, "w") as f:
            for name, tensor in env.items():
                f.create_dataset(name, data=tensor.numpy())

        # Reload
        loaded = {}
        with h5py.File(path, "r") as f:
            for name in env:
                loaded[name] = torch.from_numpy(np.array(f[name]))

        # Verify
        for name in env:
            assert torch.allclose(env[name], loaded[name]), \
                f"Checkpoint mismatch for tensor '{name}'"
    finally:
        os.remove(path)


# ════════════════════════════════════════════════════════════════════════════
# __main__ runner
# ════════════════════════════════════════════════════════════════════════════

TESTS = [
    ("numpy — SVD reconstruction & orthogonality",          test_numpy_svd),
    ("numpy — eigh correctness & reconstruction",           test_numpy_eigh),
    ("numpy — einsum matmul & trace",                       test_numpy_einsum),
    ("scipy — LAPACK SVD matches numpy",                    test_scipy_svd_vs_numpy),
    ("scipy — truncated SVD (CTMRG projector proxy)",       test_scipy_truncated_svd),
    ("opt_einsum — TN fragment correctness & path",         test_opt_einsum_path_and_result),
    ("opt_einsum — FLOP reduction vs naive ordering",       test_opt_einsum_flop_reduction),
    ("torch — dtype coverage (float32/64, complex64/128)",  test_torch_dtype_coverage),
    ("torch — GPU round-trip",                              test_torch_gpu),
    ("torch autograd — einsum gradient vs FD",             test_autograd_einsum),
    ("torch autograd — full SVD gradient vs FD",           test_autograd_svd),
    ("torch autograd — TRUNCATED SVD gradient vs FD ★",    test_autograd_truncated_svd),
    ("torch autograd — eigh gradient vs FD",               test_autograd_eigh),
    ("h5py — environment checkpoint round-trip",            test_h5py_checkpoint_roundtrip),
]

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nPython     : {sys.version[:6]}")
    print(f"torch      : {torch.__version__}")
    print(f"device     : {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    passed = failed = 0
    for label, test in TESTS:
        head(label)
        try:
            test()
            ok(label)
            passed += 1
        except Exception as e:
            fail(f"{label}  →  {e}")
            failed += 1

    print(f"\n{'═'*58}")
    print(f"  {passed} passed   {failed} failed   out of {len(TESTS)} checks")
    print(f"{'═'*58}\n")
    sys.exit(0 if failed == 0 else 1)
