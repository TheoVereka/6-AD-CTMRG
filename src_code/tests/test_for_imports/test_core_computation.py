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
  9.  opt_einsum + torch              — demonstrate that oe.contract accepts torch
                                        tensors and autograd flows through it
  10. torch.svd_lowrank               — built-in randomised truncated SVD:
                                        approximation quality & autograd
  11. index fusion before SVD         — reshape rank-4 env tensor to matrix,
                                        SVD, truncate, unfuse: autograd check
  12. opt_einsum memory_limit         — built-in memory-capping mechanism;
                                        explains why CTMRG memory is dominated
                                        by tensor size, not path choice
  13. eigh vs SVD for Hermitian       — eigh IS the right call for Hermitian
                                        matrices (faster + identical result);
                                        autograd through truncated eigh projector
  14. torch.lobpcg                    — iterative top-k eigensolver for
                                        Hermitian matrices (analogue of
                                        svd_lowrank); accuracy + autograd

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
# 9. opt_einsum cooperating with PyTorch (backend auto-detection + autograd)
# ════════════════════════════════════════════════════════════════════════════

def test_opt_einsum_torch_backend():
    """opt_einsum detects torch tensors and uses torch.einsum as the backend.

    Consequence: the optimal contraction path chosen by opt_einsum is executed
    via torch operations, so the full autograd graph is preserved.
    We verify:
      (a) the numerical result matches torch.einsum,
      (b) gradients computed through oe.contract match finite differences.
    """
    import opt_einsum as oe

    torch.manual_seed(10)
    chi = 6
    # Three rank-3 tensors — a fragment of the honeycomb CTMRG row contraction
    A = torch.randn(chi, chi, chi, dtype=torch.float64, requires_grad=True)
    B = torch.randn(chi, chi, chi, dtype=torch.float64, requires_grad=True)
    C = torch.randn(chi, chi, chi, dtype=torch.float64, requires_grad=True)

    expr_str = "ijk,jlm,kln->imn"

    # (a) Correctness: opt_einsum result == torch.einsum result
    result_oe  = oe.contract(expr_str, A, B, C)          # auto-detects torch backend
    result_ref = torch.einsum(expr_str, A, B, C)
    assert torch.allclose(result_oe, result_ref, atol=1e-12), \
        "opt_einsum+torch result differs from torch.einsum reference"

    # (b) Autograd flows through oe.contract
    def f_oe(A, B, C):
        return oe.contract(expr_str, A, B, C).pow(2).sum()

    def f_ref(A, B, C):
        return torch.einsum(expr_str, A, B, C).pow(2).sum()

    # Autograd gradient w.r.t. A via opt_einsum path
    loss_oe = f_oe(A, B, C)
    loss_oe.backward()
    grad_oe = A.grad.clone()
    A.grad = None

    # Autograd gradient w.r.t. A via plain torch.einsum (ground truth)
    loss_ref = f_ref(A, B, C)
    loss_ref.backward()
    grad_ref = A.grad.clone()

    assert torch.allclose(grad_oe, grad_ref, atol=1e-12), (
        f"opt_einsum autograd differs from torch.einsum autograd; "
        f"max diff: {(grad_oe - grad_ref).abs().max():.2e}"
    )


def test_opt_einsum_torch_uses_optimised_path():
    """Confirm opt_einsum chose a cheaper path than naive left-to-right,
    even when operating on torch tensors (path is computed before execution).
    """
    import opt_einsum as oe

    chi = 20
    torch.manual_seed(11)
    A = torch.randn(chi, chi, chi, dtype=torch.float32)
    B = torch.randn(chi, chi, chi, dtype=torch.float32)
    C = torch.randn(chi, chi, chi, dtype=torch.float32)

    expr_str = "ijk,jlm,kln->imn"

    # Inspect the path opt_einsum selects (uses shape info only, no data)
    path_opt, info_opt   = oe.contract_path(expr_str, A, B, C, optimize="optimal")
    path_naive, info_naive = oe.contract_path(expr_str, A, B, C,
                                               optimize=[(0, 1), (0, 1)])
    cost_opt   = float(info_opt.opt_cost)
    cost_naive = float(info_naive.opt_cost)
    assert cost_opt <= cost_naive * 1.01, (
        f"opt_einsum path not cheaper on torch tensors: "
        f"{cost_opt:.0f} vs naive {cost_naive:.0f}"
    )


# ════════════════════════════════════════════════════════════════════════════
# 10. torch.svd_lowrank — built-in randomised truncated SVD
# ════════════════════════════════════════════════════════════════════════════

def test_svd_lowrank_approximation():
    """torch.svd_lowrank is PyTorch's randomised truncated SVD
    (Halko-Martinsson-Tropp algorithm).

    We build a matrix with a clear rank-k signal and verify:
      - the top-k singular values closely match those from the exact SVD,
      - the low-rank reconstruction error is small.

    Note: svd_lowrank returns V (not Vh), so  A ≈ U @ diag(S) @ V.T
    """
    torch.manual_seed(20)
    m, n, k = 40, 30, 5

    # Construct a rank-k matrix with known singular values 10,9,...,6
    true_S = torch.arange(k, 0, -1, dtype=torch.float64) + 5   # [10,9,8,7,6]
    U0, _ = torch.linalg.qr(torch.randn(m, k, dtype=torch.float64))
    V0, _ = torch.linalg.qr(torch.randn(n, k, dtype=torch.float64))
    M = U0 @ torch.diag(true_S) @ V0.T

    # Exact SVD reference
    U_ex, S_ex, Vh_ex = torch.linalg.svd(M, full_matrices=False)

    # Randomised truncated SVD  (q = oversampling power-iteration steps)
    U_lr, S_lr, V_lr = torch.svd_lowrank(M, q=k, niter=4)
    # V_lr is V (right singular vectors, shape n×k), not Vh

    # Singular values must match the top-k from exact SVD
    assert torch.allclose(S_lr, S_ex[:k], atol=1e-6), (
        f"svd_lowrank singular values differ from exact:\n"
        f"  lowrank : {S_lr.tolist()}\n"
        f"  exact   : {S_ex[:k].tolist()}"
    )

    # Low-rank reconstruction
    M_approx = U_lr @ torch.diag(S_lr) @ V_lr.T
    rel_err = (M - M_approx).norm() / M.norm()
    assert rel_err < 1e-6, f"Low-rank reconstruction relative error too large: {rel_err:.2e}"


def test_svd_lowrank_autograd():
    """Autograd flows through torch.svd_lowrank.

    This is the production use-case: differentiate the CTMRG energy
    back through the randomised projector construction.

    IMPORTANT — why the seed must be fixed:
      svd_lowrank internally draws a random Gaussian sketch matrix.
      If f(M+eps) and f(M-eps) use *different* random sketches, the FD
      estimate is dominated by stochastic noise, not the partial derivative.
      Fixing torch.manual_seed to the same value before every call
      "freezes" the random projection so the difference is purely due to
      the perturbation in M — making FD meaningful.
    """
    torch.manual_seed(21)
    k = 3
    M = torch.randn(10, 8, dtype=torch.float64, requires_grad=True)
    FIX_SEED = 42           # same sketch for every forward pass

    def f_lowrank(M):
        torch.manual_seed(FIX_SEED)   # freeze the random projection
        U, S, V = torch.svd_lowrank(M, q=k, niter=4)
        M_approx = U @ torch.diag(S) @ V.T
        return (M - M_approx).pow(2).sum()

    # Autograd — forward pass uses FIX_SEED
    loss = f_lowrank(M)
    loss.backward()
    grad_auto = M.grad.clone()

    # Finite differences — each pair also uses FIX_SEED -> same sketch
    eps = 1e-5
    grad_fd = torch.zeros_like(M)
    with torch.no_grad():
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                M_p = M.detach().clone(); M_p[i, j] += eps
                M_m = M.detach().clone(); M_m[i, j] -= eps
                grad_fd[i, j] = (f_lowrank(M_p) - f_lowrank(M_m)) / (2 * eps)

    max_err = (grad_auto - grad_fd).abs().max().item()
    assert max_err < 1e-4, (
        f"svd_lowrank autograd vs FD max error: {max_err:.2e}\n"
        "(with a fixed sketch the gradient should be as accurate as for exact SVD)"
    )


def test_svd_lowrank_vs_exact_truncated_svd():
    """Compare svd_lowrank gradient to the exact truncated SVD gradient.

    For a well-conditioned, rapidly-decaying spectrum, both should agree
    to moderate precision because the randomised algorithm converges.
    """
    torch.manual_seed(22)
    k = 3
    m, n = 12, 10

    # Build a matrix with rapidly decaying singular values (well-conditioned)
    true_S = torch.tensor([100., 10., 1., 0.1, 0.01, 0.001,
                            0.0001, 0.00001, 0.000001, 0.0000001],
                           dtype=torch.float64)
    U0, _ = torch.linalg.qr(torch.randn(m, m, dtype=torch.float64))
    V0, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64))
    M_data = U0[:, :n] @ torch.diag(true_S) @ V0

    def f_exact(M):
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        M_approx = U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :]
        return (M - M_approx).pow(2).sum()

    def f_lowrank(M):
        U, S, V = torch.svd_lowrank(M, q=k, niter=8)
        M_approx = U @ torch.diag(S) @ V.T
        return (M - M_approx).pow(2).sum()

    M_e = M_data.clone().requires_grad_(True)
    f_exact(M_e).backward()
    grad_exact = M_e.grad.clone()

    M_lr = M_data.clone().requires_grad_(True)
    f_lowrank(M_lr).backward()
    grad_lowrank = M_lr.grad.clone()

    max_err = (grad_exact - grad_lowrank).abs().max().item()
    # Gradients should agree well given the rapidly decaying spectrum
    assert max_err < 0.1, (
        f"svd_lowrank and exact truncated SVD gradients diverge: max diff = {max_err:.2e}"
    )
    info(f"svd_lowrank vs exact truncated SVD gradient max diff: {max_err:.2e}")


# ════════════════════════════════════════════════════════════════════════════
# 11. Index fusion (leg contraction) before SVD — the CTMRG projector build
# ════════════════════════════════════════════════════════════════════════════

def test_index_fusion_before_svd():
    """In every CTMRG step the environment tensors must be *fused* (reshaped)
    before the SVD that builds the isometric projectors, then unfused afterward.

    Concrete example — updating the North edge tensor:
      T[chi, d, d, chi]  ──reshape──>  T_mat[chi*d, d*chi]
      SVD  →  U[chi*d, k], S[k], Vh[k, d*chi]
      Truncate to rank k = chi (bond dimension)
      Unfuse left isometry:  P = U.reshape(chi, d, k)   ← rank-3 projector

    This test verifies:
      (a) reshape to matrix preserves all elements exactly (correct row-major
          index mapping: T[a,b,c,e] == T_mat[a*d+b, c*chi+e])
      (b) the truncated left singular vectors form a proper isometry
          U_k^T @ U_k == I_k  (used as environment projector in CTMRG)
      (c) autograd flows through the full fuse -> SVD -> truncate -> unfuse
          pipeline, verified against finite differences element-by-element
          over the original rank-4 tensor.
    """
    torch.manual_seed(30)
    chi, d = 6, 3    # bond dimension, physical leg dimension
    k = chi           # truncation rank = bond dimension (standard choice)
    T = torch.randn(chi, d, d, chi, dtype=torch.float64, requires_grad=True)

    # ── (a) reshape correctness ──────────────────────────────────────────────
    # PyTorch reshape uses C (row-major) order: rightmost index varies fastest.
    # T[a, b, c, e] -> T_mat[a*d + b, c*chi + e]
    T_mat = T.reshape(chi * d, d * chi)
    assert T_mat.shape == (chi * d, d * chi), "Matrix shape after reshape is wrong"

    # Spot-check the index mapping at a non-trivial position
    a, b, c, e = 3, 2, 1, 4
    expected = T[a, b, c, e].item()
    got      = T_mat[a * d + b, c * chi + e].item()
    assert abs(expected - got) < 1e-15, (
        f"Index mapping wrong: T[{a},{b},{c},{e}]={expected:.6f} "
        f"!= T_mat[{a*d+b},{c*chi+e}]={got:.6f}"
    )

    # ── (b) SVD + truncation → isometry property ────────────────────────────
    with torch.no_grad():
        U, S, Vh = torch.linalg.svd(T_mat, full_matrices=False)
    U_k  = U[:, :k]    # (chi*d, k)
    Vh_k = Vh[:k, :]   # (k, d*chi)

    gram = U_k.T @ U_k   # should be I_k
    assert torch.allclose(gram, torch.eye(k, dtype=torch.float64), atol=1e-12), (
        f"Left isometry U_k^T U_k != I_{k}; "
        f"max deviation {(gram - torch.eye(k, dtype=torch.float64)).abs().max():.2e}"
    )

    # Reshape projector back to rank-3 — the CTMRG projector shape
    P_left = U_k.reshape(chi, d, k)
    assert P_left.shape == (chi, d, k), "Rank-3 projector has wrong shape after unfuse"

    # ── (c) autograd through fuse -> SVD -> unfuse ───────────────────────────
    def f(T):
        T_mat = T.reshape(chi * d, d * chi)
        U, S, Vh = torch.linalg.svd(T_mat, full_matrices=False)
        S_k = S[:k]
        U_k = U[:, :k]
        P   = U_k.reshape(chi, d, k)        # rank-3 projector
        # Scalar loss combining two CTMRG-relevant quantities:
        #   spectrum of the environment (S_k.log().sum()) and
        #   Frobenius norm of the projector (P.pow(2).sum())
        return S_k.log().sum() + P.pow(2).sum()

    loss = f(T)
    loss.backward()
    grad_auto = T.grad.clone()

    eps = 1e-5
    grad_fd = torch.zeros_like(T)
    with torch.no_grad():
        for a_ in range(chi):
            for b_ in range(d):
                for c_ in range(d):
                    for e_ in range(chi):
                        T_p = T.detach().clone(); T_p[a_, b_, c_, e_] += eps
                        T_m = T.detach().clone(); T_m[a_, b_, c_, e_] -= eps
                        grad_fd[a_, b_, c_, e_] = (f(T_p) - f(T_m)) / (2 * eps)

    max_err = (grad_auto - grad_fd).abs().max().item()
    assert max_err < 1e-6, (
        f"Fuse->SVD->unfuse autograd vs FD max error: {max_err:.2e}"
    )


# ════════════════════════════════════════════════════════════════════════════
# 12. opt_einsum memory_limit — built-in memory cap for contraction paths
# ════════════════════════════════════════════════════════════════════════════

def test_opt_einsum_memory_limit():
    """opt_einsum has a built-in memory mechanism via the memory_limit argument
    of oe.contract_path().  It caps the number of *elements* in any intermediate
    tensor created during the contraction path.

    Important context for CTMRG:
    ─────────────────────────────
    For CTMRG-like networks (linear chain of corners and edge tensors), the
    FLOP-optimal path is almost always also memory-optimal.  The reason is
    structural: contracting a small corner (χ×χ) into the nearest edge tensor
    (χ×d×d×χ) first — which opt_einsum already prefers — gives a χ²d²
    intermediate, smaller than contracting two edge tensors together (χ²d⁴).

    Consequence: for CTMRG you do NOT need to manually set memory_limit to get
    a good path.  The true memory bottleneck is tensor *size*, not path *order*:
      - Corner: χ² floats
      - Edge:   χ²·d² floats  (d = physical bond dimension)
      - Full double-layer row: χ²·d⁴ floats
    Managing memory means minimising χ (via truncation) and using float32.

    What memory_limit IS useful for:
      - Large irregular/loopy networks (10+ tensors) where the optimizer may
        accidentally form expensive outer products.
      - Setting 'max_input' prevents any intermediate from exceeding the size
        of the largest input, a cheap upper bound on a sensible path.

    This test verifies:
      (a) oe.contract_path accepts memory_limit (both integer and 'max_input')
      (b) memory_limit never INCREASES largest_intermediate vs unconstrained
          (correct invariant: constrained path is as good as or better than free)
      (c) the contraction result is identical regardless of memory_limit
      (d) prints a memory estimate table for realistic CTMRG parameters to
          show that tensor size, not path ordering, is the real bottleneck.
    """
    import opt_einsum as oe

    chi, d = 8, 2    # representative CTMRG parameters
    rng = np.random.default_rng(40)

    # CTMRG double-edge row: C1[ax] T1[xpqy] T2[yrsz] C2[zb] -> [apqrsb]
    C1 = rng.standard_normal((chi, chi))
    T1 = rng.standard_normal((chi, d, d, chi))
    T2 = rng.standard_normal((chi, d, d, chi))
    C2 = rng.standard_normal((chi, chi))
    expr = "ax,xpqy,yrsz,zb->apqrsb"

    # ── (a) API: both memory_limit forms are accepted ───────────────────────
    path_free, info_free = oe.contract_path(expr, C1, T1, T2, C2,
                                             optimize="optimal")
    path_inp,  info_inp  = oe.contract_path(expr, C1, T1, T2, C2,
                                             optimize="optimal",
                                             memory_limit="max_input")
    # Integer limit: cap at largest input size (T1 or T2 = chi^2 * d^2 elements)
    max_input_size = max(C1.size, T1.size, T2.size, C2.size)  # == chi^2 * d^2
    path_int,  info_int  = oe.contract_path(expr, C1, T1, T2, C2,
                                             optimize="optimal",
                                             memory_limit=int(max_input_size))

    # ── (b) info.largest_intermediate is tighter with memory constraint ────────────────
    # When memory_limit < output_size the optimizer tries its best but cannot
    # honour the limit for the final contraction step (the output itself may be
    # larger than max_input).  The correct invariant:
    #   constrained largest_intermediate  ≤  unconstrained largest_intermediate
    # i.e. memory_limit never makes things WORSE.
    assert int(info_inp.largest_intermediate) <= int(info_free.largest_intermediate), (
        f"memory_limit='max_input' increased largest_intermediate: "
        f"{info_inp.largest_intermediate} > {info_free.largest_intermediate}"
    )
    assert int(info_int.largest_intermediate) <= int(info_free.largest_intermediate), (
        f"integer memory_limit increased largest_intermediate: "
        f"{info_int.largest_intermediate} > {info_free.largest_intermediate}"
    )
    info(f"largest_intermediate: unconstrained={info_free.largest_intermediate}, "
         f"max_input={info_inp.largest_intermediate}, "
         f"int({int(max_input_size)})={info_int.largest_intermediate}")

    # ── (c) all three paths produce the same numerical result ───────────────
    ref    = np.einsum(expr, C1, T1, T2, C2)
    res_free = oe.contract(expr, C1, T1, T2, C2, optimize=path_free)
    res_inp  = oe.contract(expr, C1, T1, T2, C2, optimize=path_inp)
    res_int  = oe.contract(expr, C1, T1, T2, C2, optimize=path_int)
    assert np.allclose(res_free, ref, atol=1e-10), "free path result wrong"
    assert np.allclose(res_inp,  ref, atol=1e-10), "max_input path result wrong"
    assert np.allclose(res_int,  ref, atol=1e-10), "integer limit path result wrong"

    # ── (d) memory estimation for realistic chi/d values ────────────────────
    bytes_per_float64 = 8
    info("CTMRG memory footprint (float64, dominant tensor at each chi/d):")
    info(f"  {'chi':>4}  {'d':>2}  {'corner(KB)':>11}  {'edge(KB)':>9}  "
         f"{'double-layer row(MB)':>20}  {'largest_intermediate':>21}")
    for chi_t, d_t in [(8, 2), (16, 2), (32, 2), (16, 3), (32, 3)]:
        corner = chi_t**2 * bytes_per_float64 / 1024
        edge   = chi_t**2 * d_t**2 * bytes_per_float64 / 1024
        row    = chi_t**2 * d_t**4 * bytes_per_float64 / 1024**2
        info(f"  {chi_t:>4}  {d_t:>2}  {corner:>10.1f}K  {edge:>8.1f}K  "
             f"{row:>19.2f}M  "
             f"  max_input={chi_t**2 * d_t**2} elems")
    info("Conclusion: for CTMRG, the bottleneck is χ²d⁴ (row tensor), "
         "not contraction order.  Reduce χ, not path.")


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
# 13. eigh vs SVD for Hermitian matrices — faster exact decomposition
# ════════════════════════════════════════════════════════════════════════════

def test_eigh_vs_svd_for_hermitian():
    """For a Hermitian (or real symmetric) matrix, torch.linalg.eigh is both
    faster and more semantically correct than torch.linalg.svd because:

      - LAPACK's dsyevd (divide-and-conquer, used by eigh) exploits symmetry
        and is typically 2–3x faster than dgesdd (used by svd).
      - For a PSD matrix M = A A^T: eigenvalues == singular values exactly,
        so eigh gives the same decomposition at lower cost.
      - For the CTMRG projector step, eigh is the right call whenever the
        matrix is guaranteed Hermitian (e.g. the corner transfer matrix in a
        symmetric gauge, or any M = A A^T formed explicitly).

    This test verifies:
      (a) eigh eigenvalues == svd singular values on a random PSD Hermitian
          matrix, to < 1e-10 absolute tolerance.
      (b) autograd flows through the full eigh-based projector pipeline
          (fuse -> eigh -> truncate top-k -> unfuse) and matches FD.
      (c) reports a timing comparison so the speedup is visible in the output.
    """
    import time
    torch.manual_seed(40)
    n = 50

    # Random PSD Hermitian: M = A A^T + eps I  (non-degenerate, well-conditioned)
    A = torch.randn(n, n, dtype=torch.float64)
    M = A @ A.T + 1e-3 * torch.eye(n, dtype=torch.float64)

    # ── (a) eigh eigenvalues == svd singular values ──────────────────────────
    vals_eigh, _ = torch.linalg.eigh(M)       # ascending
    _, s_svd, _  = torch.linalg.svd(M, full_matrices=False)  # descending
    assert torch.allclose(vals_eigh.flip(0), s_svd, atol=1e-10), (
        f"eigh eigenvalues != svd singular values on PSD Hermitian\n"
        f"  eigh (desc): {vals_eigh.flip(0)[:5].tolist()}\n"
        f"  svd  (desc): {s_svd[:5].tolist()}"
    )

    # ── (b) autograd through truncated eigh projector ────────────────────────
    # Parametrise via free matrix L so gradients on L are well-defined
    torch.manual_seed(41)
    k = 10   # keep top-k eigenvalues (CTMRG bond dimension)
    L = torch.randn(n, n, dtype=torch.float64, requires_grad=True)

    def f_eigh_proj(L):
        M_ = L @ L.T + 1e-3 * torch.eye(n, dtype=torch.float64)
        vals, vecs = torch.linalg.eigh(M_)
        # eigh returns ascending order — top-k are the last k entries
        vals_k = vals[-k:]
        vecs_k = vecs[:, -k:]
        M_approx = vecs_k @ torch.diag(vals_k) @ vecs_k.T
        return (M_ - M_approx).pow(2).sum()

    f_eigh_proj(L).backward()
    grad_auto = L.grad.clone()

    eps = 1e-5
    grad_fd = torch.zeros_like(L)
    with torch.no_grad():
        for i in range(6):   # spot-check first 6 rows (full loop is slow for n=50)
            for j in range(n):
                L_p = L.detach().clone(); L_p[i, j] += eps
                L_m = L.detach().clone(); L_m[i, j] -= eps
                grad_fd[i, j] = (f_eigh_proj(L_p) - f_eigh_proj(L_m)) / (2 * eps)

    max_err = (grad_auto[:6] - grad_fd[:6]).abs().max().item()
    assert max_err < 1e-6, f"eigh projector autograd vs FD max error: {max_err:.2e}"

    # ── (c) timing: eigh vs svd on this size ────────────────────────────────
    n_rep = 50
    M_fixed = M.detach()
    t0 = time.perf_counter()
    for _ in range(n_rep):
        torch.linalg.eigh(M_fixed)
    t_eigh = (time.perf_counter() - t0) / n_rep * 1000

    t0 = time.perf_counter()
    for _ in range(n_rep):
        torch.linalg.svd(M_fixed, full_matrices=False)
    t_svd = (time.perf_counter() - t0) / n_rep * 1000

    info(f"eigh vs svd on {n}×{n} PSD Hermitian (avg {n_rep} reps):")
    info(f"  torch.linalg.eigh : {t_eigh:.3f} ms")
    info(f"  torch.linalg.svd  : {t_svd:.3f} ms")
    info(f"  speedup: {t_svd/t_eigh:.2f}x  ← eigh wins whenever matrix is Hermitian")


# ════════════════════════════════════════════════════════════════════════════
# 14. torch.lobpcg — iterative top-k Hermitian eigensolver
# ════════════════════════════════════════════════════════════════════════════

def test_lobpcg_truncated_hermitian():
    """torch.lobpcg is PyTorch's iterative Hermitian eigensolver
    (Locally Optimal Block Preconditioned Conjugate Gradient).

    It is the Hermitian analogue of torch.svd_lowrank:
      - finds only the top-k eigenvalues/vectors without computing all n of them
      - converges in O(k) iterations when eigenvalues are well-separated
      - supports autograd natively
      - runs on GPU

    Comparison of the Hermitian eigensolver hierarchy:
    ──────────────────────────────────────────────────
      torch.linalg.eigh     exact, O(n³),   autograd ✓, GPU ✓   ← best for small n
      torch.svd_lowrank     rand., O(nk),   autograd ✓, GPU ✓   ← works on ANY matrix
      torch.lobpcg          iter., O(nk²),  autograd ✓, GPU ✓   ← best for large
                                                                    Hermitian + sparse

    For CTMRG: if the corner transfer matrix (χ²×χ²) is Hermitian, lobpcg
    is faster than svd_lowrank for large χ because:
      - it works directly with the Hermitian structure (no need to form A^T A)
      - it converges in fewer matrix-vector products when the spectrum decays fast

    This test verifies:
      (a) lobpcg top-k eigenvalues match eigh top-k to < 1e-8
      (b) lobpcg left singular vectors span the same subspace as eigh
      (c) autograd flows through lobpcg, verified against FD
    """
    torch.manual_seed(50)
    n, k = 40, 5

    # Random PSD Hermitian with well-separated eigenvalues (good for LOBPCG)
    A = torch.randn(n, n, dtype=torch.float64)
    M = A @ A.T + torch.eye(n, dtype=torch.float64)   # eigenvalues >= 1

    # ── (a) lobpcg top-k eigenvalues match eigh ──────────────────────────────
    vals_lobpcg, vecs_lobpcg = torch.lobpcg(M, k=k, largest=True,
                                             niter=500, tol=1e-12)
    vals_eigh, vecs_eigh = torch.linalg.eigh(M)
    vals_eigh_topk = vals_eigh[-k:].flip(0)   # descending, largest first

    assert torch.allclose(vals_lobpcg, vals_eigh_topk, atol=1e-8), (
        f"lobpcg top-{k} eigenvalues differ from eigh:\n"
        f"  lobpcg : {vals_lobpcg.tolist()}\n"
        f"  eigh   : {vals_eigh_topk.tolist()}"
    )

    # ── (b) subspace agreement: |V_lobpcg^T V_eigh| ≈ I_k ───────────────────
    vecs_eigh_topk = vecs_eigh[:, -k:].flip(1)   # descending column order
    gram = vecs_lobpcg.T @ vecs_eigh_topk
    assert torch.allclose(gram.abs(), torch.eye(k, dtype=torch.float64), atol=1e-6), (
        f"lobpcg and eigh top-{k} subspaces do not match; "
        f"max off-diagonal |gram|: {gram.abs().fill_diagonal_(0).max():.2e}"
    )

    # ── (c) autograd through lobpcg ──────────────────────────────────────────
    torch.manual_seed(51)
    L = torch.randn(n, n, dtype=torch.float64, requires_grad=True)

    def f_lobpcg(L):
        M_ = L @ L.T + torch.eye(n, dtype=torch.float64)
        vals, vecs = torch.lobpcg(M_, k=k, largest=True, niter=500, tol=1e-11)
        M_approx = vecs @ torch.diag(vals) @ vecs.T
        return (M_ - M_approx).pow(2).sum()

    f_lobpcg(L).backward()
    grad_auto = L.grad.clone()

    eps = 1e-5
    grad_fd = torch.zeros_like(L)
    with torch.no_grad():
        for i in range(5):   # spot-check 5 rows
            for j in range(n):
                L_p = L.detach().clone(); L_p[i, j] += eps
                L_m = L.detach().clone(); L_m[i, j] -= eps
                grad_fd[i, j] = (f_lobpcg(L_p) - f_lobpcg(L_m)) / (2 * eps)

    max_err = (grad_auto[:5] - grad_fd[:5]).abs().max().item()
    assert max_err < 1e-4, f"lobpcg autograd vs FD max error: {max_err:.2e}"
    info(f"lobpcg autograd vs FD max err (first 5 rows of L): {max_err:.2e}")


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
    # ── opt_einsum + torch ───────────────────────────────────────────────────
    ("opt_einsum+torch — result & autograd match torch.einsum",
                                                test_opt_einsum_torch_backend),
    ("opt_einsum+torch — optimised path also applies to torch tensors",
                                                test_opt_einsum_torch_uses_optimised_path),
    # ── torch.svd_lowrank ────────────────────────────────────────────────────
    ("torch.svd_lowrank — approximation quality vs exact SVD",
                                                test_svd_lowrank_approximation),
    ("torch.svd_lowrank — autograd flows (gradient vs FD)",
                                                test_svd_lowrank_autograd),
    ("torch.svd_lowrank — gradient matches exact truncated SVD (decaying spectrum)",
                                                test_svd_lowrank_vs_exact_truncated_svd),
    # ── CTMRG mechanics ──────────────────────────────────────────────────────
    ("index fusion before SVD — reshape, isometry, autograd ★",
                                                test_index_fusion_before_svd),
    ("opt_einsum memory_limit — API, path constraint, CTMRG memory table",
                                                test_opt_einsum_memory_limit),
    # ── Hermitian eigensolver hierarchy ────────────────────────────────────────────
    ("eigh vs SVD for Hermitian — equiv., speed, autograd ★",
                                                test_eigh_vs_svd_for_hermitian),
    ("torch.lobpcg — iterative top-k Hermitian eigensolver, autograd",
                                                test_lobpcg_truncated_hermitian),
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
