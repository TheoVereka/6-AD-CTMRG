from __future__ import annotations
import os
import sys
import gc

# ==============================================================================
# 1. CLUSTER THREAD PINNING & TORCH INITIALIZATION
# ==============================================================================
try:
    NUM_CORES = len(os.sched_getaffinity(0))
except (AttributeError, NotImplementedError):
    NUM_CORES = os.cpu_count() or 72

os.environ["MKL_NUM_THREADS"] = str(NUM_CORES)
os.environ["OPENBLAS_NUM_THREADS"] = str(NUM_CORES)
os.environ["OMP_NUM_THREADS"] = str(NUM_CORES)
os.environ["MKL_DYNAMIC"] = "FALSE"

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union
import numpy as np
import torch

# Explicitly bind PyTorch CPU threading to match cluster environment allocations
torch.set_num_threads(NUM_CORES)
torch.set_num_interop_threads(1)

DEFAULT_RESULT_DIR = Path(r"/scratch/chye/varPri")


# ==============================================================================
# Mode-set helpers & Vector packing
# ==============================================================================

def _build_modes(K1: int, K2: int):
    full_grid = [(i, j) for i in range(-K1, K1 + 1) for j in range(-K2, K2 + 1)]
    K_plus    = [(i, j) for (i, j) in full_grid if i > 0 or (i == 0 and j > 0)]
    K_plus0   = [(0, 0)] + K_plus
    fg_idx    = {k: idx for idx, k in enumerate(full_grid)}
    return full_grid, K_plus, K_plus0, fg_idx


def _build_h_flat(H: dict, full_grid: list, fg_idx: dict, d: int) -> np.ndarray:
    d2     = d * d
    h_flat = np.zeros(len(full_grid) * d2, dtype=complex)
    for k in full_grid:
        if k in H:
            h_flat[fg_idx[k] * d2 : (fg_idx[k] + 1) * d2] = H[k].ravel()
    return h_flat


# ==============================================================================
# L: Rigorous & Type-Safe Operator Builder
# ==============================================================================

def _build_L_matvec_fast_vectorized(
    H: dict, full_grid: list, fg_idx: dict, d: int, omega: np.ndarray, dtype=np.complex128
) -> np.ndarray:
    d2 = d * d
    N_modes = len(full_grid)
    N = N_modes * d2
    modes = np.array(full_grid, dtype=np.intp)
    K1 = max(abs(i) for i, j in full_grid)
    K2 = max(abs(j) for i, j in full_grid)
    
    mode_idx_grid = np.full((2 * K1 + 1, 2 * K2 + 1), -1, dtype=np.intp)
    for idx, (i, j) in enumerate(modes):
        mode_idx_grid[i + K1, j + K2] = idx

    L_mat = np.zeros((N, N), dtype=dtype, order='C')   # was order='F' for scipy
    I_d = np.eye(d, dtype=dtype, order='C')
    d2_range = np.arange(d2)

    for p_key, H_p in H.items():
        H_p_work = np.asarray(H_p, dtype=dtype)
        C_p_herm = np.kron(I_d, H_p_work.T) - np.kron(H_p_work, I_d)
        
        k_coords = modes + np.array(p_key)
        valid = (np.abs(k_coords[:, 0]) <= K1) & (np.abs(k_coords[:, 1]) <= K2)
        m_idx = np.where(valid)[0]
        
        if len(m_idx) == 0:
            continue
            
        k_i = k_coords[m_idx, 0]
        k_j = k_coords[m_idx, 1]
        k_idx = mode_idx_grid[k_i + K1, k_j + K2]

        row_starts = k_idx * d2
        col_starts = m_idx * d2

        rows = row_starts[:, None, None] + d2_range[None, :, None]
        cols = col_starts[:, None, None] + d2_range[None, None, :]
        
        L_mat[rows, cols] += C_p_herm[None, :, :]

    # --- Diagonal frequency terms ---
    real_dtype = np.float32 if dtype == np.complex64 else np.float64
    freqs = (modes @ omega).astype(real_dtype, copy=False)
    
    diag_mask = (freqs != 0.0)
    diag_modes_idx = np.where(diag_mask)[0]
    
    if len(diag_modes_idx) > 0:
        diag_starts = diag_modes_idx * d2
        diag_rows = (diag_starts[:, None] + d2_range[None, :]).ravel()
        diag_vals = (-np.repeat(freqs[diag_modes_idx], d2)).astype(dtype, copy=False)
        L_mat[diag_rows, diag_rows] += diag_vals

    return L_mat


# ==============================================================================
# Result Dataclass
# ==============================================================================

@dataclass
class SpectralBumpResult:
    K1: int
    K2: int
    d: int
    omega: np.ndarray
    full_grid_arr: np.ndarray  
    eigenvalues: np.ndarray    
    eigenvectors: np.ndarray   
    c_proj: np.ndarray         
    reg: float                 
    X: Optional[Dict] = None   

    def __post_init__(self):
        if self.X is None:
            self.X = self.composing_filter(self.reg)

    def composing_filter(self, reg_val: float) -> Dict:
        x2 = (np.abs(self.eigenvalues) / reg_val)**2
        mask = x2 < 1.0
        
        f_val = np.zeros(len(self.eigenvalues), dtype=np.float64)
        f_val[mask] = np.exp(1.0 - 1.0 / (1.0 - x2[mask]))

        c_filtered = f_val * self.c_proj
        X_flat = self.eigenvectors @ c_filtered

        full_grid = [tuple(map(int, row)) for row in self.full_grid_arr]
        d2 = self.d * self.d
        fg_idx = {k: idx for idx, k in enumerate(full_grid)}
        
        return {
            k: X_flat[fg_idx[k]*d2 : (fg_idx[k]+1)*d2].reshape(self.d, self.d)
            for k in full_grid
        }

    def recompose_with_new_reg(self, new_reg: float) -> Dict:
        self.reg = new_reg
        self.X = self.composing_filter(new_reg)
        return self.X

    def save(self, save_dir: Optional[Union[str, Path]] = None, timestamp: Optional[str] = None) -> Path:
        out_dir = Path(save_dir) if save_dir is not None else DEFAULT_RESULT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = timestamp if timestamp is not None else datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = out_dir / f"K1_{self.K1}_K2_{self.K2}_spectral_{stamp}.npz"

        np.savez_compressed(
            file_path,
            K1=np.int32(self.K1),
            K2=np.int32(self.K2),
            d=np.int32(self.d),
            omega=np.asarray(self.omega, dtype=float),
            full_grid_arr=self.full_grid_arr,
            eigenvalues=self.eigenvalues,
            eigenvectors=self.eigenvectors,
            c_proj=self.c_proj,
            reg=np.float64(self.reg),
            X=np.array([self.X[tuple(k)] for k in self.full_grid_arr], dtype=complex)
        )
        return file_path


# ==============================================================================
# Core Solver
# ==============================================================================

def solve_eigreg(
    H: dict, K1: int, K2: int, d: int, omega,
    reg: float = 1e-6,
    use_float32: bool = False,          
    verbose: bool = True,
    save_result: bool = False,
    save_dir: Optional[Union[str, Path]] = None,
) -> SpectralBumpResult:
    real_dtype = np.float32 if use_float32 else np.float64
    omega = np.asarray(omega, dtype=real_dtype)
    
    full_grid, _, _, fg_idx = _build_modes(K1, K2)
    d2 = d * d
    N = len(full_grid) * d2

    complex_dtype = np.complex64 if use_float32 else np.complex128
    h_flat = _build_h_flat(H, full_grid, fg_idx, d).astype(complex_dtype)

    if verbose:
        print(f"Building dense Hermitian matrix (N={N}, threads={NUM_CORES}, precision={complex_dtype.__name__}) ...")
    sys.stdout.flush()
    
    L_herm = _build_L_matvec_fast_vectorized(H, full_grid, fg_idx, d, omega, dtype=complex_dtype)

    if verbose:
        print(f"Eigendecomposition: Executing single-shot ILP64 PyTorch CPU Core Solver...")
    sys.stdout.flush()
    
    # 1. Wrap numpy array directly into PyTorch without a memory copy
    L_tens = torch.from_numpy(L_herm)
    
    # 2. Run standard high-performance full spectral solver (Uses 64-bit integer indexing)
    evals_tens, vecs_tens = torch.linalg.eigh(L_tens)
    
    # 3. Pull out full eigenvalues to extract index coordinates
    evals_all = evals_tens.numpy()
    
    sort_idx = np.argsort(np.abs(evals_all))
    max_modes = min(7000, N)
    final_sort_idx = sort_idx[:max_modes]
    
    evals_7k = evals_all[final_sort_idx]
    
    if verbose:
        print(f"Slicing out targeted invariant subspace ({max_modes} modes) and releasing base memory...")
    sys.stdout.flush()

    # 4. Slice out the 7,000 vectors from PyTorch and perform an explicit copy to NumPy.
    # The .copy() completely decouples the sliced array from the large parent tensor.
    V_7k = vecs_tens[:, final_sort_idx].numpy().copy()
    
    # 5. ABSOLUTE PURGE: Wipe out the massive multi-gigabyte tensors immediately
    del L_tens, evals_tens, vecs_tens, L_herm
    gc.collect()  # Force Python to release unreferenced blocks back to the OS heap

    eigenvalues_7k = -1j * evals_7k

    if verbose:
        print(f"Projecting force vector H onto isolated subspace (Parallel BLAS)...")
    sys.stdout.flush()
    c_7k = V_7k.conj().T @ h_flat

    full_grid_arr = np.asarray(full_grid, dtype=np.int32)
    result = SpectralBumpResult(
        K1=K1,
        K2=K2,
        d=d,
        omega=omega,
        full_grid_arr=full_grid_arr,
        eigenvalues=eigenvalues_7k,
        eigenvectors=V_7k,
        c_proj=c_7k,
        reg=reg,
    )

    if save_result:
        out_path = result.save(save_dir=save_dir)
        if verbose:
            print(f"Saved optimized Spectral Result: {out_path}")
    sys.stdout.flush()
    
    return result