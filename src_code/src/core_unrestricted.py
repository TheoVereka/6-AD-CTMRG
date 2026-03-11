"""
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
"""

import numpy as np
import scipy
import opt_einsum as oe
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
import os, sys
import torch




# for now only the case of ABCDEF, nearest neighbor, exact SVD, D^4 >= chi > D^2 .

def normalize_tensor(tensor):
    """
    Normalise a tensor in-place-style by its Frobenius (L2) norm.

    Divides every element by ``||tensor||_F``.  If the norm is exactly zero
    (e.g. an all-zero tensor) the input is returned unchanged to avoid a
    division-by-zero NaN.

    Args:
        tensor (torch.Tensor): Any complex or real tensor of arbitrary shape.

    Returns:
        torch.Tensor: A new tensor with the same shape and dtype as the input
        whose Frobenius norm is 1, or the original tensor if its norm is 0.
    """
    norm = torch.norm(tensor,p=2)
    if norm > 0:
        return tensor / norm
    else:
        return tensor



def initialize_abcdef(initialize_way:str, D_bond:int, d_PHYS:int, noise_scale:float):
    """
    Initialise the six single-layer iPEPS site tensors a, b, c, d, e, f.

    Each tensor has shape ``(D_bond, D_bond, D_bond, d_PHYS)`` where the first
    three indices are virtual bond indices (one per incoming edge of the
    honeycomb-like unit cell) and the last is the physical index.
    All tensors are normalised to unit Frobenius norm after creation.

    Args:
        initialize_way (str): Initialisation strategy.
            ``'random'``  — independent complex-Gaussian random tensors.
            ``'product'`` — product-state initialisation (not yet implemented).
            ``'singlet'`` — Mz=0 singlet initialisation (not yet implemented).
        D_bond (int): Virtual bond dimension D.
        d_PHYS (int): Physical Hilbert-space dimension.
        noise_scale (float): Amplitude of added noise (used by non-random
            initialisations; currently unused for ``'random'``).

    Returns:
        Tuple[torch.Tensor, ...]: ``(a, b, c, d, e, f)``, each of shape
        ``(D_bond, D_bond, D_bond, d_PHYS)``, dtype ``torch.complex64``.

    Raises:
        ValueError: If ``initialize_way`` is not a recognised strategy.
    """

    if initialize_way == 'random' :

        a = normalize_tensor(torch.randn(D_bond, D_bond, D_bond, d_PHYS, dtype=torch.complex64))
        b = normalize_tensor(torch.randn(D_bond, D_bond, D_bond, d_PHYS, dtype=torch.complex64))
        c = normalize_tensor(torch.randn(D_bond, D_bond, D_bond, d_PHYS, dtype=torch.complex64))
        d = normalize_tensor(torch.randn(D_bond, D_bond, D_bond, d_PHYS, dtype=torch.complex64))
        e = normalize_tensor(torch.randn(D_bond, D_bond, D_bond, d_PHYS, dtype=torch.complex64))
        f = normalize_tensor(torch.randn(D_bond, D_bond, D_bond, d_PHYS, dtype=torch.complex64))

    elif initialize_way == 'product' : # product state but always with small noise

        raise NotImplementedError("'product' initialisation is not yet implemented")

    elif initialize_way == 'singlet' : # Mz=0 sector's singlet state representable as PEPS

        raise NotImplementedError("'singlet' initialisation is not yet implemented")

    else :

        raise ValueError(f"Invalid initialize_way: {initialize_way}")
    
    return a,b,c,d,e,f




def abcdef_to_ABCDEF(a,b,c,d,e,f, D_squared:int):
    """
    Build the double-layer ("bra⊗ket") site tensors A, B, C, D, E, F.

    Each double-layer tensor is obtained by contracting a single-layer tensor
    with its complex conjugate over the physical index::

        A_{(u,x),(v,y),(w,z)} = sum_φ  a_{u,v,w,φ} * conj(a_{x,y,z,φ})

    then reshaping the fused virtual indices ``(u,x)``, ``(v,y)``, ``(w,z)``
    into a ``(D_squared, D_squared, D_squared)`` tensor and finally into a
    ``(D_squared, D_squared)`` matrix (two legs fused into one on each side).
    Each output tensor is normalised to unit Frobenius norm.

    Args:
        a, b, c, d, e, f (torch.Tensor): Single-layer site tensors, each of
            shape ``(D_bond, D_bond, D_bond, d_PHYS)``.
        D_squared (int): ``D_bond ** 2``, the bond dimension of the
            double-layer tensors.

    Returns:
        Tuple[torch.Tensor, ...]: ``(A, B, C, D, E, F)``, each of shape
        ``(D_squared, D_squared, D_squared)`` (rank-3, one fused virtual-index
        pair per honeycomb leg), dtype ``torch.complex64``.
    """

    A = oe.contract("uvwφ,xyzφ->uxvywz", a,a.conj(), optimize=[(0,1)], backend='torch')
    A = normalize_tensor(A)
    A = A.reshape(D_squared, D_squared, D_squared)

    B = oe.contract("uvwφ,xyzφ->uxvywz", b,b.conj(), optimize=[(0,1)], backend='torch')
    B = normalize_tensor(B)
    B = B.reshape(D_squared, D_squared, D_squared)

    C = oe.contract("uvwφ,xyzφ->uxvywz", c,c.conj(), optimize=[(0,1)], backend='torch')
    C = normalize_tensor(C)
    C = C.reshape(D_squared, D_squared, D_squared)

    D = oe.contract("uvwφ,xyzφ->uxvywz", d,d.conj(), optimize=[(0,1)], backend='torch')
    D = normalize_tensor(D)
    D = D.reshape(D_squared, D_squared, D_squared)

    E = oe.contract("uvwφ,xyzφ->uxvywz", e,e.conj(), optimize=[(0,1)], backend='torch')
    E = normalize_tensor(E)
    E = E.reshape(D_squared, D_squared, D_squared)

    F = oe.contract("uvwφ,xyzφ->uxvywz", f,f.conj(), optimize=[(0,1)], backend='torch')
    F = normalize_tensor(F)
    F = F.reshape(D_squared, D_squared, D_squared)

    return A,B,C,D,E,F




def trunc_rhoCCC(matC21, matC32, matC13, chi, D_squared):
    """
    Compute truncated CTM projectors and renormalised corner matrices.

    Given the three environment corner transfer matrices (CTMs) ``matC21``,
    ``matC32``, ``matC13`` (each of shape ``(chi*D_squared, chi*D_squared)``),
    this function builds three density-matrix-like objects ``rho32``,
    ``rho13``, ``rho21`` by cyclically multiplying the three corners, then
    decomposes each via a full exact SVD.  The leading ``chi`` singular vectors
    form the isometric projectors ``U1/U2/U3`` (left) and ``V1/V2/V3``
    (right).  These projectors are used to project the *original* corner
    matrices down to their ``(chi, chi)`` truncated versions.

    The truncation is exact (no approximation) when ``chi >= chi_env * D_squared``
    (the full environment rank); otherwise it is the optimal rank-``chi``
    approximation in the Frobenius sense.

    Args:
        matC21 (torch.Tensor): Corner matrix C21, shape ``(chi*D_squared, chi*D_squared)``.
        matC32 (torch.Tensor): Corner matrix C32, shape ``(chi*D_squared, chi*D_squared)``.
        matC13 (torch.Tensor): Corner matrix C13, shape ``(chi*D_squared, chi*D_squared)``.
        chi (int): Target bond dimension after truncation.
        D_squared (int): ``D_bond ** 2``.

    Returns:
        Tuple of 9 tensors ``(V2, C21, U1, V3, C32, U2, V1, C13, U3)``:
            - ``C21``, ``C32``, ``C13``: Truncated, normalised corner matrices
              of shape ``(chi, chi)``.
            - ``U1``, ``U2``, ``U3``: Left isometric projectors, each reshaped
              to ``(chi, D_squared, chi)``.
            - ``V1``, ``V2``, ``V3``: Right isometric projectors, each reshaped
              to ``(chi, chi, D_squared)``.
    """

    rho32 = oe.contract("UZ,ZY,YV->UV",
                               matC13,matC32,matC21,
                               optimize=[(0,1),(0,1)],
                               backend='torch')
    
    U3, sv32, V2 = torch.linalg.svd(rho32)

    U3 = U3[:,:chi].conj()
    V2 = V2[:chi,:].conj()
    
    rho13 = oe.contract("UX,XZ,ZV->UV",
                               matC21,matC13,matC32,
                               optimize=[(0,1),(0,1)],
                               backend='torch')
    
    U1, sv13, V3 = torch.linalg.svd(rho13)

    U1 = U1[:,:chi].conj() #conjugate transpose
    V3 = V3[:chi,:].conj()
    

    rho21 = oe.contract("UY,YX,XV->UV",
                               matC32,matC21,matC13,
                               optimize=[(0,1),(0,1)],
                               backend='torch')
    
    U2, sv21, V1 = torch.linalg.svd(rho21)
    
    U2 = U2[:,:chi].conj()
    V1 = V1[:chi,:].conj()

    C21 = oe.contract("Yy,YX,xX->yx",
                               U1,matC21,V2,
                               optimize=[(0,1),(0,1)],
                               backend='torch')

    C32 = oe.contract("Zz,ZY,yY->zy",
                               U2,matC32,V3,
                               optimize=[(0,1),(0,1)],
                               backend='torch')

    C13 = oe.contract("Xx,XZ,zZ->xz",
                               U3,matC13,V1,
                               optimize=[(0,1),(0,1)],
                               backend='torch')
    
    C21 = normalize_tensor(C21)
    C32 = normalize_tensor(C32)
    C13 = normalize_tensor(C13)

    U1 = U1.reshape(chi,D_squared, chi)
    U2 = U2.reshape(chi,D_squared, chi)
    U3 = U3.reshape(chi,D_squared, chi)

    V1 = V1.reshape(chi, chi,D_squared)
    V2 = V2.reshape(chi, chi,D_squared)
    V3 = V3.reshape(chi, chi,D_squared)

    return V2, C21, U1, V3, C32, U2, V1, C13, U3 #, diagnostic_trunc





def initialize_environmentCTs_2(A,B,C,D,E,F, chi, D_squared):
    """
    Bootstrap the CTMRG environment from the double-layer site tensors.

    Because there is no pre-existing environment, an initial one is constructed
    by contracting pairs of site tensors to form seed corner matrices and seed
    transfer tensors.  Each pair of adjacent site tensors yields a small corner
    matrix and a rectangular transfer tensor:

    Corners (shape ``(D_squared, D_squared)``):
        ``C21AF``, ``C32CB``, ``C13ED``

    Transfer tensors — obtained by SVD-splitting the rank-2 products of two
    site tensors — are then paired and the full ``update_environmentCTs_3to1``
    step is run once to produce the canonical "type-1" environment tuple
    ``(C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E)``.

    Args:
        A, B, C, D, E, F (torch.Tensor): Double-layer site tensors, each of
            shape ``(D_squared, D_squared)``.
        chi (int): Desired nominal bond dimension of the environment.
        D_squared (int): ``D_bond ** 2``.

    Returns:
        Tuple of 9 torch.Tensor: ``(C21CD, C32EF, C13AB, T1F, T2A, T2B,
        T3C, T3D, T1E)`` — the type-1 environment corner matrices (shape
        ``(chi, chi)``) and edge transfer tensors (shape
        ``(chi, D_squared, D_squared)`` or ``(chi, chi, D_squared)`` after
        the first renormalisation step).
    """

    C21AF = oe.contract("oyg,xog->yx", A,F, optimize=[(0,1)], backend='torch')
    C32CB = oe.contract("aoz,ayo->zy", C,B, optimize=[(0,1)], backend='torch')
    C13ED = oe.contract("xbo,obz->xz", E,D, optimize=[(0,1)], backend='torch')

    T2ET3F = oe.contract("ubi,jki,jlk,vlg->ubvg", E,B,C,F, optimize=[(1,2),(0,1),(0,1)], backend='torch')
    T3AT1B = oe.contract("iug,ijk,kjl,avl->ugva", A,D,E,B, optimize=[(1,2),(0,1),(0,1)], backend='torch')
    T1CT2D = oe.contract("aiu,kij,lkj,lbv->uavb", C,F,A,D, optimize=[(1,2),(0,1),(0,1)], backend='torch')

    T2ET3F = T2ET3F.reshape(D_squared*D_squared, D_squared*D_squared)
    T3AT1B = T3AT1B.reshape(D_squared*D_squared, D_squared*D_squared)
    T1CT2D = T1CT2D.reshape(D_squared*D_squared, D_squared*D_squared)

    U2E, sv23, Vdag3F = torch.linalg.svd(T2ET3F)
    U3A, sv31, Vdag1B = torch.linalg.svd(T3AT1B)
    U1C, sv12, Vdag2D = torch.linalg.svd(T1CT2D)

    # adding clip to prevent sqrt of negative numbers due to numerical issues
    # cast to complex64: singular values are real (float32) but will be mixed
    # with complex64 U/V matrices in the diag matmul below — dtypes must match.
    sqrt_sv23 = torch.sqrt(torch.clamp(sv23[:chi], min=1e-9)).to(torch.complex64)
    sqrt_sv31 = torch.sqrt(torch.clamp(sv31[:chi], min=1e-9)).to(torch.complex64)
    sqrt_sv12 = torch.sqrt(torch.clamp(sv12[:chi], min=1e-9)).to(torch.complex64)

    T2E = U2E[:,:chi] @ torch.diag(sqrt_sv23)
    T3A = U3A[:,:chi] @ torch.diag(sqrt_sv31)
    T1C = U1C[:,:chi] @ torch.diag(sqrt_sv12)
    T3F = torch.diag(sqrt_sv23) @ Vdag3F[:chi,:]
    T1B = torch.diag(sqrt_sv31) @ Vdag1B[:chi,:]
    T2D = torch.diag(sqrt_sv12) @ Vdag2D[:chi,:]

    T2E = T2E.reshape(D_squared, D_squared, chi)
    T3A = T3A.reshape(D_squared, D_squared, chi)
    T1C = T1C.reshape(D_squared, D_squared, chi)
    T3F = T3F.reshape(chi, D_squared, D_squared)
    T1B = T1B.reshape(chi, D_squared, D_squared)
    T2D = T2D.reshape(chi, D_squared, D_squared)
    T2E = T2E.permute(2,0,1)
    T3A = T3A.permute(2,0,1)
    T1C = T1C.permute(2,0,1)
    



##################################################################



    matC21CD = oe.contract("YX,MYa,LXβ,amg,lbg->MmLl",
                           C21AF,T1B,T2E,C,D,
                           optimize=[(0,2),(0,3),(1,2),(0,1)],
                           backend='torch')
    
    matC21 = matC21CD.reshape(chi*D_squared,chi*D_squared)
    
    matC32EF = oe.contract("ZY,NZβ,MYg,abn,amg->NnMm",
                           C32CB,T2D,T3A,E,F,
                           optimize=[(0,2),(0,3),(1,2),(0,1)],
                           backend='torch')
    
    matC32 = matC32EF.reshape(chi*D_squared,chi*D_squared)
    
    matC13AB = oe.contract("XZ,LXg,NZa,lbg,abn->LlNn",
                           C13ED,T3F,T1C,A,B,
                           optimize=[(0,2),(0,3),(1,2),(0,1)],
                           backend='torch')
    
    matC13 = matC13AB.reshape(chi*D_squared,chi*D_squared)

    rho32 = oe.contract("UZ,ZY,YV->UV",
                               matC13,matC32,matC21,
                               optimize=[(0,1),(0,1)],
                               backend='torch')
    
    U3, sv32, V2 = torch.linalg.svd(rho32)

    U3 = U3[:,:chi].conj()
    V2 = V2[:chi,:].conj()
    
    rho13 = oe.contract("UX,XZ,ZV->UV",
                               matC21,matC13,matC32,
                               optimize=[(0,1),(0,1)],
                               backend='torch')
    
    U1, sv13, V3 = torch.linalg.svd(rho13)

    U1 = U1[:,:chi].conj() #conjugate transpose
    V3 = V3[:chi,:].conj()
    

    rho21 = oe.contract("UY,YX,XV->UV",
                               matC32,matC21,matC13,
                               optimize=[(0,1),(0,1)],
                               backend='torch')
    
    U2, sv21, V1 = torch.linalg.svd(rho21)
    
    U2 = U2[:,:chi].conj()
    V1 = V1[:chi,:].conj()

    C21 = oe.contract("Yy,YX,xX->yx",
                               U1,matC21,V2,
                               optimize=[(0,1),(0,1)],
                               backend='torch')

    C32 = oe.contract("Zz,ZY,yY->zy",
                               U2,matC32,V3,
                               optimize=[(0,1),(0,1)],
                               backend='torch')

    C13 = oe.contract("Xx,XZ,zZ->xz",
                               U3,matC13,V1,
                               optimize=[(0,1),(0,1)],
                               backend='torch')
    
    C21CD = normalize_tensor(C21)
    C32EF = normalize_tensor(C32)
    C13AB = normalize_tensor(C13)

    U1B = U1.reshape(chi,D_squared, chi)
    U2D = U2.reshape(chi,D_squared, chi)
    U3F = U3.reshape(chi,D_squared, chi)

    V1C = V1.reshape(chi, chi,D_squared)
    V2E = V2.reshape(chi, chi,D_squared)
    V3A = V3.reshape(chi, chi,D_squared)


    T3C = normalize_tensor(oe.contract("OYa,abg,ObM->YMg",
                        T1B,C,U1B,
                        optimize=[(0,1),(0,1)],
                        backend='torch'))

    T3D = normalize_tensor(oe.contract("OXb,abg,LOa->XLg",
                        T2E,D,V2E,
                        optimize=[(0,1),(0,1)],
                        backend='torch'))
    
    T1E = normalize_tensor(oe.contract("OZb,abg,OgN->ZNa",
                        T2D,E,U2D,
                        optimize=[(0,1),(0,1)],
                        backend='torch'))
    
    T1F = normalize_tensor(oe.contract("OYg,abg,MOb->YMa",
                        T3A,F,V3A,
                        optimize=[(0,1),(0,1)],
                        backend='torch'))
    
    T2A = normalize_tensor(oe.contract("OXg,abg,OaL->XLb",
                        T3F,A,U3F,
                        optimize=[(0,1),(0,1)],
                        backend='torch'))
    
    T2B = normalize_tensor(oe.contract("OZa,abg,NOg->ZNa",
                        T1C,B,V1C,
                        optimize=[(0,1),(0,1)],
                        backend='torch'))




#########################################################################
    
    # FUCK MY LIFE AND MAKE IT WORK, THAT'S ENOUGH
    # I'M PHYSICIST NOT A PRO-CODER.


    matC21EB = oe.contract("YX,MYa,LXβ,amg,lbg->MmLl",
                           C21CD,T1F,T2A,E,B,
                           optimize=[(0,2),(0,3),(1,2),(0,1)],
                           backend='torch')
    
    matC21EB = matC21EB.reshape(D_squared*D_squared,D_squared*D_squared)
    
    matC32AD = oe.contract("ZY,NZβ,MYg,abn,amg->NnMm",
                           C32EF,T2B,T3C,A,D,
                           optimize=[(0,2),(0,3),(1,2),(0,1)],
                           backend='torch')
    
    matC32AD = matC32AD.reshape(D_squared*D_squared,D_squared*D_squared)
    
    matC13CF = oe.contract("XZ,LXg,NZa,lbg,abn->LlNn",
                           C13AB,T3D,T1E,C,F,
                           optimize=[(0,2),(0,3),(1,2),(0,1)],
                           backend='torch')
    
    matC13CF = matC13CF.reshape(D_squared*D_squared,D_squared*D_squared)







    #V2A, C21EB, U1F, V3C, C32AD, U2B, V1E, C13CF, U3D = trunc_rhoCCC(
    #                    matC21EB, matC32AD, matC13CF, chi, D_squared)


    rho32 = oe.contract("UZ,ZY,YV->UV",
                               matC13CF,matC32AD,matC21EB,
                               optimize=[(0,1),(0,1)],
                               backend='torch')
    
    U3, sv32, V2 = torch.linalg.svd(rho32)

    U3 = U3[:,:chi].conj()
    V2 = V2[:chi,:].conj()
    
    rho13 = oe.contract("UX,XZ,ZV->UV",
                               matC21EB,matC13CF,matC32AD,
                               optimize=[(0,1),(0,1)],
                               backend='torch')
    
    U1, sv13, V3 = torch.linalg.svd(rho13)

    U1 = U1[:,:chi].conj() #conjugate transpose
    V3 = V3[:chi,:].conj()
    

    rho21 = oe.contract("UY,YX,XV->UV",
                               matC32AD,matC21EB,matC13CF,
                               optimize=[(0,1),(0,1)],
                               backend='torch')
    
    U2, sv21, V1 = torch.linalg.svd(rho21)
    
    U2 = U2[:,:chi].conj()
    V1 = V1[:chi,:].conj()

    C21 = oe.contract("Yy,YX,xX->yx",
                               U1,matC21EB,V2,
                               optimize=[(0,1),(0,1)],
                               backend='torch')

    C32 = oe.contract("Zz,ZY,yY->zy",
                               U2,matC32AD,V3,
                               optimize=[(0,1),(0,1)],
                               backend='torch')

    C13 = oe.contract("Xx,XZ,zZ->xz",
                               U3,matC13CF,V1,
                               optimize=[(0,1),(0,1)],
                               backend='torch')
    
    C21EB = normalize_tensor(C21)
    C32AD = normalize_tensor(C32)
    C13CF = normalize_tensor(C13)

    U1F = U1.reshape(D_squared,D_squared, chi)
    U2B = U2.reshape(D_squared,D_squared, chi)
    U3D = U3.reshape(D_squared,D_squared, chi)

    V1E = V1.reshape(chi, D_squared,D_squared)
    V2A = V2.reshape(chi, D_squared,D_squared)
    V3C = V3.reshape(chi, D_squared,D_squared)






    T3E = normalize_tensor(oe.contract("OYa,abg,ObM->YMg",
                        T1F,E,U1F,
                        optimize=[(0,1),(0,1)],
                        backend='torch'))

    T3B = normalize_tensor(oe.contract("OXb,abg,LOa->XLg",
                        T2A,B,V2A,
                        optimize=[(0,1),(0,1)],
                        backend='torch'))
    
    T1A = normalize_tensor(oe.contract("OZb,abg,OgN->ZNa",
                        T2B,A,U2B,
                        optimize=[(0,1),(0,1)],
                        backend='torch'))
    
    T1D = normalize_tensor(oe.contract("OYg,abg,MOb->YMa",
                        T3C,D,V3C,
                        optimize=[(0,1),(0,1)],
                        backend='torch'))
    
    T2C = normalize_tensor(oe.contract("OXg,abg,OaL->XLb",
                        T3D,C,U3D,
                        optimize=[(0,1),(0,1)],
                        backend='torch'))
    
    T2F = normalize_tensor(oe.contract("OZa,abg,NOg->ZNa",
                        T1E,F,V1E,
                        optimize=[(0,1),(0,1)],
                        backend='torch'))

    return C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A






def check_env_convergence(lastC21CD, lastC32EF, lastC13AB, lastT1F, lastT2A, lastT2B, lastT3C, lastT3D, lastT1E, 
                          nowC21CD, nowC32EF, nowC13AB, nowT1F, nowT2A, nowT2B, nowT3C, nowT3D, nowT1E, 
                          lastC21EB, lastC32AD, lastC13CF, lastT1D, lastT2C, lastT2F, lastT3E, lastT3B, lastT1A, 
                          nowC21EB, nowC32AD, nowC13CF, nowT1D, nowT2C, nowT2F, nowT3E, nowT3B, nowT1A, 
                          lastC21AF, lastC32CB, lastC13ED, lastT1B, lastT2E, lastT2D, lastT3A, lastT3F, lastT1C, 
                          nowC21AF, nowC32CB, nowC13ED, nowT1B, nowT2E, nowT2D, nowT3A, nowT3F, nowT1C,
                          env_conv_threshold):
    """
    Decide whether all 9 CTMRG corner matrices have converged.

    Convergence is measured by comparing the **singular-value spectra** of the
    nine corner matrices (three per environment type) between consecutive
    CTMRG cycles.  Singular values are gauge-invariant — unlike raw tensor
    elements, they are unaffected by the sign/phase ambiguity that arises from
    the SVD-based projectors inside ``trunc_rhoCCC``.  Using raw Frobenius
    distance on the environment tensors would *never* converge because those
    sign flips produce an O(1) Frobenius difference even when the physical
    environment is at its fixed point.

    The convergence metric is:

    .. math::

        d = \\max_{c \\in \\{9\\,\\text{corners}\\}}
              \\max_k \\left|
                \\sigma_k^{\\text{now}}(c) - \\sigma_k^{\\text{last}}(c)
              \\right|

    where :math:`\\sigma_k(c)` are the singular values of corner matrix ``c``
    normalised so that :math:`\\sigma_1 = 1`.  The function returns
    ``True`` when ``d < env_conv_threshold``.

    The function returns ``False`` immediately when any of the ``last*``
    corner tensors is ``None`` (i.e. during the first full cycle where not all
    three environment types have been computed yet).

    Args:
        lastC21CD … lastT1C: The 27 environment tensors from the previous
            CTMRG cycle (corner tensors may be ``None`` during warm-up;
            transfer tensors are accepted but not used for convergence).
        nowC21CD … nowT1C: The 27 environment tensors from the current cycle.
        env_conv_threshold (float): Convergence threshold for ``d``.

    Returns:
        bool: ``True`` iff ``d < env_conv_threshold``, ``False`` otherwise
        (including when any ``last*`` corner tensor is ``None``).
    """

    # Warm-up guard: all three env types must have been computed at least once.
    if lastC21CD is None or lastC21EB is None or lastC21AF is None:
        return False

    # The 9 corner pairs (last, now) across all three environment types.
    corner_pairs = [
        (lastC21CD, nowC21CD), (lastC32EF, nowC32EF), (lastC13AB, nowC13AB),
        (lastC21EB, nowC21EB), (lastC32AD, nowC32AD), (lastC13CF, nowC13CF),
        (lastC21AF, nowC21AF), (lastC32CB, nowC32CB), (lastC13ED, nowC13ED),
    ]

    max_delta = 0.0
    for last_c, now_c in corner_pairs:
        sv_last = torch.linalg.svdvals(last_c).real
        sv_now  = torch.linalg.svdvals(now_c ).real
        # Normalise so that the largest singular value is 1 (scale-invariant).
        sv_last = sv_last / (sv_last[0] + 1e-30)
        sv_now  = sv_now  / (sv_now [0] + 1e-30)
        delta = (sv_now - sv_last).abs().max().item()
        if delta > max_delta:
            max_delta = delta

    return bool(max_delta < env_conv_threshold)



def update_environmentCTs_1to2(C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E, A,B,C,D,E,F, chi, D_squared):
    """
    Perform one CTMRG renormalisation step: environment type 1 → type 2.

    Absorbs the site tensors ``E``, ``B``, ``A``, ``D``, ``C``, ``F`` into
    the type-1 corner and transfer tensors and produces the type-2 set.  The
    three new (grown) corner matrices are truncated to bond dimension ``chi``
    using ``trunc_rhoCCC``, and all six new transfer tensors are renormalised.

    Tensor naming convention:  ``C21EB`` is the corner connecting legs 2→1
    in the sublattice-EB orientation; ``T1D`` is the type-1 transfer tensor
    carrying the ``D``-site bond; etc.

    Args:
        C21CD, C32EF, C13AB (torch.Tensor): Type-1 corner matrices,
            shape ``(chi, chi)``.
        T1F, T2A, T2B, T3C, T3D, T1E (torch.Tensor): Type-1 transfer tensors,
            shape ``(chi, D_squared, D_squared)``.
        A, B, C, D, E, F (torch.Tensor): Double-layer site tensors,
            shape ``(D_squared, D_squared)``.
        chi (int): Target bond dimension.
        D_squared (int): ``D_bond ** 2``.

    Returns:
        Tuple of 9 torch.Tensor: ``(C21EB, C32AD, C13CF, T1D, T2C, T2F,
        T3E, T3B, T1A)`` — the type-2 corners (shape ``(chi, chi)``) and
        transfer tensors (shape ``(chi, D_squared, D_squared)``).
    """

    matC21EB = oe.contract("YX,MYa,LXβ,amg,lbg->MmLl",
                           C21CD,T1F,T2A,E,B,
                           optimize=[(0,2),(0,3),(1,2),(0,1)],
                           backend='torch')
    
    matC21EB = matC21EB.reshape(chi*D_squared,chi*D_squared)
    
    matC32AD = oe.contract("ZY,NZβ,MYg,abn,amg->NnMm",
                           C32EF,T2B,T3C,A,D,
                           optimize=[(0,2),(0,3),(1,2),(0,1)],
                           backend='torch')
    
    matC32AD = matC32AD.reshape(chi*D_squared,chi*D_squared)
    
    matC13CF = oe.contract("XZ,LXg,NZa,lbg,abn->LlNn",
                           C13AB,T3D,T1E,C,F,
                           optimize=[(0,2),(0,3),(1,2),(0,1)],
                           backend='torch')
    
    matC13CF = matC13CF.reshape(chi*D_squared,chi*D_squared)

    V2A, C21EB, U1F, V3C, C32AD, U2B, V1E, C13CF, U3D = trunc_rhoCCC(
                        matC21EB, matC32AD, matC13CF, chi, D_squared)

    T3E = normalize_tensor(oe.contract("OYa,abg,ObM->YMg",
                        T1F,E,U1F,
                        optimize=[(0,1),(0,1)],
                        backend='torch'))

    T3B = normalize_tensor(oe.contract("OXb,abg,LOa->XLg",
                        T2A,B,V2A,
                        optimize=[(0,1),(0,1)],
                        backend='torch'))
    
    T1A = normalize_tensor(oe.contract("OZb,abg,OgN->ZNa",
                        T2B,A,U2B,
                        optimize=[(0,1),(0,1)],
                        backend='torch'))
    
    T1D = normalize_tensor(oe.contract("OYg,abg,MOb->YMa",
                        T3C,D,V3C,
                        optimize=[(0,1),(0,1)],
                        backend='torch'))
    
    T2C = normalize_tensor(oe.contract("OXg,abg,OaL->XLb",
                        T3D,C,U3D,
                        optimize=[(0,1),(0,1)],
                        backend='torch'))
    
    T2F = normalize_tensor(oe.contract("OZa,abg,NOg->ZNa",
                        T1E,F,V1E,
                        optimize=[(0,1),(0,1)],
                        backend='torch'))

    return C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A




def update_environmentCTs_2to3(C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A, A,B,C,D,E,F, chi, D_squared):
    """
    Perform one CTMRG renormalisation step: environment type 2 → type 3.

    Absorbs site tensors ``A``, ``F``, ``C``, ``B``, ``E``, ``D`` into the
    type-2 environment and produces the type-3 set, following the same
    absorb-truncate-renormalise pattern as ``update_environmentCTs_1to2``.

    Args:
        C21EB, C32AD, C13CF (torch.Tensor): Type-2 corner matrices,
            shape ``(chi, chi)``.
        T1D, T2C, T2F, T3E, T3B, T1A (torch.Tensor): Type-2 transfer tensors,
            shape ``(chi, D_squared, D_squared)``.
        A, B, C, D, E, F (torch.Tensor): Double-layer site tensors,
            shape ``(D_squared, D_squared)``.
        chi (int): Target bond dimension.
        D_squared (int): ``D_bond ** 2``.

    Returns:
        Tuple of 9 torch.Tensor: ``(C21AF, C32CB, C13ED, T1B, T2E, T2D,
        T3A, T3F, T1C)`` — the type-3 corners and transfer tensors.
    """

    matC21AF = oe.contract("YX,MYa,LXβ,amg,lbg->MmLl",
                           C21EB,T1D,T2C,A,F,
                           optimize=[(0,2),(0,3),(1,2),(0,1)],
                           backend='torch')
    
    matC21AF = matC21AF.reshape(chi*D_squared,chi*D_squared)
    
    matC32CB = oe.contract("ZY,NZβ,MYg,abn,amg->NnMm",
                           C32AD,T2F,T3E,C,B,
                           optimize=[(0,2),(0,3),(1,2),(0,1)],
                           backend='torch')
    
    matC32CB = matC32CB.reshape(chi*D_squared,chi*D_squared)
    
    matC13ED = oe.contract("XZ,LXg,NZa,lbg,abn->LlNn",
                           C13CF,T3B,T1A,E,D,
                           optimize=[(0,2),(0,3),(1,2),(0,1)],
                           backend='torch')
    
    matC13ED = matC13ED.reshape(chi*D_squared,chi*D_squared)

    V2C, C21AF, U1D, V3E, C32CB, U2F, V1A, C13ED, U3B = trunc_rhoCCC(
                        matC21AF, matC32CB, matC13ED, chi, D_squared)

    T3A = normalize_tensor(oe.contract("OYa,abg,ObM->YMg",
                        T1D,A,U1D,
                        optimize=[(0,1),(0,1)],
                        backend='torch'))

    T3F = normalize_tensor(oe.contract("OXb,abg,LOa->XLg",
                        T2C,F,V2C,
                        optimize=[(0,1),(0,1)],
                        backend='torch'))
    
    T1C = normalize_tensor(oe.contract("OZb,abg,OgN->ZNa",
                        T2F,C,U2F,
                        optimize=[(0,1),(0,1)],
                        backend='torch'))
    
    T1B = normalize_tensor(oe.contract("OYg,abg,MOb->YMa",
                        T3E,B,V3E,
                        optimize=[(0,1),(0,1)],
                        backend='torch'))
    
    T2E = normalize_tensor(oe.contract("OXg,abg,OaL->XLb",
                        T3B,E,U3B,
                        optimize=[(0,1),(0,1)],
                        backend='torch'))
    
    T2D = normalize_tensor(oe.contract("OZa,abg,NOg->ZNa",
                        T1A,D,V1A,
                        optimize=[(0,1),(0,1)],
                        backend='torch'))

    return C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C




def update_environmentCTs_3to1(C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C, A,B,C,D,E,F, chi, D_squared):
    """
    Perform one CTMRG renormalisation step: environment type 3 → type 1.

    Absorbs site tensors ``C``, ``D``, ``E``, ``F``, ``A``, ``B`` into the
    type-3 environment and produces the type-1 set, completing the three-step
    cycle ``1→2→3→1``.

    Args:
        C21AF, C32CB, C13ED (torch.Tensor): Type-3 corner matrices,
            shape ``(chi, chi)``.
        T1B, T2E, T2D, T3A, T3F, T1C (torch.Tensor): Type-3 transfer tensors,
            shape ``(chi, D_squared, D_squared)``.
        A, B, C, D, E, F (torch.Tensor): Double-layer site tensors,
            shape ``(D_squared, D_squared)``.
        chi (int): Target bond dimension.
        D_squared (int): ``D_bond ** 2``.

    Returns:
        Tuple of 9 torch.Tensor: ``(C21CD, C32EF, C13AB, T1F, T2A, T2B,
        T3C, T3D, T1E)`` — the renewed type-1 corners and transfer tensors.
    """

    matC21CD = oe.contract("YX,MYa,LXβ,amg,lbg->MmLl",
                           C21AF,T1B,T2E,C,D,
                           optimize=[(0,2),(0,3),(1,2),(0,1)],
                           backend='torch')
    
    matC21CD = matC21CD.reshape(chi*D_squared,chi*D_squared)
    
    matC32EF = oe.contract("ZY,NZβ,MYg,abn,amg->NnMm",
                           C32CB,T2D,T3A,E,F,
                           optimize=[(0,2),(0,3),(1,2),(0,1)],
                           backend='torch')
    
    matC32EF = matC32EF.reshape(chi*D_squared,chi*D_squared)
    
    matC13AB = oe.contract("XZ,LXg,NZa,lbg,abn->LlNn",
                           C13ED,T3F,T1C,A,B,
                           optimize=[(0,2),(0,3),(1,2),(0,1)],
                           backend='torch')
    
    matC13AB = matC13AB.reshape(chi*D_squared,chi*D_squared)

    V2E, C21CD, U1B, V3A, C32EF, U2D, V1C, C13AB, U3F = trunc_rhoCCC(
                        matC21CD, matC32EF, matC13AB, chi, D_squared)

    T3C = normalize_tensor(oe.contract("OYa,abg,ObM->YMg",
                        T1B,C,U1B,
                        optimize=[(0,1),(0,1)],
                        backend='torch'))

    T3D = normalize_tensor(oe.contract("OXb,abg,LOa->XLg",
                        T2E,D,V2E,
                        optimize=[(0,1),(0,1)],
                        backend='torch'))
    
    T1E = normalize_tensor(oe.contract("OZb,abg,OgN->ZNa",
                        T2D,E,U2D,
                        optimize=[(0,1),(0,1)],
                        backend='torch'))
    
    T1F = normalize_tensor(oe.contract("OYg,abg,MOb->YMa",
                        T3A,F,V3A,
                        optimize=[(0,1),(0,1)],
                        backend='torch'))
    
    T2A = normalize_tensor(oe.contract("OXg,abg,OaL->XLb",
                        T3F,A,U3F,
                        optimize=[(0,1),(0,1)],
                        backend='torch'))
    
    T2B = normalize_tensor(oe.contract("OZa,abg,NOg->ZNa",
                        T1C,B,V1C,
                        optimize=[(0,1),(0,1)],
                        backend='torch'))

    return C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E




def CTMRG_from_init_to_stop(A,B,C,D,E,F,
                            chi: int,
                            D_squared: int,
                            a_third_max_iterations: int,
                            env_conv_threshold: float) -> tuple:
    """
    This function performs the CTMRG algorithm from the initial state to the stopping criterion.

    Args:
        A, B, C, D, E, F (torch.Tensor): The 6 local tensors.
        max_iterations (int): The maximum number of iterations to perform.
        D_squared (int): The square of the bond dimension of the local state projector a~f.
        chi (int): The nominal desired bond dimension for the transfer tensors.
        env_conv_threshold (float): The threshold for environment convergence.

    Returns:
        A tuple of 28 elements: the 27 final environment tensors followed by
        ctm_steps (int) — the number of CTMRG iterations actually performed.
    """
    # Initialize the environment corner and edge transfer tensors

    lastC21CD, lastC32EF, lastC13AB, lastT1F, lastT2A, lastT2B, lastT3C, lastT3D, lastT1E = None, None, None, None, None, None, None, None, None
    lastC21EB, lastC32AD, lastC13CF, lastT1D, lastT2C, lastT2F, lastT3E, lastT3B, lastT1A = None, None, None, None, None, None, None, None, None
    lastC21AF, lastC32CB, lastC13ED, lastT1B, lastT2E, lastT2D, lastT3A, lastT3F, lastT1C = None, None, None, None, None, None, None, None, None
    nowC21CD, nowC32EF, nowC13AB, nowT1F, nowT2A, nowT2B, nowT3C, nowT3D, nowT1E = None, None, None, None, None, None, None, None, None
    nowC21EB, nowC32AD, nowC13CF, nowT1D, nowT2C, nowT2F, nowT3E, nowT3B, nowT1A = initialize_environmentCTs_2(A,B,C,D,E,F, chi, D_squared)
    nowC21AF, nowC32CB, nowC13ED, nowT1B, nowT2E, nowT2D, nowT3A, nowT3F, nowT1C = None, None, None, None, None, None, None, None, None





    # FUCK MY LIFE WITH THIS, THIS WORKS SO SHUT UP AND CALCULATE!!!


    #     case 1 : 
    lastC21AF, lastC32CB, lastC13ED, lastT1B, lastT2E, lastT2D, lastT3A, lastT3F, lastT1C = \
    nowC21AF, nowC32CB, nowC13ED, nowT1B, nowT2E, nowT2D, nowT3A, nowT3F, nowT1C

    nowC21AF, nowC32CB, nowC13ED, nowT1B, nowT2E, nowT2D, nowT3A, nowT3F, nowT1C = update_environmentCTs_2to3(
    nowC21EB, nowC32AD, nowC13CF, nowT1D, nowT2C, nowT2F, nowT3E, nowT3B, nowT1A, A,B,C,D,E,F, chi, D_squared)
        
    #     case 2 : 
    lastC21CD, lastC32EF, lastC13AB, lastT1F, lastT2A, lastT2B, lastT3C, lastT3D, lastT1E = \
    nowC21CD, nowC32EF, nowC13AB, nowT1F, nowT2A, nowT2B, nowT3C, nowT3D, nowT1E
    
    nowC21CD, nowC32EF, nowC13AB, nowT1F, nowT2A, nowT2B, nowT3C, nowT3D, nowT1E = update_environmentCTs_3to1(
    nowC21AF, nowC32CB, nowC13ED, nowT1B, nowT2E, nowT2D, nowT3A, nowT3F, nowT1C, A,B,C,D,E,F, chi, D_squared)







    # Perform the CTMRG iterations until convergence
    ctm_steps = a_third_max_iterations  # will be overwritten on early convergence
    for iteration in range(a_third_max_iterations):

        if check_env_convergence(lastC21CD, lastC32EF, lastC13AB, lastT1F, lastT2A, lastT2B, lastT3C, lastT3D, lastT1E, 
                                 nowC21CD, nowC32EF, nowC13AB, nowT1F, nowT2A, nowT2B, nowT3C, nowT3D, nowT1E, 
                                 lastC21EB, lastC32AD, lastC13CF, lastT1D, lastT2C, lastT2F, lastT3E, lastT3B, lastT1A, 
                                 nowC21EB, nowC32AD, nowC13CF, nowT1D, nowT2C, nowT2F, nowT3E, nowT3B, nowT1A, 
                                 lastC21AF, lastC32CB, lastC13ED, lastT1B, lastT2E, lastT2D, lastT3A, lastT3F, lastT1C, 
                                 nowC21AF, nowC32CB, nowC13ED, nowT1B, nowT2E, nowT2D, nowT3A, nowT3F, nowT1C, 
                                 env_conv_threshold):
            ctm_steps = iteration + 1
            break

        # Update the environment corner and edge transfer tensors
        # match iteration % 3 :
        #     case 0 : 
        lastC21EB, lastC32AD, lastC13CF, lastT1D, lastT2C, lastT2F, lastT3E, lastT3B, lastT1A = \
        nowC21EB, nowC32AD, nowC13CF, nowT1D, nowT2C, nowT2F, nowT3E, nowT3B, nowT1A
        nowC21EB, nowC32AD, nowC13CF, nowT1D, nowT2C, nowT2F, nowT3E, nowT3B, nowT1A = update_environmentCTs_1to2(
        nowC21CD, nowC32EF, nowC13AB, nowT1F, nowT2A, nowT2B, nowT3C, nowT3D, nowT1E, A,B,C,D,E,F, chi, D_squared)
            
        #     case 1 : 
        lastC21AF, lastC32CB, lastC13ED, lastT1B, lastT2E, lastT2D, lastT3A, lastT3F, lastT1C = \
        nowC21AF, nowC32CB, nowC13ED, nowT1B, nowT2E, nowT2D, nowT3A, nowT3F, nowT1C

        nowC21AF, nowC32CB, nowC13ED, nowT1B, nowT2E, nowT2D, nowT3A, nowT3F, nowT1C = update_environmentCTs_2to3(
        nowC21EB, nowC32AD, nowC13CF, nowT1D, nowT2C, nowT2F, nowT3E, nowT3B, nowT1A, A,B,C,D,E,F, chi, D_squared)
            
        #     case 2 : 
        lastC21CD, lastC32EF, lastC13AB, lastT1F, lastT2A, lastT2B, lastT3C, lastT3D, lastT1E = \
        nowC21CD, nowC32EF, nowC13AB, nowT1F, nowT2A, nowT2B, nowT3C, nowT3D, nowT1E
        
        nowC21CD, nowC32EF, nowC13AB, nowT1F, nowT2A, nowT2B, nowT3C, nowT3D, nowT1E = update_environmentCTs_3to1(
        nowC21AF, nowC32CB, nowC13ED, nowT1B, nowT2E, nowT2D, nowT3A, nowT3F, nowT1C, A,B,C,D,E,F, chi, D_squared)
    
    return  nowC21CD, nowC32EF, nowC13AB, nowT1F, nowT2A, nowT2B, nowT3C, nowT3D, nowT1E, \
            nowC21EB, nowC32AD, nowC13CF, nowT1D, nowT2C, nowT2F, nowT3E, nowT3B, nowT1A, \
            nowC21AF, nowC32CB, nowC13ED, nowT1B, nowT2E, nowT2D, nowT3A, nowT3F, nowT1C, \
            ctm_steps







def energy_expectation_nearest_neighbor_6_bonds(a,b,c,d,e,f, 
                                                Hed,Had,Haf,Hcf,Hcb,Heb, # (d_PHYS * d_PHYS^*, d_PHYS * d_PHYS^*) matrices
                                                chi, D_bond, # d_PHYS,
                                                C21CD,C32EF,C13AB,T1F,T2A,T2B,T3C,T3D,T1E):
    """
    Compute the variational energy expectation value for the 6 bonds in the
    type-1 environment.

    The six nearest-neighbour bonds covered here are E-D, A-D, F-A, C-F,
    B-C, E-B (the bonds associated with the type-1 corner/transfer
    environment).  For each bond, an ``open`` tensor is built by contracting
    the single-layer tensor with its conjugate while leaving the physical
    indices un-contracted (the ``ij`` dangling pair).  A ``closed`` version
    (trace over ``ij``) gives the denominator norm contributions.  The
    numerator for bond ``XY`` is obtained by inserting the Hamiltonian
    ``H_XY[i,j,k,l]`` between the open physical legs.

    The final energy is::

        E_6 = (sum over 6 bonds of E_unnormed) / norm_1st_env

    where ``norm_1st_env = Tr[DE * BC * FA]`` (cyclic product of the three
    closed two-site operators around the unit cell).

    Args:
        a, b, c, d, e, f (torch.Tensor): Single-layer site tensors,
            shape ``(D_bond, D_bond, D_bond, d_PHYS)``, with
            ``requires_grad=True`` for the optimisation.
        Hed, Had, Haf, Hcf, Hcb, Heb (torch.Tensor): Two-site nearest-neighbour
            Hamiltonian matrices, shape ``(d_PHYS, d_PHYS, d_PHYS, d_PHYS)``
            (bra1, bra2, ket1, ket2 ordering).
        chi (int): Environment bond dimension.
        D_bond (int): Virtual bond dimension.
        C21CD, C32EF, C13AB (torch.Tensor): Type-1 corner matrices,
            shape ``(chi, chi)``.
        T1F, T2A, T2B, T3C, T3D, T1E (torch.Tensor): Type-1 transfer tensors,
            shape ``(chi*D_bond*D_bond, chi*D_bond*D_bond)`` before the internal
            reshape.

    Returns:
        torch.Tensor: Scalar energy per bond (complex64; imaginary part should
        vanish for a Hermitian Hamiltonian).
    """
    
    T1F = T1F.reshape(chi,chi,D_bond,D_bond)
    T2A = T2A.reshape(chi,chi,D_bond,D_bond)
    T2B = T2B.reshape(chi,chi,D_bond,D_bond)
    T3C = T3C.reshape(chi,chi,D_bond,D_bond)
    T3D = T3D.reshape(chi,chi,D_bond,D_bond)
    T1E = T1E.reshape(chi,chi,D_bond,D_bond)

    open_E = oe.contract("YX,MYar,abci,rstj->MbsXctij", C21CD, T1F, e, e.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch")
    open_D = oe.contract("MYct,abci,rstj->YarMbsij", T3C, d, d.conj(), optimize=[(0,1),(0,1)], backend="torch")
    open_A = oe.contract("ZY,NZbs,abci,rstj->NctYarij", C32EF, T2B, a, a.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch")
    open_F = oe.contract("NZar,abci,rstj->ZbsNctij", T1E, f, f.conj(), optimize=[(0,1),(0,1)], backend="torch")
    open_C = oe.contract("XZ,LXct,abci,rstj->LarZbsij", C13AB, T3D, c, c.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch")
    open_B = oe.contract("LXbs,abci,rstj->XctLarij", T2A, b, b.conj(), optimize=[(0,1),(0,1)], backend="torch")
                              
    closed_E = oe.contract("MbsXctii->MbsXct", open_E, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_D = oe.contract("YarMbsii->YarMbs", open_D, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_A = oe.contract("NctYarii->NctYar", open_A, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_F = oe.contract("ZbsNctii->ZbsNct", open_F, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_C = oe.contract("LarZbsii->LarZbs", open_C, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_B = oe.contract("XctLarii->XctLar", open_B, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)

    H_DE = oe.contract("MbsXctij,ijkl,YarMbskl->YarXct", open_E, Hed, open_D, optimize=[(0,1),(0,1)], backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    H_AD = oe.contract("NctYarij,ijkl,YarMbskl->NctMbs", open_A, Had, open_D, optimize=[(0,1),(0,1)], backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    H_FA = oe.contract("NctYarij,ijkl,ZbsNctkl->ZbsYar", open_A, Haf, open_F, optimize=[(0,1),(0,1)], backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    H_CF = oe.contract("LarZbsij,ijkl,ZbsNctkl->LarNct", open_C, Hcf, open_F, optimize=[(0,1),(0,1)], backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    H_BC = oe.contract("LarZbsij,ijkl,XctLarkl->XctZbs", open_C, Hcb, open_B, optimize=[(0,1),(0,1)], backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    H_EB = oe.contract("MbsXctij,ijkl,XctLarkl->MbsLar", open_E, Heb, open_B, optimize=[(0,1),(0,1)], backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)

    DE = torch.mm(closed_D, closed_E)
    AD = torch.mm(closed_A, closed_D)
    FA = torch.mm(closed_F, closed_A)
    CF = torch.mm(closed_C, closed_F)
    BC = torch.mm(closed_B, closed_C)
    EB = torch.mm(closed_E, closed_B)

    E_unnormed_DE = oe.contract("xy,yz,zx->", H_DE, BC, FA, backend="torch")
    E_unnormed_AD = oe.contract("xy,yz,zx->", H_AD, EB, CF, backend="torch")
    E_unnormed_FA = oe.contract("xy,yz,zx->", H_FA, DE, BC, backend="torch")
    E_unnormed_CF = oe.contract("xy,yz,zx->", H_CF, AD, EB, backend="torch")
    E_unnormed_BC = oe.contract("xy,yz,zx->", H_BC, FA, DE, backend="torch")
    E_unnormed_EB = oe.contract("xy,yz,zx->", H_EB, CF, AD, backend="torch")

    norm_1st_env = oe.contract("xy,yz,zx->", DE, BC, FA, backend="torch")
    energyNearestNeighbor_6_bonds = (E_unnormed_DE + E_unnormed_AD + E_unnormed_FA + E_unnormed_CF + E_unnormed_BC + E_unnormed_EB) / norm_1st_env
    return energyNearestNeighbor_6_bonds



def energy_expectation_nearest_neighbor_other_3_bonds(a,b,c,d,e,f, 
                                                      Hcd,Hef,Hab, # (d_PHYS, d_PHYS^*, d_PHYS, d_PHYS^*) matrices
                                                      chi, D_bond, # d_PHYS,
                                                      C21AF,C32CB,C13ED,T1B,T2E,T2D,T3A,T3F,T1C):
    """
    Compute the variational energy expectation value for the remaining 3 bonds
    in the type-3 environment.

    This function covers bonds C-D, E-F, A-B (the bonds not handled by
    ``energy_expectation_nearest_neighbor_6_bonds``).  The same open/closed
    tensor construction is used, but the environment here is the type-3 set
    ``(C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C)``.

    The energy is::

        E_3 = (E_EF + E_AB + E_CD) / norm_3rd_env

    where ``norm_3rd_env = Tr[EF * CD * AB]``.

    Args:
        a, b, c, d, e, f (torch.Tensor): Single-layer site tensors.
        Hcd, Hef, Hab (torch.Tensor): Two-site Hamiltonians for the three
            remaining bonds, shape ``(d_PHYS, d_PHYS, d_PHYS, d_PHYS)``.
        chi (int): Environment bond dimension.
        D_bond (int): Virtual bond dimension.
        C21AF, C32CB, C13ED (torch.Tensor): Type-3 corner matrices,
            shape ``(chi, chi)``.
        T1B, T2E, T2D, T3A, T3F, T1C (torch.Tensor): Type-3 transfer tensors.

    Returns:
        torch.Tensor: Scalar energy per bond (complex64).
    """
    T1B = T1B.reshape(chi,chi,D_bond,D_bond)
    T2E = T2E.reshape(chi,chi,D_bond,D_bond)
    T2D = T2D.reshape(chi,chi,D_bond,D_bond)
    T3A = T3A.reshape(chi,chi,D_bond,D_bond)
    T3F = T3F.reshape(chi,chi,D_bond,D_bond)
    T1C = T1C.reshape(chi,chi,D_bond,D_bond)

    open_C = oe.contract("YX,MYar,abci,rstj->MbsXctij", C21AF, T1B, c, c.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch")
    open_F = oe.contract("MYct,abci,rstj->YarMbsij", T3A, f, f.conj(), optimize=[(0,1),(0,1)], backend="torch")
    open_E = oe.contract("ZY,NZbs,abci,rstj->NctYarij", C32CB, T2D, e, e.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch")
    open_B = oe.contract("NZar,abci,rstj->ZbsNctij", T1C, b, b.conj(), optimize=[(0,1),(0,1)], backend="torch")
    open_A = oe.contract("XZ,LXct,abci,rstj->LarZbsij", C13ED, T3F, a, a.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch")
    open_D = oe.contract("LXbs,abci,rstj->XctLarij", T2E, d, d.conj(), optimize=[(0,1),(0,1)], backend="torch")
                              
    closed_C = oe.contract("MbsXctii->MbsXct", open_C, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_F = oe.contract("YarMbsii->YarMbs", open_F, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_E = oe.contract("NctYarii->NctYar", open_E, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_B = oe.contract("ZbsNctii->ZbsNct", open_B, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_A = oe.contract("LarZbsii->LarZbs", open_A, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_D = oe.contract("XctLarii->XctLar", open_D, backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)

    H_EF = oe.contract("NctYarij,ijkl,YarMbskl->NctMbs", open_E, Hef, open_F, optimize=[(0,1),(0,1)], backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    H_AB = oe.contract("LarZbsij,ijkl,ZbsNctkl->LarNct", open_A, Hab, open_B, optimize=[(0,1),(0,1)], backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    H_CD = oe.contract("MbsXctij,ijkl,XctLarkl->MbsLar", open_C, Hcd, open_D, optimize=[(0,1),(0,1)], backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)

    EF = torch.mm(closed_E, closed_F)
    AB = torch.mm(closed_A, closed_B)
    CD = torch.mm(closed_C, closed_D)

    E_unnormed_EF = oe.contract("xy,yz,zx->", H_EF, CD, AB, backend="torch")
    E_unnormed_AB = oe.contract("xy,yz,zx->", H_AB, EF, CD, backend="torch")
    E_unnormed_CD = oe.contract("xy,yz,zx->", H_CD, AB, EF, backend="torch")

    norm_3rd_env= oe.contract("xy,yz,zx->", EF, CD, AB, backend="torch")
    energyNearestNeighbor_3_bonds = (E_unnormed_EF + E_unnormed_AB + E_unnormed_CD) / norm_3rd_env
    return energyNearestNeighbor_3_bonds




def optmization_iPEPS(Hed,Had,Haf,Hcf,Hcb,Heb,Hcd,Hef,Hab, # (d_PHYS, d_PHYS^*, d_PHYS, d_PHYS^*) matrices
                      opt_conv_threshold: float = 1e-6, # SCALES with Hamiltonian!!!!
                      chi: int=10, D_bond: int=3, d_PHYS: int=2,
                      a_third_max_steps_CTMRG: int = 70, 
                      CTM_env_conv_threshold: float = 1e-7,
                      a2f_initialize_way: str = 'random',
                      a2f_noise_scale: float = 1e-3,
                      max_opt_steps: int = 200,
                      lbfgs_max_iter: int = 20,
                      lbfgs_lr: float = 1.0,
                      lbfgs_history: int = 100,
                      opt_tolerance_grad: float = 1e-7,
                      opt_tolerance_change: float = 1e-8,
                      init_abcdef=None):
    """
    Optimize the iPEPS tensors a,b,c,d,e,f using L-BFGS.

    Strategy (standard for AD-CTMRG):
      • Outer loop  (max_opt_steps iterations):
          1. Recompute double-layer tensors A..F and run CTMRG to convergence
             inside torch.no_grad() — environment is treated as fixed.
          2. Run L-BFGS (up to lbfgs_max_iter sub-steps with strong-Wolfe line
             search).  The closure recomputes only the cheap energy evaluation
             and calls backward() through a..f; gradients do NOT flow back
             through the CTMRG iterations.
          3. Check outer convergence on the loss change.

    This is the "environment-fixed" / implicit-differentiation variant.
    Gradients through the full CTMRG unroll are available in principle but
    are prohibitively expensive for production runs.

    Args:
        Hab..Hfa                : 2-site nearest-neighbour Hamiltonians
        chi                     : environment bond dimension
        D_bond                  : physical projector bond dimension
        d_PHYS                  : physical Hilbert-space dimension
        a_third_max_steps_CTMRG  : max CTMRG sweeps per environment update
        CTM_env_conv_threshold      : CTMRG convergence criterion
        a2f_initialize_way      : initialisation mode ('random', 'product', ...)
        a2f_noise_scale         : noise amplitude for initialisation
        max_opt_steps           : max outer optimisation iterations
        lbfgs_max_iter          : max L-BFGS sub-iterations per outer step
        lbfgs_lr                : initial step size for L-BFGS
        lbfgs_history           : L-BFGS history size
        opt_conv_threshold      : stop when |Δloss| < this value

    Returns:
        a, b, c, d, e, f  —  optimised site tensors. TODO:(still require_grad=True),
    """
    D_squared = D_bond ** 2

    # ── 1. Initialise site tensors ────────────────────────────────────────────
    if init_abcdef is not None:
        a, b, c, d, e, f = [t.detach().clone().to(torch.complex64) for t in init_abcdef]
    else:
        a, b, c, d, e, f = initialize_abcdef(a2f_initialize_way, D_bond, d_PHYS, a2f_noise_scale)
    a.requires_grad_(True)
    b.requires_grad_(True)
    c.requires_grad_(True)
    d.requires_grad_(True)
    e.requires_grad_(True)
    f.requires_grad_(True)

    prev_loss = None
    loss_item: float = float('nan')

    # ── 2 & 3. Outer optimisation loop ───────────────────────────────────────
    # Note: a–f are scale-redundant (the energy is a ratio, hence invariant to
    # the common norm of each tensor).  We therefore normalise each tensor after
    # every outer step so that their scale never drifts.  Because the environment
    # is re-converged from scratch at the start of every outer step, the L-BFGS
    # curvature history from the previous outer step is stale; a fresh optimizer
    # is created each iteration — this costs nothing and avoids misleading
    # quasi-Newton updates built against a different loss landscape.
    for opt_step in range(max_opt_steps):

        # 3a. Normalise a–f (scale redundancy; no-op on first step if init is
        #     already unit-norm, harmless otherwise).  Done via .data so that
        #     the computation graph and grad accumulators are untouched.
        with torch.no_grad():
            a.data = normalize_tensor(a.data)
            b.data = normalize_tensor(b.data)
            c.data = normalize_tensor(c.data)
            d.data = normalize_tensor(d.data)
            e.data = normalize_tensor(e.data)
            f.data = normalize_tensor(f.data)

        # 3b. Fresh L-BFGS for this outer step (environment will change, so old
        #     curvature history is irrelevant and would only mislead the solver).
        optimizer = torch.optim.LBFGS(
            [a, b, c, d, e, f],
            lr=lbfgs_lr,
            max_iter=lbfgs_max_iter,
            tolerance_grad=opt_tolerance_grad,
            tolerance_change=opt_tolerance_change,
            history_size=lbfgs_history,
            line_search_fn='strong_wolfe',
        )

        # 3c. Rebuild double-layer tensors and converge the CTMRG environment.
        #     No gradients needed here — the environment is treated as a fixed
        #     external field during the L-BFGS line search.
        with torch.no_grad():
            A, B, C, D, E, F = abcdef_to_ABCDEF(a, b, c, d, e, f, D_squared)
            
            (C21CD, C32EF, C13AB, T1F,  T2A,  T2B,  T3C,  T3D,  T1E,
             C21EB, C32AD, C13CF, T1D,  T2C,  T2F,  T3E,  T3B,  T1A,
             C21AF, C32CB, C13ED, T1B,  T2E,  T2D,  T3A,  T3F,  T1C,
             _ctm_steps) = \
            CTMRG_from_init_to_stop(A, B, C, D, E, F, chi, D_squared,
                a_third_max_steps_CTMRG, CTM_env_conv_threshold)

        # 3d. L-BFGS closure: energy evaluation + backward through a...f only.
        def closure():
            optimizer.zero_grad()

            loss = (energy_expectation_nearest_neighbor_6_bonds(
                        a,b,c,d,e,f, 
                        Hed,Had,Haf,Hcf,Hcb,Heb, 
                        chi, D_bond, # d_PHYS, 
                        C21CD,C32EF,C13AB,T1F,T2A,T2B,T3C,T3D,T1E)
                    +
                    energy_expectation_nearest_neighbor_other_3_bonds(
                        a,b,c,d,e,f, 
                        Hcd,Hef,Hab, 
                        chi, D_bond, # d_PHYS, 
                        C21AF,C32CB,C13ED,T1B,T2E,T2D,T3A,T3F,T1C)).real

            loss.backward()
            return loss

        # 3e. Run L-BFGS sub-iterations (strong-Wolfe line search built in).
        loss_val = optimizer.step(closure)

        loss_item = loss_val.item()
        delta = (loss_item - prev_loss) if prev_loss is not None else float('inf')
        print(f"  opt {opt_step:4d}  loss = {loss_item:+.10f}  Δloss = {delta:.3e}")

        # 3f. Outer convergence check.
        if prev_loss is not None and abs(delta) < opt_conv_threshold:
            print(f"  Outer convergence achieved at step {opt_step} (Δloss={delta:.3e}).")
            break
        prev_loss = loss_item

    return a, b, c, d, e, f, loss_item



def check_optimized_iPEPS(a,b,c,d,e,f, old_loss, 
                          Hed,Had,Haf,Hcf,Hcb,Heb,Hcd,Hef,Hab, # (d_PHYS, d_PHYS^*, d_PHYS, d_PHYS^*) matrices
                          new_chi, D_bond, d_PHYS,
                          a_third_max_steps_CTMRG: int = 70, 
                          CTM_env_conv_threshold: float = 1e-7,
                        # ↓ SCALES with Hamiltonian!!!! ↓
                          delta_loss_threshold: float = 1e-6):
    

    D_squared = D_bond ** 2

    with torch.no_grad():
        A, B, C, D, E, F = abcdef_to_ABCDEF(a, b, c, d, e, f, D_squared)
        
        (C21CD, C32EF, C13AB, T1F,  T2A,  T2B,  T3C,  T3D,  T1E,
         C21EB, C32AD, C13CF, T1D,  T2C,  T2F,  T3E,  T3B,  T1A,
         C21AF, C32CB, C13ED, T1B,  T2E,  T2D,  T3A,  T3F,  T1C,
         _ctm_steps) = \
        CTMRG_from_init_to_stop(A, B, C, D, E, F, new_chi, D_squared,
                a_third_max_steps_CTMRG, CTM_env_conv_threshold)
        
        new_loss_under_new_chi = (
            energy_expectation_nearest_neighbor_6_bonds(
                a,b,c,d,e,f, 
                Hed,Had,Haf,Hcf,Hcb,Heb, 
                new_chi, D_bond, # d_PHYS, 
                C21CD,C32EF,C13AB,T1F,T2A,T2B,T3C,T3D,T1E)
            +
            energy_expectation_nearest_neighbor_other_3_bonds(
                a,b,c,d,e,f, 
                Hcd,Hef,Hab, 
                new_chi, D_bond, # d_PHYS, 
                C21AF,C32CB,C13ED,T1B,T2E,T2D,T3A,T3F,T1C)
        ).item()
        
        delta_loss = new_loss_under_new_chi - old_loss
        print(f"  Check optimized iPEPS with chi={new_chi}: loss = {new_loss_under_new_chi:+.10f}  Δloss = {delta_loss:.3e}")
        return bool(abs(delta_loss) < delta_loss_threshold)