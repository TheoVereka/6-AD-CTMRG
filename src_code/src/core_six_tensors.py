
import torch
import opt_einsum as oe


CDTYPE: torch.dtype = torch.complex128   # complex dtype for all tensors
RDTYPE: torch.dtype = torch.float64    # real dtype (SVD singular values, norms)
TENSORDTYPE: torch.dtype = torch.float64  # tensor dtype (default)



def set_dtype(use_double: bool, use_real: bool) -> None:
    """Switch all core computations between float32/complex64 and float64/complex128.

    Must be called BEFORE any tensor is allocated — ideally right after import.

    Args:
        use_double: ``True``  → complex128 / float64 (double precision).
                    ``False`` → complex64  / float32 (single precision, default).
    """
    global CDTYPE, RDTYPE, TENSORDTYPE

    if use_double:
        CDTYPE = torch.complex128
        RDTYPE = torch.float64
    else:
        CDTYPE = torch.complex64
        RDTYPE = torch.float32

    if use_real:
        TENSORDTYPE = RDTYPE
    else:
        TENSORDTYPE = CDTYPE
    



def normalize_single_layer_tensor_for_double_layer(
    tensor: torch.Tensor,
    *,
    rtol: float | None = None,
    atol: float | None = None,
) -> torch.Tensor:
    """Rescale a single-layer site tensor so its *unnormalized* double-layer has unit Frobenius norm.

    In this codebase, CTMRG consumes double-layer site tensors (A..F) produced by
    ``abcdef_to_ABCDEF`` which *then* normalizes each A..F to unit Frobenius norm.
    If energy/contraction routines use the original single-layer tensors directly,
    a convention mismatch can make denominators like <iPEPS|iPEPS> far from 1 even
    when the CTMRG environment itself is well-normalized.

    This helper chooses a single-layer rescaling such that the corresponding
    unnormalized double-layer tensor already has ||DL||_F ≈ 1, making single-layer
    contractions consistent with the CTMRG convention.

    Scaling rule:
      DL(t) = contract(t, conj(t)) so ||DL(α t)||_F = |α|^2 ||DL(t)||_F.
      Choose α = 1/sqrt(||DL(t)||_F) ⇒ ||DL(α t)||_F = 1.
    """
    # Compute the unnormalized double-layer tensor DL(t) (rank-6), then fuse
    # the three (D,D) virtual index pairs into (D^2,D^2,D^2) like abcdef_to_ABCDEF.
    dl = oe.contract(
        "uvwp,xyzp->uxvywz",
        tensor,
        tensor.conj(),
        optimize=[(0, 1)],
        backend="torch",
    )
    D_bond = int(tensor.shape[0])
    D_squared = D_bond * D_bond
    dl = dl.reshape(D_squared, D_squared, D_squared)

    dl_norm = torch.linalg.norm(dl)

    # Choose dtype-appropriate default tolerances.
    if rtol is None or atol is None:
        if dl_norm.dtype == torch.float32:
            default_rtol = 4e-7
            default_atol = 4e-7
        else:
            default_rtol = 1e-13
            default_atol = 1e-13
        rtol = default_rtol if rtol is None else rtol
        atol = default_atol if atol is None else atol

    if not torch.isfinite(dl_norm):
        return tensor
    if dl_norm <= atol:
        return tensor

    one = torch.ones((), dtype=dl_norm.dtype, device=dl_norm.device)
    if torch.isclose(dl_norm, one, rtol=rtol, atol=atol):
        return tensor

    return tensor / torch.sqrt(dl_norm)



def warm_padding_init_abcdef(a,b,c,d,e,f, D_bond:int, d_PHYS:int, noise_scale:float):
    """
    pad from smaller old_D_bond of abcdef, to now D_bond, adding small noise.
    """

    pass



def initialize_abcdef(initialize_way:str, D_bond:int, d_PHYS:int, noise_scale:float, 
                      a:torch.Tensor|None=None,
                      b:torch.Tensor|None=None,
                      c:torch.Tensor|None=None,
                      d:torch.Tensor|None=None,
                      e:torch.Tensor|None=None,
                      f:torch.Tensor|None=None) -> tuple[torch.Tensor, ...]:
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
        ``(D_bond, D_bond, D_bond, d_PHYS)``, dtype ``TENSORDTYPE``.

    Raises:
        ValueError: If ``initialize_way`` is not a recognised strategy.
    """

    if initialize_way == 'random' :

        a = torch.randn(D_bond, D_bond, D_bond, d_PHYS, dtype=TENSORDTYPE)
        b = torch.randn(D_bond, D_bond, D_bond, d_PHYS, dtype=TENSORDTYPE)
        c = torch.randn(D_bond, D_bond, D_bond, d_PHYS, dtype=TENSORDTYPE)
        d = torch.randn(D_bond, D_bond, D_bond, d_PHYS, dtype=TENSORDTYPE)
        e = torch.randn(D_bond, D_bond, D_bond, d_PHYS, dtype=TENSORDTYPE)
        f = torch.randn(D_bond, D_bond, D_bond, d_PHYS, dtype=TENSORDTYPE)

    elif initialize_way == 'warm' : # padding old_D_bond to now D_bond with small noise
        if a is None or b is None or c is None or d is None or e is None or f is None:
            raise ValueError("All old tensors must be provided for 'warm' initialization")
        a, b, c, d, e, f = warm_padding_init_abcdef(a, b, c, d, e, f, D_bond, d_PHYS, noise_scale)

    elif initialize_way == 'product' : # product state but always with small noise

        raise NotImplementedError("'product' initialisation is not yet implemented")

    elif initialize_way == 'singlet' : # Mz=0 sector's singlet state representable as PEPS

        raise NotImplementedError("'singlet' initialisation is not yet implemented")

    else :

        raise ValueError(f"Invalid initialize_way: {initialize_way}")
    
    a = normalize_single_layer_tensor_for_double_layer(a)
    b = normalize_single_layer_tensor_for_double_layer(b)
    c = normalize_single_layer_tensor_for_double_layer(c)
    d = normalize_single_layer_tensor_for_double_layer(d)
    e = normalize_single_layer_tensor_for_double_layer(e)
    f = normalize_single_layer_tensor_for_double_layer(f)
    
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

    A = oe.contract("uvwp,xyzp->uxvywz",a,a.conj(),optimize=[(0, 1)],backend='torch').reshape(D_squared, D_squared, D_squared)
    B = oe.contract("uvwp,xyzp->uxvywz",b,b.conj(),optimize=[(0, 1)],backend='torch').reshape(D_squared, D_squared, D_squared)
    C = oe.contract("uvwp,xyzp->uxvywz",c,c.conj(),optimize=[(0, 1)],backend='torch').reshape(D_squared, D_squared, D_squared)
    D = oe.contract("uvwp,xyzp->uxvywz",d,d.conj(),optimize=[(0, 1)],backend='torch').reshape(D_squared, D_squared, D_squared)
    E = oe.contract("uvwp,xyzp->uxvywz",e,e.conj(),optimize=[(0, 1)],backend='torch').reshape(D_squared, D_squared, D_squared)
    F = oe.contract("uvwp,xyzp->uxvywz",f,f.conj(),optimize=[(0, 1)],backend='torch').reshape(D_squared, D_squared, D_squared)

    return A,B,C,D,E,F


def truncate_SVD(M,chi, ...!!! ):
    
    """
    TODO: let TrRho re-normalization of C be a bool option, default False
    """


#################################
# TODO: GPU USE gesvd! -- Fisherman: https://journals.aps.org/prb/pdf/10.1103/PhysRevB.98.235148
#################################



    pass


def get_2_projectors_from_2_matrices():
    """
    should also work for the T-T long leg in initialization of environment_1
    """
    pass


def get_6_projectors_from_3_Corners(, 
                                    CprojCCanticlockwise=True):
    """
    Don't use Juraj convention, he forgot to QR decomposition! 
    thus the invertibility of R is not guaranteed, 
    and the projectors may not be well-defined. 
    """
    pass


def initialize_envCTs_1(A,B,C,D,E,F, chi, D_squared, identity_init=False):
    pass


def update_envCTs():
    """
    contract enlargded C*3,
        passing to get_6_projectors_from_3_Corners
    directly contract "smallT, localtensor, projector"

    TODO: let TrRho re-normalization of T be a bool option, default False
    """
    pass


def check_3envs_convergence_from_Corners(
    lastC21CD, lastC32EF, lastC13AB, # lastT1F, lastT2A, lastT2B, lastT3C, lastT3D, lastT1E, 
    nowC21CD, nowC32EF, nowC13AB, # nowT1F, nowT2A, nowT2B, nowT3C, nowT3D, nowT1E, 
    lastC21EB, lastC32AD, lastC13CF, # lastT1D, lastT2C, lastT2F, lastT3E, lastT3B, lastT1A, 
    nowC21EB, nowC32AD, nowC13CF, # nowT1D, nowT2C, nowT2F, nowT3E, nowT3B, nowT1A, 
    lastC21AF, lastC32CB, lastC13ED, # lastT1B, lastT2E, lastT2D, lastT3A, lastT3F, lastT1C, 
    nowC21AF, nowC32CB, nowC13ED, # nowT1B, nowT2E, nowT2D, nowT3A, nowT3F, nowT1C,
    env_conv_threshold):

    """
    Check convergence of the CTMRG environment by comparing the three pairs of corner-defined rho matrices.
    """
   
    # Warm-up guard: all three env types must have been computed at least once.
    if lastC21CD is None or lastC21EB is None or lastC21AF is None:
        return False

    # The rhos defined by 1-1 cut: 13 @ 32 @ 21
    last_rhoABEFCD = oe.contract("UZ,ZY,YV->UV",
                                lastC13AB.detach(),lastC32EF.detach(),lastC21CD.detach(),
                                optimize=[(0,1),(0,1)],
                                backend='torch')
    now_rhoABEFCD = oe.contract("UZ,ZY,YV->UV",
                                nowC13AB.detach(),nowC32EF.detach(),nowC21CD.detach(),
                                optimize=[(0,1),(0,1)],
                                backend='torch')
    last_rhoCFADEB = oe.contract("UZ,ZY,YV->UV",
                                lastC13CF.detach(),lastC32AD.detach(),lastC21EB.detach(),
                                optimize=[(0,1),(0,1)],
                                backend='torch')
    now_rhoCFADEB = oe.contract("UZ,ZY,YV->UV",
                                nowC13CF.detach(),nowC32AD.detach(),nowC21EB.detach(),
                                optimize=[(0,1),(0,1)],
                                backend='torch')
    last_rhoEDCBAF = oe.contract("UZ,ZY,YV->UV",
                                lastC13ED.detach(),lastC32CB.detach(),lastC21AF.detach(),
                                optimize=[(0,1),(0,1)],
                                backend='torch')
    now_rhoEDCBAF = oe.contract("UZ,ZY,YV->UV",
                                nowC13ED.detach(),nowC32CB.detach(),nowC21AF.detach(), 
                                optimize=[(0,1),(0,1)],
                                backend='torch')

    rho_pairs = [
        (last_rhoABEFCD, now_rhoABEFCD),
        (last_rhoCFADEB, now_rhoCFADEB),
        (last_rhoEDCBAF, now_rhoEDCBAF),
    ]

    max_delta = 0.0
    for last_c, now_c in rho_pairs:
        sv_last = torch.linalg.svdvals(last_c).real
        sv_now  = torch.linalg.svdvals(now_c ).real
        # Normalise so that the largest singular value is 1 (scale-invariant).
        sv_last = sv_last / (sv_last[0] + 1e-30)
        sv_now  = sv_now  / (sv_now [0] + 1e-30)
        delta = (sv_now - sv_last).abs().max().item()
        if delta > max_delta:
            max_delta = delta

    return bool(max_delta < env_conv_threshold)



def CTMRG_from_init_to_stop(A,B,C,D,E,F,
                            chi: int,
                            D_squared: int,
                            a_third_max_iterations: int,
                            env_conv_threshold: float,
                            identity_init: bool = False):
    
    lastC21CD, lastC32EF, lastC13AB = None, None, None
    lastC21EB, lastC32AD, lastC13CF = None, None, None
    lastC21AF, lastC32CB, lastC13ED = None, None, None
    nowC21CD, nowC32EF, nowC13AB, nowT1F, nowT2A, nowT2B, nowT3C, nowT3D, nowT1E = initialize_envCTs_1(A,B,C,D,E,F, chi, D_squared, identity_init=identity_init)
    nowC21EB, nowC32AD, nowC13CF, nowT1D, nowT2C, nowT2F, nowT3E, nowT3B, nowT1A = None, None, None, None, None, None, None, None, None
    nowC21AF, nowC32CB, nowC13ED, nowT1B, nowT2E, nowT2D, nowT3A, nowT3F, nowT1C = None, None, None, None, None, None, None, None, None


    # Perform the CTMRG iterations until convergence
    ctm_steps = a_third_max_iterations  # will be overwritten on early convergence
    for iteration in range(a_third_max_iterations):

        if check_3envs_convergence_from_Corners(lastC21CD, lastC32EF, lastC13AB,
                                                nowC21CD, nowC32EF, nowC13AB,
                                                lastC21EB, lastC32AD, lastC13CF,
                                                nowC21EB, nowC32AD, nowC13CF,
                                                lastC21AF, lastC32CB, lastC13ED, 
                                                nowC21AF, nowC32CB, nowC13ED, 
                                                env_conv_threshold):
            ctm_steps = iteration + 1
            break

        # Update the environment corner and edge transfer tensors
        # match iteration % 3 :
        #     case 0 : 
        lastC21EB, lastC32AD, lastC13CF = nowC21EB, nowC32AD, nowC13CF
        nowC21EB, nowC32AD, nowC13CF, nowT1D, nowT2C, nowT2F, nowT3E, nowT3B, nowT1A = update_envCTs(
        nowC21CD, nowC32EF, nowC13AB, nowT1F, nowT2A, nowT2B, nowT3C, nowT3D, nowT1E, rearrange!A,B,C,D,E,F, chi, D_squared)
            
        #     case 1 : 
        lastC21AF, lastC32CB, lastC13ED = nowC21AF, nowC32CB, nowC13ED
        nowC21AF, nowC32CB, nowC13ED, nowT1B, nowT2E, nowT2D, nowT3A, nowT3F, nowT1C = update_envCTs(
        nowC21EB, nowC32AD, nowC13CF, nowT1D, nowT2C, nowT2F, nowT3E, nowT3B, nowT1A, rearrange!A,B,C,D,E,F, chi, D_squared)
            
        #     case 2 : 
        lastC21CD, lastC32EF, lastC13AB = nowC21CD, nowC32EF, nowC13AB
        nowC21CD, nowC32EF, nowC13AB, nowT1F, nowT2A, nowT2B, nowT3C, nowT3D, nowT1E = update_envCTs(
        nowC21AF, nowC32CB, nowC13ED, nowT1B, nowT2E, nowT2D, nowT3A, nowT3F, nowT1C, rearrange!A,B,C,D,E,F, chi, D_squared)
    
    diagnostic = (ctm_steps,)

    return  nowC21CD, nowC32EF, nowC13AB, nowT1F, nowT2A, nowT2B, nowT3C, nowT3D, nowT1E, \
            nowC21EB, nowC32AD, nowC13CF, nowT1D, nowT2C, nowT2F, nowT3E, nowT3B, nowT1A, \
            nowC21AF, nowC32CB, nowC13ED, nowT1B, nowT2E, nowT2D, nowT3A, nowT3F, nowT1C, \
            diagnostic


def evaluate_Left_Down_Right_neighbor_3_energies():
    """
    only testing, not used on cluster
    SHOULD USE /_\because it captures best the corner-edge entanglement
    NOTE: the Hamiltonian is 'ikjl' now!!!
    """
    pass



def evaluate_9_nearest_neighbor_energies_in_3_envs():
    """
    only testing, not used on cluster
    """

    pass



def evaluate_LDR_3nn_and_6secondnn_energies():
    """
    NOTE: the Hamiltonian is 'ikjl' now!!!
    """

    pass

def evaluate_9nn_and_18secondnn_energies_in_3_envs():

    pass


def Adam_optimize_iPEPS(
                        ):
    """
    [only for testing, not used on cluster]
    Optimize the iPEPS tensors using Adam, with the CTMRG environment updated at every step.

    This is an alternative optimization loop that uses the Adam optimizer instead of L-BFGS. It may be more robust to local minima and can handle noisy gradients, but may require more iterations to converge.
    """
    pass



def L_BFGS_optimize_iPEPS(
                          ):
    """
    Optimize the iPEPS tensors using L-BFGS, with the CTMRG environment updated at every step.

    This is the main optimization loop of the code, where the iPEPS tensors are iteratively updated to minimize the energy. The CTMRG environment is recomputed at each step to ensure accurate energy evaluations.
    """
    pass






def build_heisenberg_H_ikjl(J: float = 1.0, d: int = 2) -> torch.Tensor:
    # use spin s=(d-1)/2 to build spin-s operators sx, sy, sz:
    spin = (d - 1) / 2
    spin = torch.tensor(spin, dtype=RDTYPE)
    Splus = torch.zeros((d, d), dtype=RDTYPE)
    Sminus = torch.zeros((d, d), dtype=RDTYPE)
    Sz = torch.zeros((d, d), dtype=RDTYPE)
    for i in range(d):
        m = float(spin - i)                   # numeric m-value (float)
        Sz[i, i] = m                          # diagonal Sz

        if i < d - 1:
            # coefficient for the transition m -> m-1
            CScoeff = torch.sqrt(spin * (spin + 1) - m * (m - 1))  # real CS-coeff/2
            Splus[i, i + 1] = CScoeff   # S+ raises m by 1
            Sminus[i + 1, i] = CScoeff  # S- lowers m by 1

    print(Splus,Sminus,Sz)

    SdotS = torch.tensor(oe.contract("ij,kl->ikjl", Splus, Sminus) * 0.5
                        +oe.contract("ij,kl->ikjl", Sminus, Splus) * 0.5
                        +oe.contract("ij,kl->ikjl", Sz, Sz), dtype=TENSORDTYPE)
    return J * SdotS









def energy_expectation_nearest_neighbor_3ebadcf_bonds(
                a,b,c,d,e,f, 
                Heb,Had,Hcf,
                Hea,Hac,Hce,Hbd,Hdf,Hfb,
                chi, D_bond, d_PHYS, 
                C21CD,C32EF,C13AB,T1F,T2A,T2B,T3C,T3D,T1E):
    
    T1F = T1F.reshape(chi,chi,D_bond,D_bond)
    T2A = T2A.reshape(chi,chi,D_bond,D_bond)
    T2B = T2B.reshape(chi,chi,D_bond,D_bond)
    T3C = T3C.reshape(chi,chi,D_bond,D_bond)
    T3D = T3D.reshape(chi,chi,D_bond,D_bond)
    T1E = T1E.reshape(chi,chi,D_bond,D_bond)

    open_E = oe.contract("YX,MYar,abci,rstj->MbsXctij", C21CD, T1F, e, e.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch").reshape(chi*D_bond*D_bond, chi*D_bond*D_bond, d_PHYS, d_PHYS)
    open_D = oe.contract("MYct,abci,rstj->YarMbsij", T3C, d, d.conj(), optimize=[(0,1),(0,1)], backend="torch").reshape(chi*D_bond*D_bond, chi*D_bond*D_bond, d_PHYS, d_PHYS)
    open_A = oe.contract("ZY,NZbs,abci,rstj->NctYarij", C32EF, T2B, a, a.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch").reshape(chi*D_bond*D_bond, chi*D_bond*D_bond, d_PHYS, d_PHYS)
    open_F = oe.contract("NZar,abci,rstj->ZbsNctij", T1E, f, f.conj(), optimize=[(0,1),(0,1)], backend="torch").reshape(chi*D_bond*D_bond, chi*D_bond*D_bond, d_PHYS, d_PHYS)
    open_C = oe.contract("XZ,LXct,abci,rstj->LarZbsij", C13AB, T3D, c, c.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch").reshape(chi*D_bond*D_bond, chi*D_bond*D_bond, d_PHYS, d_PHYS)
    open_B = oe.contract("LXbs,abci,rstj->XctLarij", T2A, b, b.conj(), optimize=[(0,1),(0,1)], backend="torch").reshape(chi*D_bond*D_bond, chi*D_bond*D_bond, d_PHYS, d_PHYS)
                              
    closed_E = oe.contract("MXii->MX", open_E, backend="torch")#.reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_D = oe.contract("YMii->YM", open_D, backend="torch")#.reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_A = oe.contract("NYii->NY", open_A, backend="torch")#.reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_F = oe.contract("ZNii->ZN", open_F, backend="torch")#.reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_C = oe.contract("LZii->LZ", open_C, backend="torch")#.reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_B = oe.contract("XLii->XL", open_B, backend="torch")#.reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)

    # H_AD = oe.contract("NctYarij,ijkl,YarMbskl->NctMbs", open_A, Had, open_D, optimize=[(0,1),(0,1)], backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    # H_CF = oe.contract("LarZbsij,ijkl,ZbsNctkl->LarNct", open_C, Hcf, open_F, optimize=[(0,1),(0,1)], backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    # H_EB = oe.contract("MbsXctij,ijkl,XctLarkl->MbsLar", open_E, Heb, open_B, optimize=[(0,1),(0,1)], backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)


    AD = torch.mm(closed_A, closed_D)
    CF = torch.mm(closed_C, closed_F)
    EB = torch.mm(closed_E, closed_B)

    # E_unnormed_AD = oe.contract("xy,yz,zx->", H_AD, EB, CF, backend="torch")
    # E_unnormed_CF = oe.contract("xy,yz,zx->", H_CF, AD, EB, backend="torch")
    # E_unnormed_EB = oe.contract("xy,yz,zx->", H_EB, CF, AD, backend="torch")


    # norm_1st_env = oe.contract("xy,yz,zx->", AD, EB, CF, backend="torch")
    
    # """ manually hermitianize!!!

    rho = oe.contract("NYij,YMkl->ikjlNM", open_A, open_D, backend="torch").reshape(d_PHYS*d_PHYS,d_PHYS*d_PHYS,chi*D_bond*D_bond,chi*D_bond*D_bond)
    rhoAD = oe.contract('IJxy,yz,zx->IJ', rho, EB, CF, backend="torch")
    #to_print = torch.norm(rhoAD - rhoAD.conj().T).item()/2.0/torch.norm(rhoAD).item()
    #if to_print > 1e-4: print("rhoAD anti-hermiticity: ", to_print)
    theOriTrace = torch.trace(rhoAD)
    rhoAD = (rhoAD + rhoAD.conj().T)/2.0
    # positive-semidefinite projection:
    eigvals, eigvecs = torch.linalg.eigh(rhoAD)
    #print("rhoAD eigvals before clipping: ", eigvals.detach().numpy())
    eigvals_clipped = torch.clamp(eigvals, min=0.0)
    rhoAD = (eigvecs * eigvals_clipped) @ eigvecs.conj().T
    theTrace = eigvals_clipped.sum().clamp(min=1e-30)  # real scalar → safe real division
    rhoAD = (rhoAD / theTrace).reshape(d_PHYS,d_PHYS,d_PHYS,d_PHYS)
    E_AD = oe.contract("ikjl,ijkl->", rhoAD, Had, backend="torch")
    #if to_print > 1e-4: print("E_AD = ", E_AD.item(), "trace rhoAD = ", theTrace.item(), "(ori:", theOriTrace.item(), ")")

    
    rho = oe.contract("NYij,YMkl->ikjlNM", open_C, open_F, backend="torch").reshape(d_PHYS*d_PHYS,d_PHYS*d_PHYS,chi*D_bond*D_bond,chi*D_bond*D_bond)
    rhoCF = oe.contract('IJxy,yz,zx->IJ', rho, AD, EB, backend="torch")
    #to_print = torch.norm(rhoCF - rhoCF.conj().T).item()/2.0/torch.norm(rhoCF).item()
    #if to_print > 1e-4: print("rhoCF anti-hermiticity: ", to_print)
    theOriTrace = torch.trace(rhoCF)
    rhoCF = (rhoCF + rhoCF.conj().T)/2.0
    # positive-semidefinite projection:
    eigvals, eigvecs = torch.linalg.eigh(rhoCF)
    #print("rhoCF eigvals before clipping: ", eigvals.detach().numpy())
    eigvals_clipped = torch.clamp(eigvals, min=0.0)
    rhoCF = (eigvecs * eigvals_clipped) @ eigvecs.conj().T  # PSD reconstruction (was accidentally commented out)
    theTrace = eigvals_clipped.sum().clamp(min=1e-30)  # real scalar → safe real division
    rhoCF = (rhoCF / theTrace).reshape(d_PHYS,d_PHYS,d_PHYS,d_PHYS)
    E_CF = oe.contract("ikjl,ijkl->", rhoCF, Hcf, backend="torch")
    #if to_print > 1e-4: print("E_CF = ", E_CF.item(), "trace rhoCF = ", theTrace.item(), "(ori:", theOriTrace.item(), ")")


    rho = oe.contract("NYij,YMkl->ikjlNM", open_E, open_B, backend="torch").reshape(d_PHYS*d_PHYS,d_PHYS*d_PHYS,chi*D_bond*D_bond,chi*D_bond*D_bond)
    rhoEB = oe.contract('IJxy,yz,zx->IJ', rho, CF, AD, backend="torch")
    #to_print = torch.norm(rhoEB - rhoEB.conj().T).item()/2.0/torch.norm(rhoEB).item()
    #if to_print > 1e-4: print("rhoEB anti-hermiticity: ", to_print)
    theOriTrace = torch.trace(rhoEB)
    rhoEB = (rhoEB + rhoEB.conj().T)/2.0
    # positive-semidefinite projection:
    eigvals, eigvecs = torch.linalg.eigh(rhoEB)
    #print("rhoEB eigvals before clipping: ", eigvals.detach().numpy())
    eigvals_clipped = torch.clamp(eigvals, min=0.0)
    rhoEB = (eigvecs * eigvals_clipped) @ eigvecs.conj().T
    theTrace = eigvals_clipped.sum().clamp(min=1e-30)  # real scalar → safe real division
    rhoEB = (rhoEB / theTrace).reshape(d_PHYS,d_PHYS,d_PHYS,d_PHYS)
    E_EB = oe.contract("ikjl,ijkl->", rhoEB, Heb, backend="torch")
    #if to_print > 1e-4: print("E_EB = ", E_EB.item(), "trace rhoEB = ", theTrace.item(), "(ori:", theOriTrace.item(), ")")


    # """
    rho = oe.contract("",
                      open_E, 
                      closed_D, 
                      open_A.reshape(chi*D_bond*D_bond, chi*D_bond*D_bond, d_PHYS, d_PHYS))
    
    
    # compare_energy = (torch.abs(E_unnormed_AD)+torch.abs(E_unnormed_CF)+torch.abs(E_unnormed_EB)) / torch.abs(norm_1st_env)
    # print("E_unnormed_AD = ", E_unnormed_AD.real.item(),"+i*", E_unnormed_AD.imag.item())
    # print("E_unnormed_CF = ", E_unnormed_CF.real.item(),"+i*", E_unnormed_CF.imag.item())
    # print("E_unnormed_EB = ", E_unnormed_EB.real.item(),"+i*", E_unnormed_EB.imag.item())
    # print("norm_1st_env = ", norm_1st_env.real.item(),"+i*", norm_1st_env.imag.item())
    # print("energyNearestNeighbor_3_bonds = ", energyNearestNeighbor_3_bonds.item())
    # print("compare_energy = ", compare_energy.item())
    
    return torch.real(E_AD + E_CF + E_EB)


def energy_expectation_nearest_neighbor_3afcbed_bonds(a,b,c,d,e,f,
                                                      Haf,Hcb,Hed, 
                                                      Hac,Hce,Hea,Hfb,Hbd,Hdf,
                chi, D_bond, d_PHYS, 
                C21EB, C32AD,C13CF,T1D,T2C,T2F,T3E,T3B,T1A):

    T1D = T1D.reshape(chi,chi,D_bond,D_bond)
    T2C = T2C.reshape(chi,chi,D_bond,D_bond)
    T2F = T2F.reshape(chi,chi,D_bond,D_bond)
    T3E = T3E.reshape(chi,chi,D_bond,D_bond)
    T3B = T3B.reshape(chi,chi,D_bond,D_bond)
    T1A = T1A.reshape(chi,chi,D_bond,D_bond)

    open_A = oe.contract("YX,MYar,abci,rstj->MbsXctij", C21EB, T1D, a, a.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch").reshape(chi*D_bond*D_bond, chi*D_bond*D_bond, d_PHYS, d_PHYS)
    open_B = oe.contract("MYct,abci,rstj->YarMbsij", T3E, b, b.conj(), optimize=[(0,1),(0,1)], backend="torch").reshape(chi*D_bond*D_bond, chi*D_bond*D_bond, d_PHYS, d_PHYS)
    open_C = oe.contract("ZY,NZbs,abci,rstj->NctYarij", C32AD, T2F, c, c.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch").reshape(chi*D_bond*D_bond, chi*D_bond*D_bond, d_PHYS, d_PHYS)
    open_D = oe.contract("NZar,abci,rstj->ZbsNctij", T1A, d, d.conj(), optimize=[(0,1),(0,1)], backend="torch").reshape(chi*D_bond*D_bond, chi*D_bond*D_bond, d_PHYS, d_PHYS)
    open_E = oe.contract("XZ,LXct,abci,rstj->LarZbsij", C13CF, T3B, e, e.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch").reshape(chi*D_bond*D_bond, chi*D_bond*D_bond, d_PHYS, d_PHYS)
    open_F = oe.contract("LXbs,abci,rstj->XctLarij", T2C, f, f.conj(), optimize=[(0,1),(0,1)], backend="torch").reshape(chi*D_bond*D_bond, chi*D_bond*D_bond, d_PHYS, d_PHYS)
                              
    closed_A = oe.contract("MXii->MX", open_A, backend="torch")#.reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_B = oe.contract("YMii->YM", open_B, backend="torch")#.reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_C = oe.contract("NYii->NY", open_C, backend="torch")#.reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_D = oe.contract("ZNii->ZN", open_D, backend="torch")#.reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_E = oe.contract("LZii->LZ", open_E, backend="torch")#.reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_F = oe.contract("XLii->XL", open_F, backend="torch")#.reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)

    #H_CB = oe.contract("NctYarij,ijkl,YarMbskl->NctMbs", open_C, Hcb, open_B, optimize=[(0,1),(0,1)], backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    #H_ED = oe.contract("LarZbsij,ijkl,ZbsNctkl->LarNct", open_E, Hed, open_D, optimize=[(0,1),(0,1)], backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    #H_AF = oe.contract("MbsXctij,ijkl,XctLarkl->MbsLar", open_A, Haf, open_F, optimize=[(0,1),(0,1)], backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)

    CB = torch.mm(closed_C, closed_B)
    ED = torch.mm(closed_E, closed_D)
    AF = torch.mm(closed_A, closed_F)



    rho = oe.contract("NYij,YMkl->ikjlNM", open_C, open_B, backend="torch").reshape(d_PHYS*d_PHYS,d_PHYS*d_PHYS,chi*D_bond*D_bond,chi*D_bond*D_bond)
    rhoCB = oe.contract('IJxy,yz,zx->IJ', rho, AF, ED, backend="torch")
    #to_print = torch.norm(rhoCB - rhoCB.conj().T).item()/2.0/torch.norm(rhoCB).item()
    #if to_print > 1e-4: print("rhoCB anti-hermiticity: ", to_print)
    theOriTrace = torch.trace(rhoCB)
    rhoCB = (rhoCB + rhoCB.conj().T)/2.0
    # positive-semidefinite projection:
    eigvals, eigvecs = torch.linalg.eigh(rhoCB)
    #print("rhoCB eigvals before clipping: ", eigvals.detach().numpy())
    eigvals_clipped = torch.clamp(eigvals, min=0.0)
    rhoCB = (eigvecs * eigvals_clipped) @ eigvecs.conj().T
    theTrace = eigvals_clipped.sum().clamp(min=1e-30)  # real scalar → safe real division
    rhoCB = (rhoCB / theTrace).reshape(d_PHYS,d_PHYS,d_PHYS,d_PHYS)
    E_CB = oe.contract("ikjl,ijkl->", rhoCB, Hcb, backend="torch")
    #if to_print > 1e-4: print("E_CB = ", E_CB.item(), "trace rhoCB = ", theTrace.item(), "(ori:", theOriTrace.item(), ")")

    
    rho = oe.contract("NYij,YMkl->ikjlNM", open_A, open_F, backend="torch").reshape(d_PHYS*d_PHYS,d_PHYS*d_PHYS,chi*D_bond*D_bond,chi*D_bond*D_bond)
    rhoAF = oe.contract('IJxy,yz,zx->IJ', rho, ED, CB, backend="torch")
    #to_print = torch.norm(rhoAF - rhoAF.conj().T).item()/2.0/torch.norm(rhoAF).item()
    #if to_print > 1e-4: print("rhoAF anti-hermiticity: ", to_print)
    theOriTrace = torch.trace(rhoAF)
    rhoAF = (rhoAF + rhoAF.conj().T)/2.0
    # positive-semidefinite projection:
    eigvals, eigvecs = torch.linalg.eigh(rhoAF)
    #print("rhoAF eigvals before clipping: ", eigvals.detach().numpy())
    eigvals_clipped = torch.clamp(eigvals, min=0.0)
    rhoAF = (eigvecs * eigvals_clipped) @ eigvecs.conj().T
    theTrace = eigvals_clipped.sum().clamp(min=1e-30)  # real scalar → safe real division
    rhoAF = (rhoAF / theTrace).reshape(d_PHYS,d_PHYS,d_PHYS,d_PHYS)
    E_AF = oe.contract("ikjl,ijkl->", rhoAF, Haf, backend="torch")
    #if to_print > 1e-4: print("E_AF = ", E_AF.item(), "trace rhoAF = ", theTrace.item(), "(ori:", theOriTrace.item(), ")")


    rho = oe.contract("NYij,YMkl->ikjlNM", open_E, open_D, backend="torch").reshape(d_PHYS*d_PHYS,d_PHYS*d_PHYS,chi*D_bond*D_bond,chi*D_bond*D_bond)
    rhoED = oe.contract('IJxy,yz,zx->IJ', rho, CB, AF, backend="torch")
    #to_print = torch.norm(rhoED - rhoED.conj().T).item()/2.0/torch.norm(rhoED).item()
    #if to_print > 1e-4: print("rhoED anti-hermiticity: ", to_print)
    theOriTrace = torch.trace(rhoED)
    rhoED = (rhoED + rhoED.conj().T)/2.0
    # positive-semidefinite projection:
    eigvals, eigvecs = torch.linalg.eigh(rhoED)
    #print("rhoED eigvals before clipping: ", eigvals.detach().numpy())
    eigvals_clipped = torch.clamp(eigvals, min=0.0)
    rhoED = (eigvecs * eigvals_clipped) @ eigvecs.conj().T
    theTrace = eigvals_clipped.sum().clamp(min=1e-30)  # real scalar → safe real division
    rhoED = (rhoED / theTrace).reshape(d_PHYS,d_PHYS,d_PHYS,d_PHYS)
    E_ED = oe.contract("ikjl,ijkl->", rhoED, Hed, backend="torch")
    #if to_print > 1e-4: print("E_ED = ", E_ED.item(), "trace rhoED = ", theTrace.item(), "(ori:", theOriTrace.item(), ")")


    
    return torch.real(E_AF + E_CB + E_ED)





def energy_expectation_nearest_neighbor_other_3_bonds(a,b,c,d,e,f, 
                                                      Hcd,Hef,Hab,
                                                      Hce,Hea,Hac,Hdf,Hfb,Hbd,
                                                      chi, D_bond, d_PHYS,
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
        torch.Tensor: Scalar energy per bond (float32).
    """
    T1B = T1B.reshape(chi,chi,D_bond,D_bond)
    T2E = T2E.reshape(chi,chi,D_bond,D_bond)
    T2D = T2D.reshape(chi,chi,D_bond,D_bond)
    T3A = T3A.reshape(chi,chi,D_bond,D_bond)
    T3F = T3F.reshape(chi,chi,D_bond,D_bond)
    T1C = T1C.reshape(chi,chi,D_bond,D_bond)

    open_C = oe.contract("YX,MYar,abci,rstj->MbsXctij", C21AF, T1B, c, c.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch").reshape(chi*D_bond*D_bond, chi*D_bond*D_bond, d_PHYS, d_PHYS)
    open_F = oe.contract("MYct,abci,rstj->YarMbsij", T3A, f, f.conj(), optimize=[(0,1),(0,1)], backend="torch").reshape(chi*D_bond*D_bond, chi*D_bond*D_bond, d_PHYS, d_PHYS)
    open_E = oe.contract("ZY,NZbs,abci,rstj->NctYarij", C32CB, T2D, e, e.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch").reshape(chi*D_bond*D_bond, chi*D_bond*D_bond, d_PHYS, d_PHYS)
    open_B = oe.contract("NZar,abci,rstj->ZbsNctij", T1C, b, b.conj(), optimize=[(0,1),(0,1)], backend="torch").reshape(chi*D_bond*D_bond, chi*D_bond*D_bond, d_PHYS, d_PHYS)
    open_A = oe.contract("XZ,LXct,abci,rstj->LarZbsij", C13ED, T3F, a, a.conj(), optimize=[(0,1),(0,1),(0,1)], backend="torch").reshape(chi*D_bond*D_bond, chi*D_bond*D_bond, d_PHYS, d_PHYS)
    open_D = oe.contract("LXbs,abci,rstj->XctLarij", T2E, d, d.conj(), optimize=[(0,1),(0,1)], backend="torch").reshape(chi*D_bond*D_bond, chi*D_bond*D_bond, d_PHYS, d_PHYS)
                              
    closed_C = oe.contract("MXii->MX", open_C, backend="torch")#.reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_F = oe.contract("YMii->YM", open_F, backend="torch")#.reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_E = oe.contract("NYii->NY", open_E, backend="torch")#.reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_B = oe.contract("ZNii->ZN", open_B, backend="torch")#.reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_A = oe.contract("LZii->LZ", open_A, backend="torch")#.reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    closed_D = oe.contract("XLii->XL", open_D, backend="torch")#.reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)

    #H_EF = oe.contract("NctYarij,ijkl,YarMbskl->NctMbs", open_E, Hef, open_F, optimize=[(0,1),(0,1)], backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    #H_AB = oe.contract("LarZbsij,ijkl,ZbsNctkl->LarNct", open_A, Hab, open_B, optimize=[(0,1),(0,1)], backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)
    #H_CD = oe.contract("MbsXctij,ijkl,XctLarkl->MbsLar", open_C, Hcd, open_D, optimize=[(0,1),(0,1)], backend="torch").reshape(chi*D_bond*D_bond,chi*D_bond*D_bond)

    EF = torch.mm(closed_E, closed_F)
    AB = torch.mm(closed_A, closed_B)
    CD = torch.mm(closed_C, closed_D)


    rho = oe.contract("NYij,YMkl->ikjlNM", open_E, open_F, backend="torch").reshape(d_PHYS*d_PHYS,d_PHYS*d_PHYS,chi*D_bond*D_bond,chi*D_bond*D_bond)
    rhoEF = oe.contract('IJxy,yz,zx->IJ', rho, CD, AB, backend="torch")
    #to_print = torch.norm(rhoEF - rhoEF.conj().T).item()/2.0/torch.norm(rhoEF).item()
    #if to_print > 1e-4: print("rhoEF anti-hermiticity: ", to_print)
    theOriTrace = torch.trace(rhoEF)
    rhoEF = (rhoEF + rhoEF.conj().T)/2.0
    # positive-semidefinite projection:
    eigvals, eigvecs = torch.linalg.eigh(rhoEF)
    #print("rhoEF eigvals before clipping: ", eigvals.detach().numpy())
    eigvals_clipped = torch.clamp(eigvals, min=0.0)
    rhoEF = (eigvecs * eigvals_clipped) @ eigvecs.conj().T
    theTrace = eigvals_clipped.sum().clamp(min=1e-30)  # real scalar → safe real division
    rhoEF = (rhoEF / theTrace).reshape(d_PHYS,d_PHYS,d_PHYS,d_PHYS)
    E_EF = oe.contract("ikjl,ijkl->", rhoEF, Hef, backend="torch")
    #if to_print > 1e-4: print("E_EF = ", E_EF.item(), "trace rhoEF = ", theTrace.item(), "(ori:", theOriTrace.item(), ")")

    
    rho = oe.contract("NYij,YMkl->ikjlNM", open_A, open_B, backend="torch").reshape(d_PHYS*d_PHYS,d_PHYS*d_PHYS,chi*D_bond*D_bond,chi*D_bond*D_bond)
    rhoAB = oe.contract('IJxy,yz,zx->IJ', rho, EF, CD, backend="torch")
    #to_print = torch.norm(rhoAB - rhoAB.conj().T).item()/2.0/torch.norm(rhoAB).item()
    #if to_print > 1e-4: print("rhoAB anti-hermiticity: ", to_print)
    theOriTrace = torch.trace(rhoAB)
    rhoAB = (rhoAB + rhoAB.conj().T)/2.0
    # positive-semidefinite projection:
    eigvals, eigvecs = torch.linalg.eigh(rhoAB)
    #print("rhoAB eigvals before clipping: ", eigvals.detach().numpy())
    eigvals_clipped = torch.clamp(eigvals, min=0.0)
    rhoAB = (eigvecs * eigvals_clipped) @ eigvecs.conj().T
    theTrace = eigvals_clipped.sum().clamp(min=1e-30)  # real scalar → safe real division
    rhoAB = (rhoAB / theTrace).reshape(d_PHYS,d_PHYS,d_PHYS,d_PHYS)
    E_AB = oe.contract("ikjl,ijkl->", rhoAB, Hab, backend="torch")
    #if to_print > 1e-4: print("E_AB = ", E_AB.item(), "trace rhoAB = ", theTrace.item(), "(ori:", theOriTrace.item(), ")")


    rho = oe.contract("NYij,YMkl->ikjlNM", open_C, open_D, backend="torch").reshape(d_PHYS*d_PHYS,d_PHYS*d_PHYS,chi*D_bond*D_bond,chi*D_bond*D_bond)
    rhoCD = oe.contract('IJxy,yz,zx->IJ', rho, AB, EF, backend="torch")
    #to_print = torch.norm(rhoCD - rhoCD.conj().T).item()/2.0/torch.norm(rhoCD).item()
    #if to_print > 1e-4: print("rhoCD anti-hermiticity: ", to_print)
    theOriTrace = torch.trace(rhoCD)
    rhoCD = (rhoCD + rhoCD.conj().T)/2.0
    # positive-semidefinite projection:
    eigvals, eigvecs = torch.linalg.eigh(rhoCD)
    #print("rhoCD eigvals before clipping: ", eigvals.detach().numpy())
    eigvals_clipped = torch.clamp(eigvals, min=0.0)
    rhoCD = (eigvecs * eigvals_clipped) @ eigvecs.conj().T
    theTrace = eigvals_clipped.sum().clamp(min=1e-30)  # real scalar → safe real division
    rhoCD = (rhoCD / theTrace).reshape(d_PHYS,d_PHYS,d_PHYS,d_PHYS)
    E_CD = oe.contract("ikjl,ijkl->", rhoCD, Hcd, backend="torch")
    #if to_print > 1e-4: print("E_CD = ", E_CD.item(), "trace rhoCD = ", theTrace.item(), "(ori:", theOriTrace.item(), ")")
    

    return torch.real(E_EF + E_AB + E_CD)


