
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




