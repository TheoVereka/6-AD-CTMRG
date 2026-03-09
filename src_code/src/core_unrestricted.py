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
    norm = torch.norm(tensor,p=2)
    if norm > 0:
        return tensor / norm
    else:
        return tensor



def initialize_abcdef(initialize_way:str, D_bond:int, d_PHYS:int, noise_scale:float):

    if initialize_way == 'random' :

        a = torch.randn(D_bond, D_bond, D_bond, d_PHYS, dtype=torch.complex64)
        b = torch.randn(D_bond, D_bond, D_bond, d_PHYS, dtype=torch.complex64)
        c = torch.randn(D_bond, D_bond, D_bond, d_PHYS, dtype=torch.complex64)
        d = torch.randn(D_bond, D_bond, D_bond, d_PHYS, dtype=torch.complex64)
        e = torch.randn(D_bond, D_bond, D_bond, d_PHYS, dtype=torch.complex64)
        f = torch.randn(D_bond, D_bond, D_bond, d_PHYS, dtype=torch.complex64)

    elif initialize_way == 'product' : # product state but always with small noise

        pass

    elif initialize_way == 'singlet' : # Mz=0 sector's singlet state representable as PEPS

        pass

    else :

        raise ValueError(f"Invalid initialize_way: {initialize_way}")
    
    return a,b,c,d,e,f




def abcdef_to_ABCDEF(a,b,c,d,e,f, D_squared:int):

    A = oe.contract("uvwφ,xyzφ->uxvywz", a,a.conj(), optimize=[(0,1)], backend='pytorch')
    A = normalize_tensor(A)
    A = A.reshape(D_squared, D_squared)

    B = oe.contract("uvwφ,xyzφ->uxvywz", b,b.conj(), optimize=[(0,1)], backend='pytorch')
    B = normalize_tensor(B)
    B = B.reshape(D_squared, D_squared)

    C = oe.contract("uvwφ,xyzφ->uxvywz", c,c.conj(), optimize=[(0,1)], backend='pytorch')
    C = normalize_tensor(C)
    C = C.reshape(D_squared, D_squared)

    D = oe.contract("uvwφ,xyzφ->uxvywz", d,d.conj(), optimize=[(0,1)], backend='pytorch')
    D = normalize_tensor(D)
    D = D.reshape(D_squared, D_squared)

    E = oe.contract("uvwφ,xyzφ->uxvywz", e,e.conj(), optimize=[(0,1)], backend='pytorch')
    E = normalize_tensor(E)
    E = E.reshape(D_squared, D_squared)

    F = oe.contract("uvwφ,xyzφ->uxvywz", f,f.conj(), optimize=[(0,1)], backend='pytorch')
    F = normalize_tensor(F)
    F = F.reshape(D_squared, D_squared)

    return A,B,C,D,E,F




def trunc_rhoCCC(matC21, matC32, matC13, chi, D_squared):

    rho32 = oe.contract("UZ,ZY,YV->UV",
                               matC13,matC32,matC21,
                               optimize=[(0,1),(0,1)],
                               backend='pytorch')
    
    U3, sv32, V2 = torch.linalg.svd(rho32,driver='gesvd')

    U3 = U3[:,:chi].conjugate()
    V2 = V2[:chi,:].conjugate()
    
    rho13 = oe.contract("UX,XZ,ZV->UV",
                               matC21,matC13,matC32,
                               optimize=[(0,1),(0,1)],
                               backend='pytorch')
    
    U1, sv13, V3 = torch.linalg.svd(rho13,driver='gesvd')

    U1 = U1[:,:chi].conjugate() #conjugate transpose
    V3 = V3[:chi,:].conjugate()
    

    rho21 = oe.contract("UY,YX,XV->UV",
                               matC32,matC21,matC13,
                               optimize=[(0,1),(0,1)],
                               backend='pytorch')
    
    U2, sv21, V1 = torch.linalg.svd(rho21,driver='gesvd')
    
    U2 = U2[:,:chi].conjugate()
    V1 = V1[:chi,:].conjugate()

    C21 = oe.contract("Yy,YX,xX->yx",
                               U1,matC21,V2,
                               optimize=[(0,1),(0,1)],
                               backend='pytorch')

    C32 = oe.contract("Zz,ZY,yY->zy",
                               U2,matC32,V3,
                               optimize=[(0,1),(0,1)],
                               backend='pytorch')

    C13 = oe.contract("Xx,XZ,zZ->xz",
                               U3,matC13,V1,
                               optimize=[(0,1),(0,1)],
                               backend='pytorch')
    
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





def initialize_environmentCTs_1(A,B,C,D,E,F, chi, D_squared):

    C21AF = oe.contract("oyg,xog->yx", A,F, optimize=[(0,1)], backend='pytorch')
    C32CB = oe.contract("aoz,ayo->zy", C,B, optimize=[(0,1)], backend='pytorch')
    C13ED = oe.contract("xbo,obz->xz", E,D, optimize=[(0,1)], backend='pytorch')

    T2ET3F = oe.contract("ubi,jki,jlk,vlg->ubvg", E,B,C,F, optimize=[(1,2),(0,1),(0,1)], backend='pytorch')
    T3AT1B = oe.contract("iug,ijk,kjl,avl->ugva", A,D,E,B, optimize=[(1,2),(0,1),(0,1)], backend='pytorch')
    T1CT2D = oe.contract("aiu,kij,lkj,lbv->uavb", C,F,A,D, optimize=[(1,2),(0,1),(0,1)], backend='pytorch')

    T2ET3F = T2ET3F.reshape(D_squared*D_squared, D_squared*D_squared)
    T3AT1B = T3AT1B.reshape(D_squared*D_squared, D_squared*D_squared)
    T1CT2D = T1CT2D.reshape(D_squared*D_squared, D_squared*D_squared)

    U2E, sv23, Vdag3F = torch.linalg.svd(T2ET3F,driver='gesvd')
    U3A, sv31, Vdag1B = torch.linalg.svd(T3AT1B,driver='gesvd')
    U1C, sv12, Vdag2D = torch.linalg.svd(T1CT2D,driver='gesvd')

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
    
    C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E = update_environmentCTs_3to1(
        C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C, A,B,C,D,E,F, chi, D_squared)

    return C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E




def check_env_convergence(lastC21CD, lastC32EF, lastC13AB, lastT1F, lastT2A, lastT2B, lastT3C, lastT3D, lastT1E, 
                          nowC21CD, nowC32EF, nowC13AB, nowT1F, nowT2A, nowT2B, nowT3C, nowT3D, nowT1E, 
                          lastC21EB, lastC32AD, lastC13CF, lastT1D, lastT2C, lastT2F, lastT3E, lastT3B, lastT1A, 
                          nowC21EB, nowC32AD, nowC13CF, nowT1D, nowT2C, nowT2F, nowT3E, nowT3B, nowT1A, 
                          lastC21AF, lastC32CB, lastC13ED, lastT1B, lastT2E, lastT2D, lastT3A, lastT3F, lastT1C, 
                          nowC21AF, nowC32CB, nowC13ED, nowT1B, nowT2E, nowT2D, nowT3A, nowT3F, nowT1C,
                          env_conv_threshold):
    
    if lastT1C is None:
        return False
    
    total_numel = lastC21CD.numel() + lastC32EF.numel() + lastC13AB.numel() + \
                  lastT1F.numel()  + lastT2A.numel()  + lastT2B.numel()  + \
                  lastT3C.numel()  + lastT3D.numel()  + lastT1E.numel()  + \
                  lastC21EB.numel() + lastC32AD.numel() + lastC13CF.numel() + \
                  lastT1D.numel()   + lastT2C.numel()   + lastT2F.numel()   + \
                  lastT3E.numel()   + lastT3B.numel()   + lastT1A.numel()   + \
                  lastC21AF.numel() + lastC32CB.numel() + lastC13ED.numel() + \
                  lastT1B.numel()   + lastT2E.numel()   + lastT2D.numel()   + \
                  lastT3A.numel()   + lastT3F.numel()   + lastT1C.numel()
    

    total_sq =  torch.sum(torch.abs(lastC21CD - nowC21CD) ** 2) + \
                torch.sum(torch.abs(lastC32EF - nowC32EF) ** 2) + \
                torch.sum(torch.abs(lastC13AB - nowC13AB) ** 2) + \
                torch.sum(torch.abs(lastT1F   - nowT1F  ) ** 2) + \
                torch.sum(torch.abs(lastT2A   - nowT2A  ) ** 2) + \
                torch.sum(torch.abs(lastT2B   - nowT2B  ) ** 2) + \
                torch.sum(torch.abs(lastT3C   - nowT3C  ) ** 2) + \
                torch.sum(torch.abs(lastT3D   - nowT3D  ) ** 2) + \
                torch.sum(torch.abs(lastT1E   - nowT1E  ) ** 2) + \
                torch.sum(torch.abs(lastC21EB - nowC21EB) ** 2) + \
                torch.sum(torch.abs(lastC32AD - nowC32AD) ** 2) + \
                torch.sum(torch.abs(lastC13CF - nowC13CF) ** 2) + \
                torch.sum(torch.abs(lastT1D   - nowT1D  ) ** 2) + \
                torch.sum(torch.abs(lastT2C   - nowT2C  ) ** 2) + \
                torch.sum(torch.abs(lastT2F   - nowT2F  ) ** 2) + \
                torch.sum(torch.abs(lastT3E   - nowT3E  ) ** 2) + \
                torch.sum(torch.abs(lastT3B   - nowT3B  ) ** 2) + \
                torch.sum(torch.abs(lastT1A   - nowT1A  ) ** 2) + \
                torch.sum(torch.abs(lastC21AF - nowC21AF) ** 2) + \
                torch.sum(torch.abs(lastC32CB - nowC32CB) ** 2) + \
                torch.sum(torch.abs(lastC13ED - nowC13ED) ** 2) + \
                torch.sum(torch.abs(lastT1B   - nowT1B  ) ** 2) + \
                torch.sum(torch.abs(lastT2E   - nowT2E  ) ** 2) + \
                torch.sum(torch.abs(lastT2D   - nowT2D  ) ** 2) + \
                torch.sum(torch.abs(lastT3A   - nowT3A  ) ** 2) + \
                torch.sum(torch.abs(lastT3F   - nowT3F  ) ** 2) + \
                torch.sum(torch.abs(lastT1C   - nowT1C  ) ** 2)

    rms_diff = torch.sqrt(total_sq) / torch.sqrt(total_numel)
    return bool(rms_diff.item() < env_conv_threshold)



def update_environmentCTs_1to2(C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E, A,B,C,D,E,F, chi, D_squared):

    matC21EB = oe.contract("YX,MYa,LXβ,amg,lbg->MmLl",
                           C21CD,T1F,T2A,E,B,
                           optimize=[(0,2),(0,3),(1,2),(0,1)],
                           backend='pytorch')
    
    matC21EB = matC21EB.view(chi*D_squared,chi*D_squared)
    
    matC32AD = oe.contract("ZY,NZβ,MYg,abn,amg->NnMm",
                           C32EF,T2B,T3C,A,D,
                           optimize=[(0,2),(0,3),(1,2),(0,1)],
                           backend='pytorch')
    
    matC32AD = matC32AD.view(chi*D_squared,chi*D_squared)
    
    matC13CF = oe.contract("XZ,LXg,NZa,lbg,abn->LlNn",
                           C13AB,T3D,T1E,C,F,
                           optimize=[(0,2),(0,3),(1,2),(0,1)],
                           backend='pytorch')
    
    matC13CF = matC13CF.view(chi*D_squared,chi*D_squared)

    V2A, C21EB, U1F, V3C, C32AD, U2B, V1E, C13CF, U3D = trunc_rhoCCC(
                        matC21EB, matC32AD, matC13CF, chi, D_squared)

    T3E = normalize_tensor(oe.contract("OYa,abg,ObM->MYg",
                        T1F,E,U1F,
                        optimize=[(0,1),(0,1)],
                        backend='pytorch'))

    T3B = normalize_tensor(oe.contract("OXb,abg,LOa->LXg",
                        T2A,B,V2A,
                        optimize=[(0,1),(0,1)],
                        backend='pytorch'))
    
    T1A = normalize_tensor(oe.contract("OZb,abg,OgN->NZa",
                        T2B,A,U2B,
                        optimize=[(0,1),(0,1)],
                        backend='pytorch'))
    
    T1D = normalize_tensor(oe.contract("OYg,abg,MOb->MYa",
                        T3C,D,V3C,
                        optimize=[(0,1),(0,1)],
                        backend='pytorch'))
    
    T2C = normalize_tensor(oe.contract("OXg,abg,OaL->LXb",
                        T3D,C,U3D,
                        optimize=[(0,1),(0,1)],
                        backend='pytorch'))
    
    T2F = normalize_tensor(oe.contract("OZa,abg,NOg->NZb",
                        T1E,F,V1E,
                        optimize=[(0,1),(0,1)],
                        backend='pytorch'))

    return C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A




def update_environmentCTs_2to3(C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A, A,B,C,D,E,F, chi, D_squared):

    matC21AF = oe.contract("YX,MYa,LXβ,amg,lbg->MmLl",
                           C21EB,T1D,T2C,A,F,
                           optimize=[(0,2),(0,3),(1,2),(0,1)],
                           backend='pytorch')
    
    matC21AF = matC21AF.view(chi*D_squared,chi*D_squared)
    
    matC32CB = oe.contract("ZY,NZβ,MYg,abn,amg->NnMm",
                           C32AD,T2F,T3E,C,B,
                           optimize=[(0,2),(0,3),(1,2),(0,1)],
                           backend='pytorch')
    
    matC32CB = matC32CB.view(chi*D_squared,chi*D_squared)
    
    matC13ED = oe.contract("XZ,LXg,NZa,lbg,abn->LlNn",
                           C13CF,T3B,T1A,E,D,
                           optimize=[(0,2),(0,3),(1,2),(0,1)],
                           backend='pytorch')
    
    matC13ED = matC13ED.view(chi*D_squared,chi*D_squared)

    V2C, C21AF, U1D, V3E, C32CB, U2F, V1A, C13ED, U3B = trunc_rhoCCC(
                        matC21AF, matC32CB, matC13ED, chi, D_squared)

    T3A = normalize_tensor(oe.contract("OYa,abg,ObM->MYg",
                        T1D,A,U1D,
                        optimize=[(0,1),(0,1)],
                        backend='pytorch'))

    T3F = normalize_tensor(oe.contract("OXb,abg,LOa->LXg",
                        T2C,F,V2C,
                        optimize=[(0,1),(0,1)],
                        backend='pytorch'))
    
    T1C = normalize_tensor(oe.contract("OZb,abg,OgN->NZa",
                        T2F,C,U2F,
                        optimize=[(0,1),(0,1)],
                        backend='pytorch'))
    
    T1B = normalize_tensor(oe.contract("OYg,abg,MOb->MYa",
                        T3E,B,V3E,
                        optimize=[(0,1),(0,1)],
                        backend='pytorch'))
    
    T2E = normalize_tensor(oe.contract("OXg,abg,OaL->LXb",
                        T3B,E,U3B,
                        optimize=[(0,1),(0,1)],
                        backend='pytorch'))
    
    T2D = normalize_tensor(oe.contract("OZa,abg,NOg->NZb",
                        T1A,D,V1A,
                        optimize=[(0,1),(0,1)],
                        backend='pytorch'))

    return C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C




def update_environmentCTs_3to1(C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C, A,B,C,D,E,F, chi, D_squared):

    matC21CD = oe.contract("YX,MYa,LXβ,amg,lbg->MmLl",
                           C21AF,T1B,T2E,C,D,
                           optimize=[(0,2),(0,3),(1,2),(0,1)],
                           backend='pytorch')
    
    matC21CD = matC21CD.view(chi*D_squared,chi*D_squared)
    
    matC32EF = oe.contract("ZY,NZβ,MYg,abn,amg->NnMm",
                           C32CB,T2D,T3A,E,F,
                           optimize=[(0,2),(0,3),(1,2),(0,1)],
                           backend='pytorch')
    
    matC32EF = matC32EF.view(chi*D_squared,chi*D_squared)
    
    matC13AB = oe.contract("XZ,LXg,NZa,lbg,abn->LlNn",
                           C13ED,T3F,T1C,A,B,
                           optimize=[(0,2),(0,3),(1,2),(0,1)],
                           backend='pytorch')
    
    matC13AB = matC13AB.view(chi*D_squared,chi*D_squared)

    V2E, C21CD, U1B, V3A, C32EF, U2D, V1C, C13AB, U3F = trunc_rhoCCC(
                        matC21CD, matC32EF, matC13AB, chi, D_squared)

    T3C = normalize_tensor(oe.contract("OYa,abg,ObM->MYg",
                        T1B,C,U1B,
                        optimize=[(0,1),(0,1)],
                        backend='pytorch'))

    T3D = normalize_tensor(oe.contract("OXb,abg,LOa->LXg",
                        T2E,D,V2E,
                        optimize=[(0,1),(0,1)],
                        backend='pytorch'))
    
    T1E = normalize_tensor(oe.contract("OZb,abg,OgN->NZa",
                        T2D,E,U2D,
                        optimize=[(0,1),(0,1)],
                        backend='pytorch'))
    
    T1F = normalize_tensor(oe.contract("OYg,abg,MOb->MYa",
                        T3A,F,V3A,
                        optimize=[(0,1),(0,1)],
                        backend='pytorch'))
    
    T2A = normalize_tensor(oe.contract("OXg,abg,OaL->LXb",
                        T3F,A,U3F,
                        optimize=[(0,1),(0,1)],
                        backend='pytorch'))
    
    T2B = normalize_tensor(oe.contract("OZa,abg,NOg->NZb",
                        T1C,B,V1C,
                        optimize=[(0,1),(0,1)],
                        backend='pytorch'))

    return C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E




def CTMRG_from_init_to_stop(A,B,C,D,E,F,
                            chi: int, 
                            D_squared: int,
                            a_third_max_iterations: int, 
                            env_conv_threshold: float) :
    """
    This function performs the CTMRG algorithm from the initial state to the stopping criterion.

    Args:
        A, B, C, D, E, F (torch.Tensor): The 6 local tensors.
        max_iterations (int): The maximum number of iterations to perform.
        D_squared (int): The square of the bond dimension of the local state projector a~f.
        chi (int): The nominal desired bond dimension for the transfer tensors.
        env_conv_threshold (float): The threshold for environment convergence.

    Returns:
        A tuple containing the final environment corner and edge transfer tensors, and the number of iterations performed.
    """
    # Initialize the environment corner and edge transfer tensors

    lastC21CD, lastC32EF, lastC13AB, lastT1F, lastT2A, lastT2B, lastT3C, lastT3D, lastT1E = None, None, None, None, None, None, None, None, None
    lastC21EB, lastC32AD, lastC13CF, lastT1D, lastT2C, lastT2F, lastT3E, lastT3B, lastT1A = None, None, None, None, None, None, None, None, None
    lastC21AF, lastC32CB, lastC13ED, lastT1B, lastT2E, lastT2D, lastT3A, lastT3F, lastT1C = None, None, None, None, None, None, None, None, None
    nowC21CD, nowC32EF, nowC13AB, nowT1F, nowT2A, nowT2B, nowT3C, nowT3D, nowT1E = initialize_environmentCTs_1(A,B,C,D,E,F, chi, D_squared)
    nowC21EB, nowC32AD, nowC13CF, nowT1D, nowT2C, nowT2F, nowT3E, nowT3B, nowT1A = None, None, None, None, None, None, None, None, None
    nowC21AF, nowC32CB, nowC13ED, nowT1B, nowT2E, nowT2D, nowT3A, nowT3F, nowT1C = None, None, None, None, None, None, None, None, None

    # Perform the CTMRG iterations until convergence
    for iteration in range(a_third_max_iterations):

        if check_env_convergence(lastC21CD, lastC32EF, lastC13AB, lastT1F, lastT2A, lastT2B, lastT3C, lastT3D, lastT1E, 
                                 nowC21CD, nowC32EF, nowC13AB, nowT1F, nowT2A, nowT2B, nowT3C, nowT3D, nowT1E, 
                                 lastC21EB, lastC32AD, lastC13CF, lastT1D, lastT2C, lastT2F, lastT3E, lastT3B, lastT1A, 
                                 nowC21EB, nowC32AD, nowC13CF, nowT1D, nowT2C, nowT2F, nowT3E, nowT3B, nowT1A, 
                                 lastC21AF, lastC32CB, lastC13ED, lastT1B, lastT2E, lastT2D, lastT3A, lastT3F, lastT1C, 
                                 nowC21AF, nowC32CB, nowC13ED, nowT1B, nowT2E, nowT2D, nowT3A, nowT3F, nowT1C, 
                                 env_conv_threshold):
            print(f"Convergence achieved at iteration {iteration}.")
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
            nowC21AF, nowC32CB, nowC13ED, nowT1B, nowT2E, nowT2D, nowT3A, nowT3F, nowT1C




def Loss_as_energy_expectation(a,b,c,d,e,f, a_lot_of_bond_Hamiltonians, chi, D_bond, d_PHYS,
                              C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E,
                              C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A,
                              C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C):
    pass
    # return bra_H_ket



def energy_expectation_nearest_neighbor_6_bonds(a,b,c,d,e,f, 
                                                Hab,Hbc,Hcd,Hde,Hef,Hfa, # (d_PHYS * d_PHYS, d_PHYS * d_PHYS) matrices
                                                chi, D_bond, d_PHYS,
                                                C21CD,C32EF,C13AB,T1F,T2A,T2B,T3C,T3D,T1E):
    pass
                              




def optmization_iPEPS(Hab,Hbc,Hcd,Hde,Hef,Hfa,
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
                      opt_tolerance_change: float = 1e-8):
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
        a, b, c, d, e, f  —  optimised site tensors (still require_grad=True)
    """
    D_squared = D_bond ** 2

    # ── 1. Initialise site tensors ────────────────────────────────────────────
    a, b, c, d, e, f = initialize_abcdef(a2f_initialize_way, D_bond, d_PHYS, a2f_noise_scale)
    a.requires_grad_(True)
    b.requires_grad_(True)
    c.requires_grad_(True)
    d.requires_grad_(True)
    e.requires_grad_(True)
    f.requires_grad_(True)

    # ── 2. Set up L-BFGS ─────────────────────────────────────────────────────
    optimizer = torch.optim.LBFGS(
        [a, b, c, d, e, f],
        lr=lbfgs_lr,
        max_iter=lbfgs_max_iter,
        tolerance_grad=opt_tolerance_grad,
        tolerance_change=opt_tolerance_change,
        history_size=lbfgs_history,
        line_search_fn='strong_wolfe',
    )

    prev_loss = None

    # ── 3. Outer optimisation loop ────────────────────────────────────────────
    for opt_step in range(max_opt_steps):

        # 3a. Rebuild double-layer tensors and converge the CTMRG environment.
        #     No gradients needed here — the environment is treated as a fixed
        #     external field during the L-BFGS line search.
        with torch.no_grad():
            A, B, C, D, E, F = abcdef_to_ABCDEF(a, b, c, d, e, f, D_squared)
            
            (C21CD, C32EF, C13AB, T1F,  T2A,  T2B,  T3C,  T3D,  T1E,
             C21EB, C32AD, C13CF, T1D,  T2C,  T2F,  T3E,  T3B,  T1A,
             C21AF, C32CB, C13ED, T1B,  T2E,  T2D,  T3A,  T3F,  T1C) = \
            CTMRG_from_init_to_stop(A, B, C, D, E, F, chi, D_squared,
                a_third_max_steps_CTMRG, CTM_env_conv_threshold)

        # 3b. L-BFGS closure: energy evaluation + backward through a..f only.
        def closure():
            optimizer.zero_grad()
            loss = energy_expectation_nearest_neighbor_6_bonds(
                a, b, c, d, e, f,
                Hab, Hbc, Hcd, Hde, Hef, Hfa,
                chi, D_bond, d_PHYS,
                C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E)
            loss.backward()
            return loss

        # 3c. Run L-BFGS sub-iterations (strong-Wolfe line search built in).
        loss_val = optimizer.step(closure)

        loss_item = loss_val.item()
        delta = (loss_item - prev_loss) if prev_loss is not None else float('inf')
        print(f"  opt {opt_step:4d}  loss = {loss_item:+.10f}  Δloss = {delta:.3e}")

        # 3d. Outer convergence check.
        if prev_loss is not None and abs(delta) < opt_conv_threshold:
            print(f"  Outer convergence achieved at step {opt_step} (Δloss={delta:.3e}).")
            break
        prev_loss = loss_item

    return a, b, c, d, e, f, loss_item



def check_optimized_iPEPS(a,b,c,d,e,f, old_loss, 
                          Hab,Hbc,Hcd,Hde,Hef,Hfa,
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
        C21AF, C32CB, C13ED, T1B,  T2E,  T2D,  T3A,  T3F,  T1C) = \
        CTMRG_from_init_to_stop(A, B, C, D, E, F, new_chi, D_squared,
                a_third_max_steps_CTMRG, CTM_env_conv_threshold)
        
        new_loss_under_new_chi = energy_expectation_nearest_neighbor_6_bonds(
            a,b,c,d,e,f,Hab,Hbc,Hcd,Hde,Hef,Hfa, new_chi, D_bond, d_PHYS, 
            C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E).item()
        
        delta_loss = new_loss_under_new_chi - old_loss
        print(f"  Check optimized iPEPS with chi={new_chi}: loss = {new_loss_under_new_chi:+.10f}  Δloss = {delta_loss:.3e}")
        return bool(abs(delta_loss) < delta_loss_threshold)
    



