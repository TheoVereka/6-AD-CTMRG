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

def initialize_environmentCTs_1(A,B,C,D,E,F, D_squared, chi):
    
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



def SVD_trunc_CCC(matC21, matC32, matC13, D_squared, chi):
    
    return U, C21, Vdag, U, C32, Vdag, U, C13, Vdag



def update_environmentCTs_1to2(C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E, A,B,C,D,E,F, D_squared, chi):

    bigC21EB = oe.contract("YX,MYa,LXβ,amg,lbg->MmLl",
                           C21CD,T1F,T2A,E,B,
                           optimize=[(0,2),(0,3),(1,2),(0,1)],
                           backend='pytorch')
    
    matC21EB = bigC21EB.view(chi*D_squared,chi*D_squared)
    
    bigC32AD = oe.contract("ZY,NZβ,MYg,abn,amg->NnMm",
                           C32EF,T2B,T3C,A,D,
                           optimize=[(0,2),(0,3),(1,2),(0,1)],
                           backend='pytorch')
    
    matC32AD = bigC32AD.view(chi*D_squared,chi*D_squared)
    
    bigC13CF = oe.contract("XZ,LXg,NZa,lbg,abn->LlNn",
                           C13AB,T3D,T1E,C,F,
                           optimize=[(0,2),(0,3),(1,2),(0,1)],
                           backend='pytorch')
    
    matC13CF = bigC13CF.view(chi*D_squared,chi*D_squared)

    U, C21EB, Vdag, U, C32AD, Vdag, U, C13CF, Vdag = SVDs_trunc_CCC(
                        matC21EB, matC32AD, matC13CF, D_squared, chi)

    T3E = oe.contract("OYa,abg,MOb->MYg",
                        T1F,E,Vdag1F,
                        optimize=[(0,1),(0,1)],
                        backend='pytorch')

    T3B = oe.contract("OXb,abg,LOa->LXg",
                        T2A,B,U2A,
                        optimize=[(0,1),(0,1)],
                        backend='pytorch')
    
    T1A = oe.contract("OZb,abg,NOg->NZa",
                        T2B,A,Vdag2B,
                        optimize=[(0,1),(0,1)],
                        backend='pytorch')
    
    T1D = oe.contract("OYg,abg,MOb->MYa",
                        T3C,D,U3C,
                        optimize=[(0,1),(0,1)],
                        backend='pytorch')
    
    T2C = oe.contract("OXg,abg,LOa->LXb",
                        T3D,C,Vdag3D,
                        optimize=[(0,1),(0,1)],
                        backend='pytorch')
    
    T2F = oe.contract("OZa,abg,NOg->NZb",
                        T1E,F,U1E,
                        optimize=[(0,1),(0,1)],
                        backend='pytorch')

    return C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A




def update_environmentCTs_2to3(C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A, A,B,C,D,E,F, D_squared, chi):

    return C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C




def update_environmentCTs_3to1(C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C, A,B,C,D,E,F, D_squared, chi):

    return C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E




def CTMRG_from_init_to_stop(A,B,C,D,E,F,
                            D_squared: int,
                            chi: int, 
                            max_iterations: int, 
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
    nowC21CD, nowC32EF, nowC13AB, nowT1F, nowT2A, nowT2B, nowT3C, nowT3D, nowT1E = initialize_environmentCTs_1(A,B,C,D,E,F, D_squared, chi)
    nowC21EB, nowC32AD, nowC13CF, nowT1D, nowT2C, nowT2F, nowT3E, nowT3B, nowT1A = None, None, None, None, None, None, None, None, None
    nowC21AF, nowC32CB, nowC13ED, nowT1B, nowT2E, nowT2D, nowT3A, nowT3F, nowT1C = None, None, None, None, None, None, None, None, None

    # Perform the CTMRG iterations until convergence
    for iteration in range(max_iterations):

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
        match iteration % 3 :
            case 0 : 
                lastC21EB, lastC32AD, lastC13CF, lastT1D, lastT2C, lastT2F, lastT3E, lastT3B, lastT1A = \
                nowC21EB, nowC32AD, nowC13CF, nowT1D, nowT2C, nowT2F, nowT3E, nowT3B, nowT1A
                nowC21EB, nowC32AD, nowC13CF, nowT1D, nowT2C, nowT2F, nowT3E, nowT3B, nowT1A = update_environmentCTs_1to2(
                nowC21CD, nowC32EF, nowC13AB, nowT1F, nowT2A, nowT2B, nowT3C, nowT3D, nowT1E, A,B,C,D,E,F, D_squared, chi)
            
            case 1 : 
                lastC21AF, lastC32CB, lastC13ED, lastT1B, lastT2E, lastT2D, lastT3A, lastT3F, lastT1C = \
                nowC21AF, nowC32CB, nowC13ED, nowT1B, nowT2E, nowT2D, nowT3A, nowT3F, nowT1C

                nowC21AF, nowC32CB, nowC13ED, nowT1B, nowT2E, nowT2D, nowT3A, nowT3F, nowT1C = update_environmentCTs_2to3(
                nowC21EB, nowC32AD, nowC13CF, nowT1D, nowT2C, nowT2F, nowT3E, nowT3B, nowT1A, A,B,C,D,E,F, D_squared, chi)
            
            case 2 : 
                lastC21CD, lastC32EF, lastC13AB, lastT1F, lastT2A, lastT2B, lastT3C, lastT3D, lastT1E = \
                nowC21CD, nowC32EF, nowC13AB, nowT1F, nowT2A, nowT2B, nowT3C, nowT3D, nowT1E
                
                nowC21CD, nowC32EF, nowC13AB, nowT1F, nowT2A, nowT2B, nowT3C, nowT3D, nowT1E = update_environmentCTs_3to1(
                nowC21AF, nowC32CB, nowC13ED, nowT1B, nowT2E, nowT2D, nowT3A, nowT3F, nowT1C, A,B,C,D,E,F, D_squared, chi)
    
    return iteration % 3, \
        nowC21CD, nowC32EF, nowC13AB, nowT1F, nowT2A, nowT2B, nowT3C, nowT3D, nowT1E, \
        nowC21EB, nowC32AD, nowC13CF, nowT1D, nowT2C, nowT2F, nowT3E, nowT3B, nowT1A, \
        nowC21AF, nowC32CB, nowC13ED, nowT1B, nowT2E, nowT2D, nowT3A, nowT3F, nowT1C





