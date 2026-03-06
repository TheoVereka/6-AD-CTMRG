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
    pass
    # return nowC21CD, nowC32EF, nowC13AB, nowT1F, nowT2A, nowT2B, nowT3C, nowT3D, nowT1E

def check_env_convergence(lastC21CD, lastC32EF, lastC13AB, lastT1F, lastT2A, lastT2B, lastT3C, lastT3D, lastT1E, 
                          nowC21CD, nowC32EF, nowC13AB, nowT1F, nowT2A, nowT2B, nowT3C, nowT3D, nowT1E, 
                          last_environmentCTs_2, 
                          now_environmentCTs_2, 
                          last_environmentCTs_3, 
                          now_environmentCTs_3, 
                          env_conv_threshold):
    if last_environmentCTs_3 is None:
        return False
    pass

def update_environmentCTs_1to2(nowC21CD, nowC32EF, nowC13AB, nowT1F, nowT2A, nowT2B, nowT3C, nowT3D, nowT1E, A,B,C,D,E,F, D_squared, chi):

    pass

def update_environmentCTs_2to3(now_environmentCTs_2, A,B,C,D,E,F, D_squared, chi):

    pass

def update_environmentCTs_3to1(now_environmentCTs_3, A,B,C,D,E,F, D_squared, chi):

    pass

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
        last_environmentCTs (dict): A dictionary containing the last environment corner and edge transfer tensors.
        now_environmentCTs (dict): A dictionary containing the current environment corner and edge transfer tensors.
    """
    # Initialize the environment corner and edge transfer tensors


    lastC21CD, lastC32EF, lastC13AB, lastT1F, lastT2A, lastT2B, lastT3C, lastT3D, lastT1E = None, None, None, None, None, None, None, None, None
    last_environmentCTs_2 = None, None, None, None, None, None, None, None, None
    last_environmentCTs_3 = None, None, None, None, None, None, None, None, None
    nowC21CD, nowC32EF, nowC13AB, nowT1F, nowT2A, nowT2B, nowT3C, nowT3D, nowT1E = initialize_environmentCTs_1(A,B,C,D,E,F, D_squared, chi)
    now_environmentCTs_2 = None, None, None, None, None, None, None, None, None
    now_environmentCTs_3 = None, None, None, None, None, None, None, None, None

    # Perform the CTMRG iterations until convergence
    for iteration in tqdm(range(max_iterations)):

        if check_env_convergence(lastC21CD, lastC32EF, lastC13AB, lastT1F, lastT2A, lastT2B, lastT3C, lastT3D, lastT1E, 
                                 nowC21CD, nowC32EF, nowC13AB, nowT1F, nowT2A, nowT2B, nowT3C, nowT3D, nowT1E, 
                                 last_environmentCTs_2, 
                                 now_environmentCTs_2, 
                                 last_environmentCTs_3, 
                                 now_environmentCTs_3, 
                                 env_conv_threshold):
            print(f"Convergence achieved at iteration {iteration}.")
            break

        # Update the environment corner and edge transfer tensors
        match iteration % 3 :
            case 0 : 
                last_environmentCTs_2 = \
                now_environmentCTs_2
                now_environmentCTs_2 = update_environmentCTs_1to2(
                nowC21CD, nowC32EF, nowC13AB, nowT1F, nowT2A, nowT2B, nowT3C, nowT3D, nowT1E, A,B,C,D,E,F, D_squared, chi)
            
            case 1 : 
                last_environmentCTs_3 = \
                now_environmentCTs_3

                now_environmentCTs_3 = update_environmentCTs_2to3(
                now_environmentCTs_2, A,B,C,D,E,F, D_squared, chi)
            
            case 2 : 
                lastC21CD, lastC32EF, lastC13AB, lastT1F, lastT2A, lastT2B, lastT3C, lastT3D, lastT1E = \
                nowC21CD, nowC32EF, nowC13AB, nowT1F, nowT2A, nowT2B, nowT3C, nowT3D, nowT1E
                
                nowC21CD, nowC32EF, nowC13AB, nowT1F, nowT2A, nowT2B, nowT3C, nowT3D, nowT1E = update_environmentCTs_3to1(
                now_environmentCTs_3, A,B,C,D,E,F, D_squared, chi)
    
    return iteration%3, \
        nowC21CD, nowC32EF, nowC13AB, nowT1F, nowT2A, nowT2B, nowT3C, nowT3D, nowT1E, \
        now_environmentCTs_2, \
        now_environmentCTs_3





