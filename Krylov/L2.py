class L2:
    def __init__(self):
        '''Constructor for this class. '''

import torch
import math

def IP(v,w,dx):
    v=torch.conj(v)
    if v.dim() == 1:
        v = torch.reshape(v,(len(v),1))
    vstar = torch.transpose(v,0,1)
    return dx * vstar @ w

def Norm(v,dx):
    # return math.sqrt(dx)*(torch.linalg.vector_norm(v,ord=2))
    return IP(v,v,dx)**(1/2)

def Error(v,w,dx):
    return Norm(torch.subtract(v,w),dx)
