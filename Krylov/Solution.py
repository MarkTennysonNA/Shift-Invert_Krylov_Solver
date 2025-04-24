class Solution:
    def __init__(self):
        '''Constructor for this class.'''

import torch
import Krylov
import numpy as np
# import scipy.special

def Polynomial(Vm,Hm,dt,beta,func):
    #Define e1, the first unit vector
    m=len(Hm[0])
    e1=torch.zeros((m,),dtype=torch.complex128)
    e1[0]=1

    #Concatenate Matrices
    # Vm=Vm[:,0:-1]
    # Hm=Hm[0:-1,0:-1]

    #Calculate Polynomial Krylov Estimate
    f=func(dt*Hm)
    Estimate=beta*torch.matmul(Vm,torch.matmul(f,e1))
    return Estimate

def Shift_Invert(Vm,Hm,dt,beta,shift,func):
    #Define e1, the first unit vector
    m=len(Hm[0])-1
    e1=torch.zeros((m,),dtype=torch.complex128)
    e1[0]=1

    #Concatenate Matrices
    Vm=Vm[:,0:-1]
    Hm=Hm[0:-1,0:-1]

    #Calculate Polynomial Krylov Estimate
    f=func(dt*(shift*torch.eye(m)-torch.inverse(Hm)))
    Estimate=beta*torch.matmul(Vm,torch.matmul(f,e1))
    return Estimate

def Exact(A,v,dt,func):
    return torch.matmul(func(dt*A),v)

def defect(A,Vm,Am,beta,dt,dx,m):
    e1 = torch.zeros(len(Am),dtype = torch.complex128)
    e1[0] = 1
    dtEst = beta * Vm @ Am @ torch.matrix_exp(Am*dt) @ e1
    AEst = beta * A @ Vm @ torch.matrix_exp(Am*dt) @ e1
    return (dt/(m+1))*Krylov.L2.Error(dtEst,AEst,dx)

def Crank_Nicolson(A,v,dt):
    I = torch.eye(len(v))
    return torch.linalg.solve(I-(dt/2)*A,torch.matmul(I+(dt/2)*A,v))

def Rayleigh_Quotient(A,Vm,Hm,gamma,dx):
    if not(isinstance(gamma,torch.Tensor)):
        gamma = torch.tensor(gamma)
    m = len(Hm) - 1
    em = torch.zeros((1,m),dtype = torch.complex128)
    em[0,-1] = 1
    #Calculate Rayleigh quotient based on whether A is Skew-hermitian or not.
    if torch.max(torch.abs(A + torch.transpose(torch.conj(A),0,1))) < 1e-14: #Skew-Hermitian A
        kappa = Krylov.L2.IP(Vm[:,-1],A @ Vm[:,-1],dx)
        ym = em @ torch.linalg.inv(Hm[0:-1,0:-1])
        Am = torch.linalg.inv(Hm[0:-1,0:-1]) + gamma*torch.eye(m) + (Hm[-1,-2]**2)*(kappa + torch.conj(gamma))*torch.transpose(torch.conj(ym),0,1) @ ym
    else: # General A
        Am = torch.linalg.inv(Hm[0:-1,0:-1]) + gamma*torch.eye(m) - Hm[-1,-2]*torch.reshape(Krylov.L2.IP(Vm[:,0:-1],A @ Vm[:,-1],dx),(m,1)) @ em @torch.linalg.inv(Hm[0:-1,0:-1])
    return Am

def RQ_defect(A,Vm,Hm,beta,gamma,dt,dx):
    if not(isinstance(gamma,torch.Tensor)):
        gamma = torch.tensor(gamma)
    #Compute Rayleigh quotient matrix
    Am = Rayleigh_Quotient(A,Vm,Hm,gamma,dx)
    #Compute unit vectors
    m = len(Hm) - 1
    em = torch.zeros((1,m),dtype = torch.complex128)
    em[0,-1] = 1
    e1 = torch.zeros((m,),dtype = torch.complex128)
    e1[0] = 1
    #Compute integral of defect norm
    ymstar = em @ torch.linalg.inv(Hm[0:-1,0:-1])
    ym = torch.transpose(torch.conj(ymstar),0,1)
    ym = torch.reshape(ym,(ym.shape[0],))
    def integrand(s):
        return torch.abs(ymstar @ torch.matrix_exp(s*Am) @ e1)
    integral = torch.tensor([0.],dtype = torch.float64)
    roots = torch.tensor([-0.97390653, -0.86506337, -0.67940957, -0.43339539, -0.14887434,  0.14887434, 0.43339539,  0.67940957,  0.86506337,  0.97390653])
    weights = torch.tensor([0.06667134, 0.14945135, 0.21908636, 0.26926672, 0.29552422, 0.29552422, 0.26926672, 0.21908636, 0.14945135, 0.06667134])
    for i in range (len(roots)):
        integral += dt*integrand(dt/2 * roots[i] + dt/2)*(weights[i])
    # nquad = 1
    # h = torch.linspace(0,dt,nquad + 1)
    # h = h[1:]
    # for i in range (nquad):
    #     integral += dt*integrand(h[i])/nquad
    #Compute vector norm for general or skew-Hermitian matrices
    if torch.max(torch.abs(A + torch.transpose(torch.conj(A),0,1))) < 1e-14: #Skew-Hermitian A
        kappa = Krylov.L2.IP(Vm[:,-1],A @ Vm[:,-1],dx)
        # vec = Hm[-1,-2]*(kappa + np.conj(gamma))*Vm[:,0:-1]@ym + (A-gamma*torch.eye(len(Vm[:,0])))@Vm[:,-1]
        vnorm = (Krylov.L2.Norm((A - gamma*torch.eye(len(A[0,:])))@Vm[:,-1],dx)**2 - (Hm[-1,-2]*torch.abs(kappa + torch.conj(gamma))*Krylov.L2.Norm(ym,1))**2)**(1/2)
    else: # General A
        vec = (A - gamma*torch.eye(len(Vm[:,0]))) @ Vm[:,-1] - Vm[:,0:-1] @ Krylov.L2.IP(Vm[:,0:-1], A@Vm[:,-1],dx)
        vnorm = Krylov.L2.Norm(vec,dx)
    return (beta * Hm[-1,-2] * vnorm * integral).real

def SI_defect(A,Vm,Hm,beta,gamma,dt,dx):
    m = len(Hm) - 1
    Am = torch.linalg.inv(Hm[0:-1,0:-1]) + gamma*torch.eye(m)
    em = torch.zeros((1,m),dtype = torch.complex128)
    em[0,-1] = 1
    e1 = torch.zeros((m,),dtype = torch.complex128)
    e1[0] = 1
    #Compute integral of defect norm
    ym = em @ torch.linalg.inv(Hm[0:-1,0:-1])
    def integrand(s):
        return torch.abs(ym @ torch.matrix_exp(s*Am) @ e1)
    integral = torch.tensor([0.],dtype = torch.float64)
    roots = torch.tensor([-0.97390653, -0.86506337, -0.67940957, -0.43339539, -0.14887434,  0.14887434, 0.43339539,  0.67940957,  0.86506337,  0.97390653])
    weights = torch.tensor([0.06667134, 0.14945135, 0.21908636, 0.26926672, 0.29552422, 0.29552422, 0.26926672, 0.21908636, 0.14945135, 0.06667134])
    integral = torch.tensor([0.],dtype = torch.float64)
    for i in range (len(roots)):
        integral += dt*integrand(dt/2 * roots[i] + dt/2)*(weights[i])
    #Compute vector norm
    vnorm = Krylov.L2.Norm((A - gamma*torch.eye(len(Vm[:,0])))@Vm[:,-1],dx)
    return (beta * Hm[-1,-2] * vnorm * integral).real