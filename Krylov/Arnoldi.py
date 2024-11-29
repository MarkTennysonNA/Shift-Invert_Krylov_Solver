class Arnoldi:
    def __init__(self):
        '''Constructor for this class.'''

import torch
from Krylov import L2

def Rational(A,v,m,shifts,dx):
    N=len(v)
    #dx=(2*240)/N
    Vm=torch.zeros((N,m+1),dtype=torch.complex128)
    Hm=torch.zeros((m+1,m+1),dtype=torch.complex128)
    beta=L2.Norm(v,dx)
    Vm[:,0]=v/beta
    for j in range (1,m+1):
        if (j<=len(shifts)):
            B=torch.eye(N)-A/shifts[j-1]
            #hold=torch.matmul(torch.inverse(B),Vm[:,j-1])
            hold=torch.linalg.solve(B,torch.matmul(A,Vm[:,j-1]))
        else:
            hold=torch.matmul(A,Vm[:,j-1])
        for i in range (j):
            Hm=Hm.clone()
            Hm[i,j-1]=L2.IP(Vm[:,i],hold,dx)
            hold=hold-Hm[i,j-1]*Vm[:,i]
        Hm=Hm.clone()
        Hm[j,j-1]=L2.Norm(hold,dx)
        Vm=Vm.clone()
        Vm[:,j]=torch.divide(hold,Hm[j,j-1])
            
    return Vm,Hm,beta

def Polynomial(A,v,m,dx,reo = 1):
    N = len(v)
    Vm = torch.zeros((N,m+1),dtype =torch.complex128)
    Hm = torch.zeros((m+1,m+1),dtype = torch.complex128)
    beta = L2.Norm(v,dx)
    Vm[:,0] = v/beta
    for j in range (1,m+1):
        hold = torch.matmul(A,Vm[:,j-1])
        for i in range (j):
            Hm = Hm.clone()
            Hm[i,j-1] = L2.IP(Vm[:,i],hold,dx)
            hold = hold.clone()
            hold -= Hm[i,j-1]*Vm[:,i]
        if (reo > 0):
            for i in range (j):
                Htemp = L2.IP(Vm[:,i],hold,dx)
                Hm = Hm.clone()
                Hm[i,j-1] += Htemp[0]
                hold = hold.clone()
                hold -= Htemp*Vm[:,i]
        Hm = Hm.clone()
        Hm[j,j-1] = L2.Norm(hold,dx)
        Vm = Vm.clone()
        Vm[:,j] = hold/Hm[j][j-1]
    return Vm,Hm,beta
