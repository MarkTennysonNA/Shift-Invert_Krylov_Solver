import torch
import Krylov
import scipy.integrate as spi
import scipy.optimize
import scipy.interpolate
import scipy.special
import math
import numpy as np

ind = 0
iter = 1


class TerminationException(Exception):
    pass

def Optimise(A,v,dx,m,true_dt,coarse_dt,coarse_N,options,tol,bounds = [(None,None),(None,None)],rand_n = 3,rand_bounds = [[0,10],[-10,0]],x_range = [-1,1],x0 = False,loss_fn="defect",print_output=False,return_loss = False, ndt = 0):
    true_N = len(v)
    global iter
    def opt(A,v,dt,dx,m,gamma0,loss_fn,print_output):
        shifts = torch.zeros((10000,),dtype = torch.complex128)
        losses = torch.zeros((10000,))
        global ind
        global iter

        def obj_func(gamma):
            return loss(A,v,dt,dx,m,gamma,loss_fn)
        
        g = torch.tensor(gamma0)
        opt = torch.optim.LBFGS([g])
        # opt1 = torch.optim.LBFGS([g])
        def grad(gamma):
            gamma1 = torch.tensor(gamma,requires_grad = True,dtype = torch.float64)
            opt.zero_grad()
            objective = obj_func(gamma1)
            objective.backward()
            out = gamma1.grad.detach().numpy()
            # print("fd: ",fd(gamma1))
            # print("pt: ",out)
            return out
        
        if print_output == True:
            def callbackf(X):
                global ind
                global iter
                shifts[ind] = X[0] + 1j*X[1]
                losses[ind] = obj_func(X)
                print ('n = {3:5d}: iteration {0:4d} - shift = {1: 3.6f} loss = {2: 3.2e}'.format(iter, (X[0] + 1j*X[1]), float(obj_func(X)),int(ndt)), end = '\r')
                # print(iter)
                # print(X[0] + 1j*X[1])
                # print(obj_func(X))
                ind += 1
                iter += 1
                if obj_func(X) <= tol:
                    raise TerminationException("Function Tolerance Achieved")

        else:
            def callbackf(X):
                global ind
                global iter
                shifts[ind] = X[0] + 1j*X[1]
                losses[ind] = obj_func(X)
                ind += 1
                iter += 1
        try:
            # print(torch.tensor(gamma0).shape)
            res = scipy.optimize.minimize(obj_func,
                                    gamma0,
                                    method='L-BFGS-B',
                                    # method='SLSQP',
                                    callback=callbackf,
                                    jac=grad,
                                    # hess= pthess,
                                    options = options,
                                    bounds=bounds)
        except TerminationException as a:
            print("Optimisation terminated: ",a, end = '\r')
        # print(res.message)

        shifts = shifts[0:ind]
        losses = losses[0:ind]
        ind = 0
        iter = 1
        return shifts,losses

    #Calculate hamiltonian and initial condition for first problem
    if (coarse_N != true_N):
        v_downsample = interpv(v,coarse_N,x_range)
        A_downsample = 1j*interpA(-1j*A,true_N,coarse_N,dx,x_range)
        dx_downsample = (true_N - 1)*dx/(coarse_N - 1)
    else:
        v_downsample = v
        A_downsample = A
        dx_downsample = dx
    
    #Find initial guess of shift from random sample
    if x0 == False:
        init_gamma,randgammas,randerrors = randshift(A_downsample,v_downsample,coarse_dt,dx_downsample,m,loss_fn,rand_n,rand_bounds)
        # print(init_gamma)
    else:
        init_gamma = x0
        randgammas = x0[0] + 1j*x0[1]

    rawshifts = torch.tensor([init_gamma[0] + 1j*init_gamma[1]],dtype = torch.complex128)
    init_pp = pp(true_dt,coarse_dt,init_gamma)
    shifts = torch.tensor([init_pp[0] + 1j*init_pp[1]],dtype = torch.complex128)
    obj_loss = torch.tensor([loss(A,v,coarse_dt,dx_downsample,m,init_gamma,loss_fn)])
    if print_output == True:
        print ('n = {3:5d}: iteration {0:4d} - shift = {1: 3.6f} loss = {2: 3.2e}'.format(0, rawshifts[0], obj_loss[0],int(ndt)), end = '\r')
    # print ('{0:4d}   {1: 3.6f}   {2: 3.2e}'.format(0, rawshifts[0], obj_loss[0]))

    gamma = init_gamma
    #Calculate hamiltonian and initial condition for first problem
    s,l = opt(A_downsample,v_downsample,coarse_dt,dx_downsample,m,gamma,loss_fn,print_output)
    rawshifts = torch.cat((rawshifts,s))
    obj_loss = torch.cat((obj_loss,l))
    pps = torch.zeros(s.shape,dtype = torch.complex128)
    for j in range (len(s)):
        ppsvec = pp(true_dt,coarse_dt,[rawshifts[-(j+1)].real,rawshifts[-(j+1)].imag])
        pps[-(j+1)] = ppsvec[0] + 1j*ppsvec[1]
    shifts = torch.cat((shifts,pps))

    #Calculate hamiltonian and initial condition for final problem
    true_loss = torch.zeros(shifts.shape)
    if return_loss == True:
        for k in range (len(shifts)):
            true_loss[k] = loss(A,v,true_dt,dx,m,[shifts[k].real,shifts[k].imag],"l2")
        
    iter = 1
    ind = 0
    return shifts, true_loss, rawshifts, obj_loss, randgammas


def fcd_K2(dx,N):
    return torch.diag((dx**-2)*torch.ones((N-1,)),-1) + torch.diag((-2*dx**-2)*torch.ones((N,)),0) + torch.diag((dx**-2)*torch.ones((N-1,)),1)

def interpv(v,N,range):
    return Krylov.resample.downsample1D(v,N,range)

def interpA(A,N1,N0,dx,range):
    #Downsamples Hamiltonian matrix of the form K2 + Dv
    K2 = fcd_K2(dx,N1)
    Dv = A - K2
    Dv_downsample = torch.diag(interpv(torch.diag(Dv),N0,range))
    dx_downsample = (N1-1)*dx/(N0-1)
    K2_downsample = fcd_K2(dx_downsample,N0)
    return K2_downsample + Dv_downsample

def Ham(V,dx):
    n = len(V)
    return -1j*(torch.diag((-1*dx**-2)*torch.ones((n-1,)),-1)+torch.diag((2*dx**-2)*torch.ones((n,))+V,0)+torch.diag((-1*dx**-2)*torch.ones((n-1,)),1))

def loss(A,v,dt,dx,m,gamma,loss_fn):
    if loss_fn == "defect":
        shift = gamma[0] + 1j*gamma[1]
        X = torch.linalg.inv(A - shift*torch.eye(len(v)))
        Vm,Hm,beta = Krylov.Arnoldi.Polynomial(X,v,m,dx)
        #Rayleigh Quotient
        # out = Krylov.Solution.RQ_defect(A,Vm,Hm,beta,shift,dt,dx)
        #Shift Invert
        out = Krylov.Solution.SI_defect(A,Vm,Hm,beta,shift,dt,dx)
    elif loss_fn == "l2":
        shift = gamma[0] + 1j*gamma[1]
        Ref = Krylov.Solution.Exact(A,v,dt,torch.matrix_exp)
        X = torch.linalg.inv(A - shift*torch.eye(len(v)))
        Vm,Hm,beta = Krylov.Arnoldi.Polynomial(X,v,m,dx)
        # Am = Krylov.Solution.Rayleigh_Quotient(A,Vm,Hm,shift,dx)
        Am = shift*torch.eye(m) + torch.linalg.inv(Hm[0:-1,0:-1])
        # Am = Krylov.L2.IP(Vm[:,0:-1],A@Vm[:,0:-1],dx)
        Est = Krylov.Solution.Polynomial(Vm[:,0:-1],Am,dt,beta, torch.matrix_exp)
        out = Krylov.L2.Error(Ref,Est,dx).real
    elif loss_fn == "poly":
        shift = gamma[0] + 1j*gamma[1]
        PVm,PHm,PBeta = Krylov.Arnoldi.Polynomial(A,v,m,dx)
        Ref = Krylov.Solution.Polynomial(PVm,PHm,dt,PBeta,torch.matrix_exp)
        X = torch.linalg.inv(A - shift*torch.eye(len(v)))
        RVm,RHm,RBeta = Krylov.Arnoldi.Polynomial(X,v,m,dx)
        Est = Krylov.Solution.Shift_Invert(RVm,RHm,dt,RBeta,shift,torch.matrix_exp)
        out = Krylov.L2.Error(Ref,Est,dx)
    return out

def randshift(A,v,dt,dx,m,loss_fn,rand_n,rand_bounds):
    sample = rand_n
    errors = torch.zeros((sample,))
    shiftsr = torch.rand((sample,),dtype = torch.float64)*(rand_bounds[0][1]-rand_bounds[0][0]) + rand_bounds[0][0]
    shiftsi = torch.rand((sample,),dtype = torch.float64)*(rand_bounds[1][1]-rand_bounds[1][0]) + rand_bounds[1][0]
    for i in range (sample):
        errors[i] = loss(A,v,dt,dx,m,[shiftsr[i],shiftsi[i]],loss_fn)
    errors = torch.tensor([1e30 if math.isnan(x) else x for x in errors])
    minerrind =  (errors == torch.min(errors)).nonzero(as_tuple=True)[0]

    return [shiftsr[minerrind][0],shiftsi[minerrind][0]],shiftsr + 1j*shiftsi,errors

def pp(dtn,dtnm1,shift):
    gamma = torch.tensor(shift)
    dtfact = torch.tensor([(dtnm1/dtn),1.])
    return torch.mul(dtfact,gamma)