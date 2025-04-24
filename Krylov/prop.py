import Krylov
import torch

def SIK_propagator(true_A,true_v,true_dx,surrogate_A,surrogate_v,surrogate_dx,m,proptype,opttype,maxiter=1000,ftol=1e-15,rand_n = 3,opt_bounds=[(0,100),(-100,0)],x0=None,shift = None):
    #Find Optimal shift
    options = {'iprint':0,
           'maxiter':maxiter,
           'disp': True,
           'gtol':1e-15,
           'ftol':1e-15}
    if shift == None:
        shifts,obj_loss,randshifts = Krylov.Optimise.Optimise(A = surrogate_A,
                                                            v = surrogate_v,
                                                            dx = surrogate_dx,
                                                            m = m,
                                                            dt = 1,
                                                            options = options,
                                                            tol = ftol,
                                                            bounds = opt_bounds,
                                                            rand_n = rand_n,
                                                            rand_bounds = opt_bounds,
                                                            x0 = x0,
                                                            loss_fn = opttype)
        #Collect optimisation data into a dict
        opt_data = {'shifts':shifts,
                    'objective':obj_loss,
                    'random_shifts':randshifts}
    else:
        shifts = [shift]
        opt_data = {'shifts':shifts}


    #Propagate in time
    X = torch.linalg.inv(true_A - shifts[-1]*torch.eye(len(true_v)))
    Vm,Hm,beta = Krylov.Arnoldi.Polynomial(X,true_v,m,true_dx)
    if proptype == "RQ":
        Am = Krylov.Solution.Rayleigh_Quotient(true_A,Vm,Hm,shifts[-1],true_dx)
    elif proptype == "SI":
        Am = shifts[-1]*torch.eye(m) + torch.linalg.inv(Hm[0:-1,0:-1])
    else:
        print("Error - variable 'proptype' must be either 'RQ' or 'SI'.")
    Est = Krylov.Solution.Polynomial(Vm[:,0:-1],Am,1,beta,torch.matrix_exp)
    return Est, opt_data

def PolyK_propagator(true_A,true_v,m,true_dx):
    Vm,Hm,beta = Krylov.Arnoldi.Polynomial(true_A,true_v,m,true_dx)
    Est = Krylov.Solution.Polynomial(Vm[:,0:-1],Hm[0:-1,0:-1],1,beta,torch.matrix_exp)
    return Est

def Pade_propagator(true_A,true_v):
    Est = torch.matrix_exp(true_A) @ true_v
    return Est

