import torch
import math
import numpy as np

def fourierproduct(fn, c, u, d):
    v = np.reshape(u,-1)
    app = np.multiply(fn(c),v)
    return app

def circshift(f,a,d):
    f = torch.tensor(f)
    return torch.roll(f,a,d)

def centralffthelper(f, d, s, fn):
    import math
    N = len(f)
    o = [0,0]
    if N%2 == 1:
        m = (N-1)/2
    else:
        m = N/2
        if s==1:
            o[1] = -1
        else:
            o[0] = -1
    X = (math.sqrt(N)**s) * circshift(fn(circshift(f, int(-(m+o[0])), d),[],d),int(m+o[1]),d)
    return X

def fft(f,n,d):
    import torch
    X = torch.fft.fft(f,dim = d)
    return X

def ifft(f,n,d):
    import torch
    X = torch.fft.ifft(f,dim = d)
    return X

def cifft(f):
    return centralffthelper(f,0,1,ifft)

def cfft(f):
    return centralffthelper(f,0,-1,fft)

def downsample1D(u0,n,xrange):
    import math
    a = xrange[0]
    b = xrange[1]
    n0 = len(u0)

    if n==n0:
        un = u0
    else: 
        U = cfft(u0)

        midpt = lambda n: (n/2) + 1 - (n%2)/2
        midpt0 = midpt(n0)
        midptn = midpt(n)

        rs = torch.arange(int(midpt0-midptn),int(midpt0+(n-midptn)))
        
        Un = U[rs]
        midgrid = lambda n,a,b: a+((2/n)*(1/2)*(1-n%2)+1)*(b-a)/2
        s = midgrid(n,a,b)-midgrid(n0,a,b)

        c = np.transpose(-1j*(torch.arange(1,n+1)-midptn)/((b-a)/2)*s*math.pi)
        Un = fourierproduct(torch.exp,c,Un,0)*math.sqrt(n/n0)
        un = cifft(Un)
    return un