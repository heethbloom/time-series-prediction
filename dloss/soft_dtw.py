import numpy as np
import torch
from numba import jit
from torch.autograd import Function, Variable

# def shape_descriptor(x, l): #shape descriptors
#     '''
#     Input: x is a Nx1 array (univariate sequence) 
#     Output: Nxl matrix (numtil dimensional sequence = sequence of subsequences)
#     '''
#     n = len(x)
#     for _ in range(l // 2):
#         x = torch.cat([x[0:1], x])
#         x = torch.cat([x, x[-1:]])
#     main_seq = torch.zeros((n, l), requires_grad=True)
#     for i in range(0, n):  
#         main_seq[i] = x[i:i+l, 0]
#     return main_seq

# def shape_descriptor(x, l): #shape descriptors
#     '''
#     Input: x is a Nx1 array (univariate sequence) 
#     Output: Nxl matrix (numtil dimensional sequence = sequence of subsequences)
#     '''
#     n = len(x)
#     for _ in range(l // 2):
#         x = torch.cat([x[0:1], x])
#         x = torch.cat([x, x[-1:]])
#     main_seq = torch.zeros(n, l)
#     for i in range(0, n):  
#         main_seq[i] = x[i:i+l, 0]
#     main_seq = Variable(main_seq, requires_grad = True)
#     return main_seq
def shape_descriptor(x, l): #shape descriptors
    '''
    Input: x is a Nx1 array (univariate sequence) 
    Output: Nxl matrix (numtil dimensional sequence = sequence of subsequences)
    '''
    n = len(x)
    for _ in range(l // 2):
        x = torch.cat([x[0:1], x])
        x = torch.cat([x, x[-1:]])
    main_seq = torch.zeros((n, l))
    for i in range(0, n):  
        main_seq[i] = x[i:i+l, 0]
    return main_seq
    


def derivative_form(x):
    '''
    Input: x is a Nx1 array (univariate sequence) 
    Output: Nx1 matrix 
    '''
    x1  = torch.cat([x[1:], x[-1:]])
    x2  = torch.cat([x[0:1], x[:-1]])
    return (x - x2 + (x1 - x2)/2)/2

def batch_shape_descriptor(x, l):
    '''
    Input: x is a kxNx1 array (univariate sequence) 
    Output: kxNxl matrix (numtil dimensional sequence = sequence of subsequences)
    '''
    n = x.shape[1]
    b = x.shape[0]
    X = torch.zeros((b, n, l ))
    for _ in range(l // 2):
        x = torch.cat([x[:,0:1, :], x], dim = 1)
        x = torch.cat([x, x[:, -1:, :]], dim = 1)
    for i in range(n):  
        X[:, i, :] = x[:,i:i+l, 0]
    return X

def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, float('inf'))

@jit(nopython = True)
def compute_softdtw(D, gamma):
  N = D.shape[0]
  M = D.shape[1]
  R = np.zeros((N + 2, M + 2)) + 1e8
  R[0, 0] = 0
  for j in range(1, M + 1):
    for i in range(1, N + 1):
      r0 = -R[i - 1, j - 1] / gamma
      r1 = -R[i - 1, j] / gamma
      r2 = -R[i, j - 1] / gamma
      rmax = max(max(r0, r1), r2)
      rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
      softmin = - gamma * (np.log(rsum) + rmax)
      R[i, j] = D[i - 1, j - 1] + softmin
  return R

@jit(nopython = True)
def compute_softdtw_backward(D_, R, gamma):
  N = D_.shape[0]
  M = D_.shape[1]
  D = np.zeros((N + 2, M + 2))
  E = np.zeros((N + 2, M + 2))
  D[1:N + 1, 1:M + 1] = D_
  E[-1, -1] = 1
  R[:, -1] = -1e8
  R[-1, :] = -1e8
  R[-1, -1] = R[-2, -2]
  for j in range(M, 0, -1):
    for i in range(N, 0, -1):
      a0 = (R[i + 1, j] - R[i, j] - D[i + 1, j]) / gamma
      b0 = (R[i, j + 1] - R[i, j] - D[i, j + 1]) / gamma
      c0 = (R[i + 1, j + 1] - R[i, j] - D[i + 1, j + 1]) / gamma
      a = np.exp(a0)
      b = np.exp(b0)
      c = np.exp(c0)
      E[i, j] = E[i + 1, j] * a + E[i, j + 1] * b + E[i + 1, j + 1] * c
  return E[1:N + 1, 1:M + 1]
 

class SoftDTWBatch(Function):
    @staticmethod
    def forward(ctx, D, gamma = 1.0): # D.shape: [batch_size, N , N]
        dev = D.device
        batch_size,N,N = D.shape
        gamma = torch.FloatTensor([gamma]).to(dev)
        D_ = D.detach().cpu().numpy()
        g_ = gamma.item()

        total_loss = 0
        R = torch.zeros((batch_size, N+2 ,N+2)).to(dev)   
        for k in range(0, batch_size): # loop over all D in the batch    
            Rk = torch.FloatTensor(compute_softdtw(D_[k,:,:], g_)).to(dev)
            R[k:k+1,:,:] = Rk
            total_loss = total_loss + Rk[-2,-2]
        ctx.save_for_backward(D, R, gamma)
        return total_loss / batch_size
  
    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        D, R, gamma = ctx.saved_tensors
        batch_size,N,N = D.shape
        D_ = D.detach().cpu().numpy()
        R_ = R.detach().cpu().numpy()
        g_ = gamma.item()

        E = torch.zeros((batch_size, N ,N)).to(dev) 
        for k in range(batch_size):         
            Ek = torch.FloatTensor(compute_softdtw_backward(D_[k,:,:], R_[k,:,:], g_)).to(dev)
            E[k:k+1,:,:] = Ek

        return grad_output * E, None
      



