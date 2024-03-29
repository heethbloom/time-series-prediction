#%%
import torch
import numpy as np

#%%



#%%
def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    print(x_norm)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    print(y_norm)
    print(x_norm + y_norm)
    print(torch.mm(x, y_t))
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    print(dist)
    return torch.clamp(dist, 0.0, float('inf'))
#%%
x = torch.tensor(np.array([[1], [2], [3]]))
y = torch.tensor(np.array([[2], [3], [4]]))

#%%
# pairwise_distances(x, y)
# %%
N_output = 10
pairwise_distances(torch.range(1,N_output).view(N_output,1))
# %%
import torch
import soft_dtw
import path_soft_dtw 
import torch.nn
import numpy as np

def dilate_loss(outputs, targets, alpha, gamma, device):
	# outputs, targets: shape (batch_size, N_output, 1)
	batch_size, N_output = outputs.shape[0:2]
	loss_shape = 0
	softdtw_batch = soft_dtw.SoftDTWBatch.apply
	C = torch.zeros((batch_size, N_output,N_output )).to(device)
	for k in range(batch_size):
		Ck = soft_dtw.pairwise_distances(targets[k,:,:].view(-1,1),outputs[k,:,:].view(-1,1))
		C[k:k+1,:,:] = Ck     
  
	loss_shape = softdtw_batch(C,gamma)	
	path_dtw = path_soft_dtw.PathDTWBatch.apply
	path = path_dtw(C,gamma)           
	Omega =  soft_dtw.pairwise_distances(torch.range(1,N_output).view(N_output,1)).to(device)
	loss_temporal =  torch.sum( path*Omega ) / (N_output*N_output) 
	loss = alpha*loss_shape + (1-alpha)*loss_temporal 
	return loss
#%%
def dilate_div_loss(outputs, targets, alpha, gamma, device):
	# outputs, targets: shape (batch_size, N_output, 1)
	batch_size, N_output = outputs.shape[0:2]
	loss_shape = 0
	softdtw_batch = soft_dtw.SoftDTWBatch.apply
	C = torch.zeros((batch_size, N_output,N_output )).to(device)
	CXX = torch.zeros((batch_size, N_output,N_output )).to(device)
	CYY = torch.zeros((batch_size, N_output,N_output )).to(device)
	for k in range(batch_size):
		Ck = soft_dtw.pairwise_distances(targets[k,:,:].view(-1,1),outputs[k,:,:].view(-1,1))
		C[k:k+1,:,:] = Ck
		CXXk = soft_dtw.pairwise_distances(targets[k,:,:].view(-1,1),targets[k,:,:].view(-1,1))
		CXX[k:k+1,:,:] = CXXk 
		CYYk = soft_dtw.pairwise_distances(outputs[k,:,:].view(-1,1),outputs[k,:,:].view(-1,1))
		CYY[k:k+1,:,:] = CYYk 
  
	loss_shape = softdtw_batch(C,gamma)	- softdtw_batch(CXX,gamma)/2 - softdtw_batch(CYY,gamma)/2
	path_dtw = path_soft_dtw.PathDTWBatch.apply
	path = path_dtw(C,gamma)           
	pathXX = path_dtw(CXX,gamma) 
	pathYY = path_dtw(CYY,gamma)     
	Omega =  soft_dtw.pairwise_distances(torch.range(1,N_output).view(N_output,1)).to(device)
	loss_temporal =  torch.sum(path*Omega) / (N_output*N_output)
	loss_temporal =  (torch.sum(path*Omega) - torch.sum(pathXX*Omega)/2 - torch.sum(pathYY*Omega)/2) / (N_output*N_output) 
	loss = alpha*loss_shape + (1-alpha)*loss_temporal 
	return loss
#%%
def derivative_dilate_loss(outputs, targets,  alpha, gamma, device):
	# outputs, targets: shape (batch_size, N_output, 1)
    batch_size, N_output = outputs.shape[0:2]
    loss_shape = 0
    softdtw_batch = soft_dtw.SoftDTWBatch.apply
    C = torch.zeros((batch_size, N_output,N_output)).to(device)

    for k in range(batch_size):
        Ck = soft_dtw.pairwise_distances(soft_dtw.derivative_form(targets[k,:,:]),
                                   soft_dtw.derivative_form(outputs[k,:,:]))
        C[k:k+1,:,:] = Ck     
    loss_shape = softdtw_batch(C,gamma)	
    path_dtw = path_soft_dtw.PathDTWBatch.apply
    path = path_dtw(C,gamma)           
    Omega =  soft_dtw.pairwise_distances(torch.range(1,N_output).view(N_output,1)).to(device)
    loss_temporal =  torch.sum( path*Omega ) / ((N_output)*(N_output)) 
    loss = (alpha*loss_shape + (1-alpha)*loss_temporal)
    return loss


#%%
outputs = torch.tensor(np.array([[[1], [3], [3], [1], [2], [3]]]))
targets = torch.tensor(np.array([[[1], [3], [3], [1], [2], [3]], [[1], [3], [3], [1], [2], [3]]]))
# y = torch.tensor(np.array([[[1], [3], [3], [50], [2], [3]]]))
#%%
d_targets = targets.clone()
d_outputs = outputs.clone()
batch_size, N_output = outputs.shape[0:2]
#%%
for i in range(1,N_output-1):
    d_targets[:,i,:][0] = ((targets[:,i,:][0]  - targets[:,i-1,:][0] ) + ((targets[:,i+ 1,:][0] - targets[:,i-1,:][0])/2))/2
d_targets[:,0,:][0]  =  d_targets[:,1,:][0] 
d_targets[:,N_output-1,:][0]  = d_targets[:,N_output-2,:][0] 
#%%
for i in range(1,N_output-1):
    d_targets[:,i,:] = ((targets[:,i,:]  - targets[:,i-1,:] ) + ((targets[:,i+ 1,:] - targets[:,i-1,:])/2))/2
d_targets[:,0,:] =  d_targets[:,1,:]
d_targets[:,N_output-1,:]  = d_targets[:,N_output-2,:]

#%%
d_targets
#%%
targets.shape

#%%
def derivative_form(x):
    '''
    Input: x is a Nx1 array (univariate sequence) 
    Output: Nx1 matrix 
    '''
    n = len(x)
    x1 = x.clone().detach()
    x2 = x.clone().detach()
    x1  = torch.cat([x1[1:], x1[-1:]])
    x2  = torch.cat([x2[0:1], x2[:-1]])
   
    return (x - x2 + (x1 -x2)/2)/2
#%%
x = torch.tensor(np.array([[1], [2], [3], [4], [5]]))
x.shape

#%%
x1 = x.clone().detach()
x2 = x.clone().detach()
x1  = torch.cat([x1[1:], x1[-1:]])
x2  = torch.cat([x2[0:1], x2[:-1]])

#%%
x1, x2

#%%

(x - x2 + (x1 -x2)/2)/2
#%%
derivative_form(x).shape
#%%
derivative_dilate_loss(x, y, 0.5, 0.01, 'cuda:2')

#%%
dx = torch.empty_like(x).copy_(x).to(dtype=torch.float)

#%%
x
#%%
targets = torch.tensor(np.array([[[1], [2], [3], [4], [5]]]))
batch_size, N_output = targets.shape[0:2]
d_targets = torch.empty_like(targets).copy_(targets)
#%%
d_targets[:,1,:]
#%%
# print(d_targets)
for i in range(1,N_output-1):
    d_targets[:,i,:][0] = ((targets[:,i,:][0] - targets[:,i-1,:][0]) + ((targets[:,i+ 1,:][0] - targets[:,i-1,:][0])/2))/2
print(d_targets)
#%%
d_targets[:,0,:][0] =  d_targets[:,1,:][0]
d_targets[:,N_output-1,:][0] = d_targets[:,N_output-2,:][0]
d_targets
#%%
dx[:,0:1,:] =  dx[:,1:2,:]
dx[:,N_output-1:N_output,:] = dx[:,N_output-2:N_output-1,:]

#%%
dx
#%%
derivative_dilate_loss(x, y, 0.5, 0.01, 'cuda:0')

#%%
for i in np.arange(0, 1.1, 0.1):
    val = derivative_dilate_loss(x, y, i, 0.01, 'cuda:0')
    print(i, val)
# %%
dilate_div_loss(x, y, 0, 0.01, 'cuda:2')
# %%
def weighted_pairwise_distances(x, y, g = 0.05):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. wdist[i,j] = w(i,j) * ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    print(dist)
    # w = 1 / (1 + np.exp(-g()))
    n = x.size()[0]
    m = y.size()[0]
    w = torch.zeros((n, m))
    for i in range(n):
        for j in range(m):
            w[i][j]  =  1 / (pow((1 + np.exp(-g * (np.abs(i-j)-n/2))),2))
    print(w)
    wdist = w * dist
    print(wdist)
    return torch.clamp(wdist, 0.0001, float('inf'))

#%%
x = torch.tensor(np.array([[1], [2], [3]]))
y = torch.tensor(np.array([[2], [3], [4]]))
x.size()[0]
y.size()[0]
#%%
weighted_pairwise_distances(x, y, g = 0.05)
# %%
