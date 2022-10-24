
#%%
import torch
import numpy as np
x = torch.tensor(np.array([[1], [2], [3]]))
y = torch.tensor(np.array([[2], [3], [4]]))

#%%
def shape_descriptor(x, l): #shape descriptors
    n = len(x)
    for _ in range(l // 2):
        x = torch.cat([x[0:1], x])
        x = torch.cat([x, x[-1:]])
    main_seq = torch.zeros(n, l)
    for i in range(0, n):  
        main_seq[i] = x[i:i+l, 0]
    return main_seq

def batch_shape_descriptor(x, l):
    '''
    Input: x is a kxNx1 array (univariate sequence) 
    Output: kxNxl matrix (numtil dimensional sequence = sequence of subsequences)
    '''
    n = x.shape[1]
    b = x.shape[0]
    C = torch.zeros((b, n, l ))
    for _ in range(l // 2):
        x = torch.cat([x[:,0:1, :], x], dim = 1)
        x = torch.cat([x, x[:, -1:, :]], dim = 1)
    for i in range(n):  
        C[:, i, :] = x[:,i:i+l, 0]
    return C
    
#%%
l = 5
x = torch.tensor(np.array([[[1], [2], [3]], [[2], [3], [4]]]))   # 2 x 3 x 1

#%%
batch_shape_descriptor(x, l) # kxlxN


#%%
for i in range(l):
    print(batch_shape_descriptor(x, l)[0, :, i]) # shape = 3

#%%
for i in range(l):
    print(batch_shape_descriptor(x, l)[0, :, i:i+1]) # shape = 3 x 1
    
#%%           
    
# def shape_descriptor(x, l): #shape descriptors
#     '''
#     Input: x is a Nx1 array (univariate sequence) 
#     Output: Nxl matrix (numtil dimensional sequence = sequence of subsequences)
#     '''
#     x = list(np.array(x[:, 0]))
#     assert l % 2 ==1
#     main_seq = []
#     sub_seq = []
#     addition =  l // 2
#     for _ in range(addition):
#       x.insert(0, x[0])
#       x.append(x[-1])     # len(x) : N + l -1
#     for i in range(len(x)-l+1):  
#       sub_seq = x[i:i+l]
#       main_seq.append(sub_seq)  #main_seq.shape : Nxl
#     return torch.tensor(main_seq)
      

def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    # print(x_norm)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    # print(y_norm)
    # print(x_norm + y_norm)
    # print(torch.mm(x, y_t))
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # print(dist)
    return torch.clamp(dist, 0.0, float('inf'))
#%%
x = torch.tensor(np.array([[1], [2], [3]]))
y = torch.tensor(np.array([[2], [3], [4]]))

#%%
shape_x = shape_descriptor(x, 5)
shape_y = shape_descriptor(y, 5)
shape_x, shape_y
#%%
shape_x.shape
#%%
pairwise_distances(shape_x, shape_y)
#%%
pairwise_distances(x, y)

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
    print(targets[0,:,:].shape, targets[0,:,:].view(-1,1).shape)
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

def shape_dilate_loss_dep(outputs, targets, alpha, gamma, device, l = 5):
	# outputs, targets: shape (batch_size, N_output, 1)
	batch_size, N_output = outputs.shape[0:2]
	loss_shape = 0
	softdtw_batch = soft_dtw.SoftDTWBatch.apply
	C = torch.zeros((batch_size, N_output, N_output )).to(device)
	print('shape dep shape:', soft_dtw.shape_descriptor(targets[0,:,:], l).shape, 
       soft_dtw.shape_descriptor(targets[0,:,:], l).view(-1,1).shape)
	for k in range(batch_size):
		Ck = soft_dtw.pairwise_distances(soft_dtw.shape_descriptor(targets[k,:,:], l), #  input : N x l
                                   soft_dtw.shape_descriptor(outputs[k,:,:], l))  # output : N x N
		C[k:k+1,:,:] = Ck     
	loss_shape = softdtw_batch(C,gamma)	
	path_dtw = path_soft_dtw.PathDTWBatch.apply
	path = path_dtw(C,gamma)           
	Omega =  soft_dtw.pairwise_distances(torch.range(1,N_output).view(N_output,1)).to(device)
	loss_temporal =  torch.sum( path*Omega ) / (N_output*N_output) 
	loss = alpha*loss_shape + (1-alpha)*loss_temporal 
	return loss
#%%
def shape_dilate_loss_indep(outputs, targets, alpha, gamma, device, l = 5):
	# outputs, targets: shape (batch_size, N_output, 1)

    batch_size, N_output = outputs.shape[0:2]
    s_outputs = batch_shape_descriptor(outputs, l) #shape (batch_size, N_output, l )
    s_targets = batch_shape_descriptor(targets, l) #shape (batch_size, N_output, l )
    loss_shape = [] #--> list로?
    path = []
    loss_temporal = []
    softdtw_batch = soft_dtw.SoftDTWBatch.apply
    C = torch.zeros((batch_size, l, N_output, N_output)).to(device) # b x l x N x N
    print(s_outputs , '\n', s_targets)
    print(s_outputs[0, :, 1:2] , '\n', s_targets[0, :, 1:2])
    for k in range(batch_size):
        for i in range(l):
            Cki = soft_dtw.pairwise_distances(s_targets[k, :, i:i+1], s_outputs[k, :, i:i+1]) # input N x 1
            C[k,i:i+1,:,:] = Cki # N x N
            
    for i in range(l):
        loss_shape.append(softdtw_batch(C[:,i, :, :],gamma))
        path_dtw = path_soft_dtw.PathDTWBatch.apply
        path.append(path_dtw(C[:,i, :, :],gamma))                                                      
        Omega =  soft_dtw.pairwise_distances(torch.range(1,N_output).view(N_output,1)).to(device)
        loss_temporal.append(torch.sum(path[i]*Omega ) / (N_output*N_output))
    loss = alpha*sum(loss_shape) + (1-alpha)* sum(loss_temporal)
    return loss

#%%

#%%
x = torch.tensor(np.array([[[1], [3], [3], [1], [2], [3]]]))
y = torch.tensor(np.array([[[1], [3], [3], [1], [2], [3]]]))
# y = torch.tensor(np.array([[[1], [3], [3], [50], [2], [3]]]))
#%%
dilate_loss(x, y, 1, 0.01, 'cuda:1')
# %%
shape_dilate_loss_dep(x, y, 1, 0.01, 'cuda:1')
# %%

shape_dilate_loss_indep(x, y, 1, 0.01, 'cuda:1')



#%%
x = torch.tensor(np.array([[1], [2], [3]]))
y = torch.tensor(np.array([[2], [3], [4]]))
x.size()[0]
y.size()[0]
#%%
# %%
