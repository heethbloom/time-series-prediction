from pydoc import tempfilepager
import torch
from . import soft_dtw
from . import path_soft_dtw 
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

def dilate_mse_loss(outputs, targets, alpha, gamma, device):
	# outputs, targets: shape (batch_size, N_output, 1)
	batch_size, N_output = outputs.shape[0:2]
	loss_shape = 0
	softdtw_batch = soft_dtw.SoftDTWBatch.apply
	D = torch.zeros((batch_size, N_output,N_output )).to(device)
	for k in range(batch_size):
		Dk = soft_dtw.pairwise_distances(targets[k,:,:].view(-1,1),outputs[k,:,:].view(-1,1))
		D[k:k+1,:,:] = Dk     
	loss_shape = softdtw_batch(D,gamma)	
 
	path_dtw = path_soft_dtw.PathDTWBatch.apply
	path = path_dtw(D,gamma)           
	Omega =  soft_dtw.pairwise_distances(torch.range(1,N_output).view(N_output,1)).to(device)
	loss_temporal =  torch.sum( path*Omega ) / (N_output*N_output) 
	
	mse_loss_function = torch.nn.MSELoss()
	mse_loss = mse_loss_function(outputs, targets)
	loss = (alpha*loss_shape + (1-alpha)*loss_temporal)*0.5 + mse_loss*0.5
	return loss

def tdi_mse_loss(outputs, targets, alpha, gamma, device):
	# outputs, targets: shape (batch_size, N_output, 1)
	batch_size, N_output = outputs.shape[0:2]
	loss_shape = 0
	softdtw_batch = soft_dtw.SoftDTWBatch.apply
	D = torch.zeros((batch_size, N_output,N_output )).to(device)
	for k in range(batch_size):
		Dk = soft_dtw.pairwise_distances(targets[k,:,:].view(-1,1),outputs[k,:,:].view(-1,1))
		D[k:k+1,:,:] = Dk     
  
	loss_shape = softdtw_batch(D,gamma)	
	path_dtw = path_soft_dtw.PathDTWBatch.apply
	path = path_dtw(D,gamma)           
	Omega =  soft_dtw.pairwise_distances(torch.range(1,N_output).view(N_output,1)).to(device)
	loss_temporal =  torch.sum( path*Omega ) / (N_output*N_output) 
	
	mse_loss_function = torch.nn.MSELoss()
	mse_loss = mse_loss_function(outputs, targets)
	loss =  (1-alpha)*loss_temporal + alpha*mse_loss
	return loss


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
	# loss_temporal =  torch.sum(path*Omega) / (N_output*N_output)
	loss_temporal =  (torch.sum(path*Omega) - torch.sum(pathXX*Omega)/2 - torch.sum(pathYY*Omega)/2) / (N_output*N_output) 
	loss = alpha*loss_shape + (1-alpha)*loss_temporal 
	return loss


def exp_weighted_dilate_loss(outputs, targets, g , alpha, gamma, device):
	# outputs, targets: shape (batch_size, N_output, 1)
	batch_size, N_output = outputs.shape[0:2]
	loss_shape = 0
	n = targets.shape[1]
	m = outputs.shape[1]
	w = torch.zeros((n, m)).to(device)
	for i in range(n):
		for j in range(m):
			w[i][j]  =  torch.tensor(1 / (pow((1 + np.exp(-g * (np.abs(i-j)-n/2))),2)))
	softdtw_batch = soft_dtw.SoftDTWBatch.apply
	C = torch.zeros((batch_size, N_output,N_output )).to(device)

	f = torch.full((outputs.shape[0],N_output+1, outputs.shape[2]),0.0).to(device)
	f[:,1, :] = outputs[:,0,:]
	for t in range(2,N_output):  
		f[:,t,:] = 0.4*outputs[:,t-1,:] + 0.6*f[:,t-1,:] 
	f[:,N_output,:] = f[:,N_output-1,:]
	outputs = f[:,1:,:]
	
	g = torch.full((targets.shape[0],N_output+1, targets.shape[2]),0.0).to(device)
	g[:,1, :] = targets[:,0,:]
	for t in range(2, N_output):  
		g[:,t,:] = 0.4*targets[:,t-1,:] + 0.6*g[:,t-1,:] 
	g[:,N_output,:] = g[:,N_output-1,:]
	targets = g[:,1:,:]
 
	for k in range(batch_size):
		Ck = soft_dtw.pairwise_distances(targets[k,:,:].view(-1,1),outputs[k,:,:].view(-1,1))
		C[k:k+1,:,:] = w * Ck     
	
	loss_shape = softdtw_batch(C,gamma)	
	path_dtw = path_soft_dtw.PathDTWBatch.apply
	path = path_dtw(C,gamma)           
	Omega =  soft_dtw.pairwise_distances(torch.range(1,N_output).view(N_output,1)).to(device)
	loss_temporal =  torch.sum( path*Omega ) / (N_output*N_output) 
	loss = alpha*loss_shape + (1-alpha)*loss_temporal 
	return loss

def weighted_dilate_loss(outputs, targets, g , alpha, gamma, device):
	# outputs, targets: shape (batch_size, N_output, 1)
	batch_size, N_output = outputs.shape[0:2]
	loss_shape = 0
	n = targets.shape[1]
	m = outputs.shape[1]
	w = torch.zeros((n, m)).to(device)
	for i in range(n):
		for j in range(m):
			w[i][j]  =  torch.tensor(1 / (pow((1 + np.exp(-g * (np.abs(i-j)-n/2))),2)))
	softdtw_batch = soft_dtw.SoftDTWBatch.apply
	C = torch.zeros((batch_size, N_output,N_output )).to(device)
 
	for k in range(batch_size):
		Ck = soft_dtw.pairwise_distances(targets[k,:,:].view(-1,1),outputs[k,:,:].view(-1,1))
		C[k:k+1,:,:] = w * Ck     
	
	loss_shape = softdtw_batch(C,gamma)	
	path_dtw = path_soft_dtw.PathDTWBatch.apply
	path = path_dtw(C,gamma)           
	Omega =  soft_dtw.pairwise_distances(torch.range(1,N_output).view(N_output,1)).to(device)
	loss_temporal =  torch.sum( path*Omega ) / (N_output*N_output) 
	loss = alpha*loss_shape + (1-alpha)*loss_temporal 
	return loss




# def derivative_dilate_loss(outputs, targets,  alpha, gamma, device):
# 	# outputs, targets: shape (batch_size, N_output, 1)
# 	batch_size, N_output = outputs.shape[0:2]
# 	loss_shape = 0
# 	softdtw_batch = soft_dtw.SoftDTWBatch.apply
# 	C = torch.zeros((batch_size, N_output-2,N_output-2)).to(device)
 
# 	# f = torch.full((outputs.shape[0],N_output+1, outputs.shape[2]),0.0).to(device)
# 	# f[:,1, :] = outputs[:,0,:]
# 	# for t in range(2,N_output):  
# 	# 	f[:,t,:] = 0.4*outputs[:,t-1,:] + 0.6*f[:,t-1,:] 
# 	# f[:,N_output,:] = f[:,N_output-1,:]
# 	# outputs = f[:,1:,:]
	
# 	# g = torch.full((targets.shape[0],N_output+1, targets.shape[2]),0.0).to(device)
# 	# g[:,1, :] = targets[:,0,:]
# 	# for t in range(2, N_output):  
# 	# 	g[:,t,:] = 0.4*targets[:,t-1,:] + 0.6*g[:,t-1,:] 
# 	# g[:,N_output,:] = g[:,N_output-1,:]
# 	# targets = g[:,1:,:]
 
 
# 	for i in range(1,N_output-1):
# 		targets[:,i,:] = ((targets[:,i,:] - targets[:,i-1,:]) + ((targets[:,i+ 1,:] - targets[:,i-1,:])/2))/2
# 		outputs[:,i,:] = ((outputs[:,i,:] - outputs[:,i-1,:]) + ((outputs[:,i+ 1,:] - outputs[:,i-1,:])/2))/2
# 	# targets[:,0,:] =  targets[:,1,:]
# 	# targets[:,N_output-1,:] = targets[:,N_output-2,:]
# 	# outputs[:,0,:] =  outputs[:,1,:]
# 	# outputs[:,N_output-1,:] = outputs[:,N_output-2,:]
# 	targets = targets[:,1:N_output-1, :]
# 	outputs = outputs[:,1:N_output-1, :]
		
# 	for k in range(batch_size):
# 		Ck = soft_dtw.pairwise_distances(targets[k,:,:].view(-1,1),outputs[k,:,:].view(-1,1))
# 		C[k:k+1,:,:] = Ck     
# 	loss_shape = softdtw_batch(C,gamma)	

# 	path_dtw = path_soft_dtw.PathDTWBatch.apply
# 	path = path_dtw(C,gamma)           
# 	Omega =  soft_dtw.pairwise_distances(torch.range(1,N_output-2).view(N_output-2,1)).to(device)
# 	loss_temporal =  torch.sum( path*Omega ) / ((N_output-2)*(N_output-2)) 
# 	loss = alpha*loss_shape + (1-alpha)*loss_temporal 
# 	return loss

# def derivative_dilate_loss(outputs, targets,  alpha, gamma, device):
# 	# outputs, targets: shape (batch_size, N_output, 1)
# 	batch_size, N_output = outputs.shape[0:2]
# 	loss_shape = 0
# 	softdtw_batch = soft_dtw.SoftDTWBatch.apply
# 	C = torch.zeros((batch_size, N_output,N_output)).to(device)
# 	d_targets = targets
# 	d_outputs = outputs
 
# 	for i in range(1,N_output-1):
# 		d_targets[:,i,:] = ((targets[:,i,:] - targets[:,i-1,:]) + ((targets[:,i+ 1,:] - targets[:,i-1,:])/2))/2
# 		d_outputs[:,i,:] = ((outputs[:,i,:] - outputs[:,i-1,:]) + ((outputs[:,i+ 1,:] - outputs[:,i-1,:])/2))/2
# 	d_targets[:,0,:] =  d_targets[:,1,:]
# 	d_targets[:,N_output-1,:] = d_targets[:,N_output-2,:]
# 	d_outputs[:,0,:] =  d_outputs[:,1,:]
# 	d_outputs[:,N_output-1,:] = d_outputs[:,N_output-2,:]

# 	for k in range(batch_size):
# 		Ck = soft_dtw.pairwise_distances(d_targets[k,:,:].view(-1,1),d_outputs[k,:,:].view(-1,1))
# 		C[k:k+1,:,:] = Ck     
# 	loss_shape = softdtw_batch(C,gamma)	

# 	path_dtw = path_soft_dtw.PathDTWBatch.apply
# 	path = path_dtw(C,gamma)           
# 	Omega =  soft_dtw.pairwise_distances(torch.range(1,N_output).view(N_output,1)).to(device)
# 	loss_temporal =  torch.sum( path*Omega ) / ((N_output)*(N_output)) 
 
# 	# mse_loss_function = torch.nn.MSELoss()
# 	# mse_loss = mse_loss_function(outputs, targets)
# 	# loss = (alpha*loss_shape + (1-alpha)*loss_temporal) * 0.2 +  mse_loss * 0.8
# 	loss = (alpha*loss_shape + (1-alpha)*loss_temporal)
# 	return loss

def derivative_dilate_loss(outputs, targets,  alpha, gamma, device):
	# outputs, targets: shape (batch_size, N_output, 1)
	batch_size, N_output = outputs.shape[0:2]
	loss_shape = 0
	softdtw_batch = soft_dtw.SoftDTWBatch.apply
	C = torch.zeros((batch_size, N_output,N_output)).to(device)
	d_targets = targets
	d_outputs = outputs
 
	for i in range(1,N_output-1):
		d_targets[:,i,:][1] = ((targets[:,i,:][1] - targets[:,i-1,:][1]) + ((targets[:,i+ 1,:][1] - targets[:,i-1,:][1])/2))/2
		d_outputs[:,i,:][1] = ((outputs[:,i,:][1] - outputs[:,i-1,:][1]) + ((outputs[:,i+ 1,:][1] - outputs[:,i-1,:][1])/2))/2
	d_targets[:,0,:][1] =  d_targets[:,1,:][1]
	d_targets[:,N_output-1,:][1] = d_targets[:,N_output-2,:][1]
	d_outputs[:,0,:][1] =  d_outputs[:,1,:][1]
	d_outputs[:,N_output-1,:][1] = d_outputs[:,N_output-2,:][1]

	for k in range(batch_size):
		Ck = soft_dtw.pairwise_distances(d_targets[k,:,:].view(-1,1),d_outputs[k,:,:].view(-1,1))
		C[k:k+1,:,:] = Ck     
	loss_shape = softdtw_batch(C,gamma)	

	path_dtw = path_soft_dtw.PathDTWBatch.apply
	path = path_dtw(C,gamma)           
	Omega =  soft_dtw.pairwise_distances(torch.range(1,N_output).view(N_output,1)).to(device)
	loss_temporal =  torch.sum( path*Omega ) / ((N_output)*(N_output)) 
 
	# mse_loss_function = torch.nn.MSELoss()
	# mse_loss = mse_loss_function(outputs, targets)
	# loss = (alpha*loss_shape + (1-alpha)*loss_temporal) * 0.2 +  mse_loss * 0.8
	loss = (alpha*loss_shape + (1-alpha)*loss_temporal)
	return loss


def w_derivative_dilate_loss(outputs, targets,  alpha, beta, gamma, device):
	# outputs, targets: shape (batch_size, N_output, 1)
	batch_size, N_output = outputs.shape[0:2]
	loss_shape = 0
	n = targets.shape[1]
	m = outputs.shape[1]
	softdtw_batch = soft_dtw.SoftDTWBatch.apply
	C = torch.zeros((batch_size, N_output,N_output )).to(device)
	dC = torch.zeros((batch_size, N_output,N_output )).to(device)
	d_targets = targets
	d_outputs = outputs
 
	for i in range(1,N_output-1):
		d_targets[:,i,:][1] = ((targets[:,i,:][1] - targets[:,i-1,:][1]) + ((targets[:,i+ 1,:][1] - targets[:,i-1,:][1])/2))/2
		d_outputs[:,i,:][1] = ((outputs[:,i,:][1] - outputs[:,i-1,:][1]) + ((outputs[:,i+ 1,:][1] - outputs[:,i-1,:][1])/2))/2
	d_targets[:,0,:][1] =  d_targets[:,1,:][1]
	d_targets[:,N_output-1,:][1] = d_targets[:,N_output-2,:][1]
	d_outputs[:,0,:][1] =  d_outputs[:,1,:][1]
	d_outputs[:,N_output-1,:][1] = d_outputs[:,N_output-2,:][1]

		
	for k in range(batch_size):
		Ck = soft_dtw.pairwise_distances(targets[k,:,:].view(-1,1),outputs[k,:,:].view(-1,1))
		C[k:k+1,:,:] = Ck     
	loss_shape = softdtw_batch(C,gamma)	

	for k in range(batch_size):
		dCk = soft_dtw.pairwise_distances(d_targets[k,:,:].view(-1,1),d_outputs[k,:,:].view(-1,1))
		dC[k:k+1,:,:] = dCk    	
	loss_shape_d = softdtw_batch(dC,gamma)	
 
	path_dtw = path_soft_dtw.PathDTWBatch.apply
	path = path_dtw(C,gamma)       
	
	Omega =  soft_dtw.pairwise_distances(torch.range(1,N_output).view(N_output,1)).to(device)
	loss_temporal =  torch.sum( path*Omega ) / (N_output*N_output) 
	path_d = path_dtw(dC,gamma)
	Omega_d =  soft_dtw.pairwise_distances(torch.range(1,N_output).view(N_output,1)).to(device)
	loss_temporal_d =  torch.sum( path_d*Omega_d ) / (N_output*N_output) 
 
	loss = (alpha*loss_shape + (1-alpha)*loss_temporal) *(1-beta)+  (alpha*loss_shape_d + (1-alpha)*loss_temporal_d) * (beta)
	return loss

