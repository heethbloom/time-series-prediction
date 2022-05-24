from torch.nn.modules import Module
# from torch.nn.functional import F
from torch import Tensor
# from torch.nn import _reduction as _Reduction
from typing import Callable, Optional
from dloss.dilate_loss import dilate_loss, dilate_mse_loss,tdi_mse_loss, dilate_div_loss, weighted_dilate_loss, derivative_dilate_loss, w_derivative_dilate_loss, exp_weighted_dilate_loss

# class _Loss(Module):
#     reduction: str

#     def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
#         super(_Loss, self).__init__()
#         if size_average is not None or reduce is not None:
#             self.reduction: str = _Reduction.legacy_get_string(size_average, reduce)
#         else:
#             self.reduction = reduction
            
class DILATEloss(Module):
    def __init__(self, alpha=1, gamma = 0.01, device= 'cuda:3') -> None:
        super(DILATEloss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.device = device

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return dilate_loss(input, target,  alpha=self.alpha, gamma= self.gamma, device= self.device) 

class DILATEMSEloss(Module):
    def __init__(self, alpha=1, gamma = 0.01, device= 'cuda:3') -> None:
        super(DILATEMSEloss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.device = device

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return dilate_mse_loss(input, target,  alpha=self.alpha, gamma= self.gamma, device= self.device) 

class TDIMSEloss(Module):
    def __init__(self, alpha=1, gamma = 0.01, device= 'cuda:3') -> None:
        super(TDIMSEloss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.device = device

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return tdi_mse_loss(input, target,  alpha=self.alpha, gamma= self.gamma, device= self.device) 

class DILATEDIVloss(Module):
    def __init__(self, alpha=1, gamma = 0.01, device= 'cuda:3') -> None:
        super(DILATEDIVloss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.device = device

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return dilate_div_loss(input, target,  alpha=self.alpha, gamma= self.gamma, device= self.device)
    

class WeightedDILATEloss(Module):
    def __init__(self, alpha=1, gamma = 0.01, device= 'cuda:3', g = 0.05) -> None:
        super(WeightedDILATEloss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.device = device
        self.g = g

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return weighted_dilate_loss(input, target, alpha=self.alpha, g = self.g, gamma= self.gamma, device= self.device)

class EXPWeightedDILATEloss(Module):
    def __init__(self, alpha=1, gamma = 0.01, device= 'cuda:3', g = 0.05) -> None:
        super(EXPWeightedDILATEloss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.device = device
        self.g = g

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return exp_weighted_dilate_loss(input, target, alpha=self.alpha, g = self.g, gamma= self.gamma, device= self.device)


class DerivativeDILATEloss(Module):
    def __init__(self, alpha=1, gamma = 0.01, device= 'cuda:3') -> None:
        super(DerivativeDILATEloss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.device = device

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return derivative_dilate_loss(input, target, alpha=self.alpha, gamma= self.gamma, device= self.device) 
    
class WDerivativeDILATEloss(Module):
    def __init__(self, alpha=1, beta=0.1, gamma = 0.01, device= 'cuda:3') -> None:
        super(WDerivativeDILATEloss, self).__init__()
        self.alpha = alpha
        self.beta= beta
        self.gamma = gamma
        self.device = device

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return w_derivative_dilate_loss(input, target, alpha=self.alpha,beta=self.beta, gamma= self.gamma, device= self.device)       