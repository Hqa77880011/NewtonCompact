import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Model(nn.Module):
    def __init__(self,c):
        super().__init__();self.c=c;v=c.get('input_shape',[3])[-1];w=c.get('width',32)
        self.core=nn.Sequential(nn.Linear(v+c.get('technology_dimensions',2),w),nn.SiLU(),nn.Linear(w,w),nn.SiLU(),nn.Linear(w,2*v))
    def forward(self,b):
        voltage=b['x'].float().detach().requires_grad_(True);raw=self.core(torch.cat([voltage,b['technology'].float()],-1));current,charge=raw.chunk(2,-1)
        derivative=[]
        for j in range(current.shape[1]):
            scalar=current[:,j]+self.c.get('alpha',.1)*charge[:,j]
            derivative.append(torch.autograd.grad(scalar.sum(),voltage,create_graph=self.training,retain_graph=True)[0])
        stamp=torch.stack(derivative,1);full_current=torch.cat([current,-current.sum(-1,keepdim=True)],-1);full_charge=torch.cat([charge,-charge.sum(-1,keepdim=True)],-1)
        return {'prediction':current,'charge':charge,'terminal_current':full_current,'terminal_charge':full_charge,'stamp':stamp}
    def objective(self,b,p):
        loss=F.mse_loss(p['prediction'],b['y'].float())
        if self.c['variant']!='no_charge':loss=loss+.1*F.mse_loss(p['charge'],b['charge'].float())
        if self.c['variant']!='baseline':
            reference=b['stamp'].float();difference=p['stamp']-reference
            if self.c['variant']!='no_preconditioning':difference=torch.linalg.solve(reference+self.c.get('regularization',.001)*torch.eye(reference.shape[-1],device=reference.device),difference)
            eta=torch.linalg.matrix_norm(difference,ord=2);loss=loss+F.relu(eta-self.c.get('eta_limit',.1)).square().mean()
        return loss


def demo(seed,n,c):
    rng=np.random.default_rng(seed);x=rng.uniform(-1,1,(n,3)).astype('float32');tech=rng.uniform(0,1,(n,2)).astype('float32');factor=1+tech[:,:1]
    y=factor*x**3;charge=.1*x;stamp=np.zeros((n,3,3),np.float32)
    for j in range(3):stamp[:,j,j]=(3*factor[:,0]*x[:,j]**2)+c.get('alpha',.1)*.1
    return {'x':x,'technology':tech,'y':y.astype('float32'),'charge':charge,'stamp':stamp,'group':np.arange(n)}


def measure(p,d,c):
    reference=d['stamp'];matrix=reference+c.get('regularization',.001)*np.eye(reference.shape[-1]);eta=np.linalg.norm(np.linalg.solve(matrix,p['stamp']-reference),ord=2,axis=(-2,-1))
    return {'mean_eta':float(eta.mean()),'eta_below_one':float(np.mean(eta<1)),
            'current_conservation_error':float(abs(p['terminal_current'].sum(-1)).max()),'charge_conservation_error':float(abs(p['terminal_charge'].sum(-1)).max()),'charge_rmse':float(np.sqrt(np.mean((p['charge']-d['charge'])**2)))}
