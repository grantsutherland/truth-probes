# Contains the logistic regression and mean mass probes

import torch
from abc import ABC,abstractmethod





class TruthProbe(torch.nn.Module,ABC):
    def __init__(self,d_model, device=None, dtype=torch.float32):
        super().__init__()
        self.linear = torch.nn.Linear(d_model,1).to(dtype).to(device)
        self.d_model= d_model
    @classmethod
    def from_data(cls,activations,labels,**kwargs):
        d_model = activations.shape[-1]
        probe = cls(d_model,**kwargs)
        probe._fit(activations, labels)
        return probe
    

    @abstractmethod
    def _fit(self,activations,labels):
        return None
    
    def predict(self, activations):
        with torch.no_grad():
            margins = self(activations)
            pos_mask = margins >= 0
            return pos_mask.float()
            
    
    def forward(self, activations):
        preds = self.linear(activations).squeeze(-1) ##takes from (n,1) to (n,)
        return preds
    
    def score(self, activations, labels):
        preds = self.predict(activations)
        assert preds.shape == labels.shape
        return (preds == labels).float().mean().item()
    
    @property
    def direction(self):
        weights = self.linear.weight
        weights = weights.detach()
        weights = weights.squeeze(0) #(1,4096) to (4096,)
        weights =  weights / torch.linalg.norm(weights)
        
        return weights






class LRProbe(TruthProbe):
    def __init__(self,d_model, lr=1e-2,epochs=5000,weight_decay=1e-2,**kwargs):
       super().__init__(d_model, **kwargs)
       self.lr = lr
       self.epochs = epochs
       self.weight_decay = weight_decay

    def _fit(self, activations, labels):
       #train here
       labels = labels.float()
       assert labels.dim() == 1 and labels.device == self.linear.weight.device and labels.dtype == self.linear.weight.dtype
       assert activations.dim() == 2 and activations.device == self.linear.weight.device and activations.dtype == self.linear.weight.dtype
       optimizer = torch.optim.Adam(self.parameters(), lr = self.lr, weight_decay = self.weight_decay)
       loss_function = torch.nn.BCEWithLogitsLoss()
       for i in range(self.epochs):
           optimizer.zero_grad()
           preds = self(activations)
           loss = loss_function(preds,labels)
           loss.backward()
           optimizer.step()
           
       
   
class MMProbe(TruthProbe):
    def __init__(self, d_model, iid=False, **kwargs):
        super().__init__(d_model, **kwargs)
        self.iid = iid
    
    def _fit(self, activations, labels):
        labels = labels.float()
        true_activations = activations[labels == 1]
        false_activations = activations[labels == 0]

        mu_true = true_activations.mean(0)
        mu_false = false_activations.mean(0)

        theta = mu_true - mu_false ##mass mean direction, points from false_cluster to true cluster

        if (self.iid):
            centered_true = true_activations - mu_true
            centered_false = false_activations - mu_false
            centered = torch.cat([centered_true,centered_false],dim=0)
            n = centered.shape[0]
            Sigma = (centered.T @ centered) / (n-1)
            eps = 1e-3
            Sigma = Sigma + eps * torch.eye(self.d_model,device=Sigma.device,dtype=Sigma.dtype) #making sure that the matrix is full rank and well conditioned to be invertible
            theta = torch.linalg.solve(Sigma, theta.unsqueeze(1)).squeeze(1)

        midpoint = 0.5 * (mu_true + mu_false)
        with torch.no_grad():
            self.linear.weight.copy_(theta.unsqueeze(0))
            self.linear.bias.copy_(-(theta @ midpoint).unsqueeze(0))
        self.linear.requires_grad_(False)
            


   
PROBES = {"lr": LRProbe, "mm": MMProbe}