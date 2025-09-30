import torch
import numpy as np
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

class RunningMeanStd(torch.nn.Module):
    def __init__(self, epsilon=1e-2, shape=()):
        super().__init__()
        self.shape = shape
        
        self.register_buffer("_sum", torch.zeros(shape, dtype=torch.float64))
        self.register_buffer("_sumsq", torch.zeros(shape, dtype=torch.float64))
        self.register_buffer("_count", torch.tensor(epsilon, dtype=torch.float64))
    
    @property
    def mean(self):
        return (self._sum / self._count).float()
    
    @property
    def std(self):
        variance = (self._sumsq / self._count) - (self.mean ** 2)
        return torch.sqrt(torch.clamp(variance, min=1e-2))
    
    def update(self, x):
        x = torch.tensor(x, dtype=torch.float64)
        n = np.prod(self.shape)
        
        addvec = torch.cat([
            x.sum(dim=0).flatten(),
            (x**2).sum(dim=0).flatten(),
            torch.tensor([x.shape[0]], dtype=torch.float64)
        ])
        
        if MPI is not None:
            totalvec = torch.zeros_like(addvec)
            MPI.COMM_WORLD.Allreduce(addvec.numpy(), totalvec.numpy(), op=MPI.SUM)
        else:
            totalvec = addvec
        
        with torch.no_grad():
            self._sum += totalvec[:n].reshape(self.shape)
            self._sumsq += totalvec[n:2*n].reshape(self.shape)
            self._count += totalvec[2*n]
