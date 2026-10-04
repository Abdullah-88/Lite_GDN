import torch
import torch.nn.functional as F
from torch import nn, Tensor

class VecDyT(nn.Module):
    def __init__(self, input_shape):

        super().__init__()

        self.alpha = nn.Parameter(torch.randn(input_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x

class VecDyGeluSine(nn.Module):
    def __init__(self, input_shape):
        super().__init__()

        self.alpha = nn.Parameter(torch.randn(input_shape))
        self.beta = nn.Parameter(torch.randn(input_shape))
        self.gamma = nn.Parameter(torch.randn(1))
        self.etta = nn.Parameter(torch.randn(1))
        self.gelu = nn.GELU()

    def forward(self, x):

        x = self.gamma * self.gelu(self.alpha * x) + self.etta * torch.sin(self.beta * x)

        return x

class FFUnit(nn.Module):
    def __init__(self, dim):

        super().__init__()

        self.proj = nn.Linear(dim, dim, bias = False)
        self.modulate = VecDyGeluSine(dim)

    def forward(self, x):

        u, v = x, x

        u = self.modulate(u)
        v = self.proj(v)
        g = u * v

        return g

class Delta(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        
        self.d_model = d_model
        self.proj_source = nn.Linear(d_model, d_model, bias = False)
        self.proj_destination = nn.Linear(d_model, d_model, bias = False)
        self.to_alpha_gate = nn.Linear(d_model, 1, bias = False)
        self.to_beta_gate = nn.Linear(d_model, 1, bias = False)
       
    def forward(self, x):
        B, T, D = x.shape
        device = x.device

        M_S = x.new_zeros(B, D, D)
        
        outputs = []

        alpha = torch.sigmoid(self.to_alpha_gate(x))
        beta = torch.sigmoid(self.to_beta_gate(x))
        s = self.proj_source(x)
        d = self.proj_destination(x)
        d = F.normalize(d, p = 2, dim = -1)
       
        for t in range(T):

            s_t = s[:, t, :].unsqueeze(2)
            d_t = d[:, t, :].unsqueeze(2)
            alpha_t = alpha[:, t, :].unsqueeze(2)
            beta_t = beta[:, t, :].unsqueeze(2) 
           
            err = d_t - torch.bmm(M_S, d_t)
            dt_T = d_t.transpose(1, 2)   
            M_S = beta_t * M_S + alpha_t * torch.bmm(err, dt_T)
  
            out_t = torch.bmm(M_S, s_t) 
            outputs.append(out_t.squeeze(2))
        return torch.stack(outputs, dim = 1)

class GDN(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.proj = Delta(dim)
        self.modulate = VecDyGeluSine(dim)

    def forward(self, x):

        u, v = x, x

        u = self.modulate(u)
        v = self.proj(v)
        g = u * v

        return g

class LiteGDNBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm_1 =  VecDyT(dim)
        self.norm_2 =  VecDyT(dim)
        self.memory = GDN(dim)
        self.feedforward = FFUnit(dim)

    def forward(self, x):

        residual = x

        x = self.norm_1(x)

        x = self.memory(x)

        x = x + residual

        residual = x

        x = self.norm_2(x)

        x = self.feedforward(x)

        x = x + residual

        return x

class LiteGDN(nn.Module):
    def __init__(self, d_model, num_layers):
        super().__init__()

        self.model = nn.Sequential(
            *[LiteGDNBlock(d_model) for _ in range(num_layers)]
        )

    def forward(self, x):

        return self.model(x)