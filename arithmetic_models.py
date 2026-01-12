import torch.nn as nn
import torch


class MLP(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.embedding = nn.Embedding(params.p, params.embed_dim) 
        self.linear1r = nn.Linear(params.embed_dim, params.hidden_size, bias=True)
        self.linear1l = nn.Linear(params.embed_dim, params.hidden_size, bias=True)
        self.linear2 = nn.Linear(params.hidden_size, params.p, bias=False)
        self.act = nn.GELU()
        self.vocab_size = params.p

    def forward(self, x):
        x1 = self.embedding(x[..., 0]) # x[..., 0] is the first element of the pair, then embedded
        x2 = self.embedding(x[..., 1]) # x[..., 1] is the second element of the pair, then embedded
        x1 = self.linear1l(x1)
        x2 = self.linear1r(x2)
        x = x1 + x2
        x = self.act(x) # non-linear activation
        x = self.linear2(x) # linear layer to produce logits
        # No need for softmax here, as we use CrossEntropyLoss which applies softmax internally, so expects raw logits
        return x
    
class paperModel(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.p = params.p
        self.K = params.embed_dim
        self.W = nn.Linear(2*self.p, self.K, bias=False)  # Input to embedding
        self.V = nn.Linear(self.K, self.p, bias=False)    # Embedding to output
    
    def forward(self, x):
        # One-hot encode the inputs
        # x1_onehot = nn.functional.one_hot(x[..., 0], num_classes=self.p).float() 
        # x2_onehot = nn.functional.one_hot(x[..., 1], num_classes=self.p).float()
        # x_onehot = torch.cat([x1_onehot, x2_onehot], dim=-1)  # Concatenate one-hot vectors
        z = (self.W(x)).pow(2)  # Linear transformation + square activation
        Y_hat = self.V(z)       # Linear transformation to output space
        return Y_hat