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
        self.p = params.p

    def forward(self, x):
        x1 = self.embedding(torch.argmax(x[..., 0:self.p], dim=-1).long())
        x2 = self.embedding(torch.argmax(x[..., self.p:2*self.p], dim=-1).long())
        x1 = self.linear1l(x1)
        x2 = self.linear1r(x2)
        x = x1 + x2
        x = self.act(x) # non-linear activation
        x = self.linear2(x) # linear layer to produce logits
        # No need for softmax here, as we use CrossEntropyLoss which applies softmax internally, so expects raw logits
        return x

class transformerModel(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.embedding = nn.Embedding(params.p, params.embed_dim) 
        encoder_layer = nn.TransformerEncoderLayer(d_model=params.embed_dim, nhead=1, dim_feedforward=64, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.linear_out = nn.Linear(params.embed_dim, params.p, bias=False)
        self.p = params.p


    def forward(self, x):
        x1 = torch.argmax(x[..., 0:self.p], dim=-1).long()
        x2 = torch.argmax(x[..., self.p:2*self.p], dim=-1).long()
        x1 = self.embedding(x1)  # Shape: (batch_size, embed_dim)
        x2 = self.embedding(x2)  # Shape: (batch_size, embed_dim)
        x = torch.stack((x1, x2), dim=0)  # Shape: (2, batch_size, embed_dim)
        x = self.transformer_encoder(x)    # Transformer expects input shape (seq_len, batch_size, embed_dim)
        x = x.mean(dim=0)                  # Aggregate over sequence length dimension
        x = self.linear_out(x)             # Linear layer to produce logits
        return x
    
class paperModel(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.p = params.p
        self.K = params.embed_dim
        self.W = nn.Linear(2*self.p, self.K, bias=False)  # Input to embedding
        self.V = nn.Linear(self.K, self.p, bias=False)    # Embedding to output
        if params.activation == "relu":
            self.activation = nn.ReLU()
        elif params.activation == "quadratic":
            self.activation = lambda x: x.pow(2)
    
    def forward(self, x):
        # One-hot encode the inputs
        # x1_onehot = nn.functional.one_hot(x[..., 0], num_classes=self.p).float() 
        # x2_onehot = nn.functional.one_hot(x[..., 1], num_classes=self.p).float()
        # x_onehot = torch.cat([x1_onehot, x2_onehot], dim=-1)  # Concatenate one-hot vectors
        z = self.activation(self.W(x))  # Linear transformation + activation
        Y_hat = self.V(z)       # Linear transformation to output space
        return Y_hat
    
class centred_loss(nn.Module):
    """
    L = (1/2) || P_perp (Y - Yhat) ||_F^2
    where P_perp centers along the sample dimension (rows).
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        # logits, Y: (n, M)
        E = Y - logits                                  # (n, M)
        E = E - E.mean(dim=0, keepdim=True)             # apply P_perp along samples

        sq = (E * E).sum() # Frobenius norm squared
        n = logits.shape[0] # size of sample dimension
        return 0.5 * sq / n
