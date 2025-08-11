import torch.nn as nn


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