import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerQnet(nn.Module):
    def __init__(self, d_model, nhead, num_layers, num_seq):
        super(TransformerQnet, self).__init__()
        # Transformer encoder
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)
        self.linear_layer = nn.Linear(d_model * num_seq, 4)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1) # concat 3hr-sequences into one vector
        out = self.linear_layer(x)
        
        return out

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(-1, 1)
        else:
            return out.argmax().item() 

if __name__ == "__main__":
    model = TransformerQnet()
    src = torch.rand(32, 24, 6) # (batch, seq, dim)
    out = model.forward(src)
    print(out, out.shape)
