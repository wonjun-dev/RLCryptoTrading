import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerQnet(nn.Module):
    def __init__(self):
        super(TransformerQnet, self).__init__()
        # Transformer encoder
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=4, nhead=2, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
        self.linear_layer = nn.Linear(8, 3)

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
    src = torch.rand(5, 2, 4) # (batch, seq, dim)
    out = model(src)
    print(out)
