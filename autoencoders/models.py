import torch
import torch.nn as nn

class LinearBlock(nn.Module):
    def __init__(self, dimension, num_layers, activation, residual=True):
        super(LinearBlock, self).__init__()
        self.residual = residual
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.Linear(dimension, dimension))
            self.layers.append(activation)

    def forward(self, x):
        residual = x
        for layer in self.layers:
            x = layer(x)
        return x + residual if self.residual else x
        
class Encoder(nn.Module):
    def __init__(self, input_dim, dims, blocks, layers, activation, residual=True):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, dims),
            *[LinearBlock(dims, layers, activation, residual) for _ in range(blocks)],
            nn.Linear(dims, 1)
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, dims, blocks, layers, activation):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(1, dims),
            *[LinearBlock(dims, layers, activation) for _ in range(blocks)],
            nn.Linear(dims, 2)
        )

    def forward(self, x):
        return self.decoder(x)
    
class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder, noise, norm=False):
        super(Autoencoder, self).__init__()
        self.noise = noise
        self.encoder = encoder
        self.decoder = decoder
        self.norm_fn = torch.nn.BatchNorm1d(1)
        self.norm = norm

    def forward(self, x):
        latent = self.norm_fn(self.encoder(x)) if self.norm else self.encoder(x)
        latent += torch.normal(0, self.noise, size=latent.shape, device=latent.device)
        reconstructed = self.decoder(latent)
        return reconstructed

    # wrote this but default init seems to work better lol
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
