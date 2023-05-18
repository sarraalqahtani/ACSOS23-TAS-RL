import os
import torch
import torch.nn as nn
import torch.optim as optim

def grad_false(network):
    for param in network.parameters():
        param.requires_grad = False
        
def check_float_tensor(x):
    if not isinstance(x, torch.Tensor) or x.dtype != torch.float32:
        x = torch.tensor(x, dtype=torch.float32)
    return x

# Define the encoder network, which maps the state to a latent representation
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, obs, act):
        obs = check_float_tensor(obs)
        act = check_float_tensor(act)
        x = torch.cat([obs, act], dim=-1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, device):
        self.device = device
        self.load_state_dict(torch.load(path, map_location=device))

# Define the decoder network, which maps the latent representation back to a state
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, device):
        self.device = device
        self.load_state_dict(torch.load(path, map_location=device))

# Define the full VAE model, which consists of the encoder and decoder
class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, latent_dim):
        super(VAE, self).__init__()
        input_dim = state_dim+action_dim
        self.encoder = Encoder(input_dim, hidden_size, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_size, state_dim)

    def forward(self, state, action):
        latent = self.encoder(state, action)
        reconstructed = self.decoder(latent)
        return reconstructed
    
    def predict(self, state, action):
        state = check_float_tensor(state)
        action = check_float_tensor(action)
        with torch.no_grad():
            latent =  self.encoder(state, action)
            reconstructed_state = self.decoder(latent)
        return reconstructed_state
    
    def save(self, path):
        vae_path = os.path.join(path,"VAE")
        if not os.path.exists(vae_path):
            os.makedirs(vae_path)
        encoder_path = os.path.join(vae_path, "encoder.pth")
        decoder_path = os.path.join(vae_path, "decoder.pth")
        self.encoder.save(encoder_path)
        self.decoder.save(decoder_path)

    def load(self, path, device):
        self.device = device
        encoder_path = os.path.join(path, "encoder.pth")
        decoder_path = os.path.join(path, "decoder.pth")
        self.encoder.load(encoder_path, self.device)
        self.decoder.load(decoder_path, self.device)
        grad_false(self.encoder)
        grad_false(self.decoder)
        