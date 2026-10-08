# models/vae.py
import torch as th
import torch.nn as nn

class SmallVAE(nn.Module):
    def __init__(self, in_ch=1, z_dim=32, img_size=28):
        super().__init__()
        self.img_size = img_size
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 2, 1), nn.ReLU(),   # H/2
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),      # H/4
            nn.Flatten(),
        )
        h_dim = img_size // 4
        self.h_dim = h_dim
        self.enc_mu = nn.Linear(64*h_dim*h_dim, z_dim)
        self.enc_logvar = nn.Linear(64*h_dim*h_dim, z_dim)
        self.dec_fc = nn.Linear(z_dim, 64*h_dim*h_dim)
        self.dec = nn.Sequential(
            nn.Unflatten(1, (64,h_dim,h_dim)),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),  # H/2
            nn.ConvTranspose2d(32, in_ch, 4, 2, 1), nn.Sigmoid()  # H
        )

    def encode(self, x):  # x in [0,1], (N,C,H,W)
        h = self.enc(x)
        mu, logvar = self.enc_mu(h), self.enc_logvar(h)
        return mu, logvar

    def sample_z(self, mu, logvar):
        std = th.exp(0.5*logvar)
        eps = th.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        x = self.dec_fc(z)
        return self.dec(x)

def load_vae(path:str=None, device="cuda", in_ch=1, z_dim=32, img_size=28):
    # Auto-detect based on path if defaults are used
    if path and "cifar10" in path and in_ch==1 and img_size==28:
        print(f"Auto-detected CIFAR-10 from path '{path}'. Setting in_ch=3, img_size=32.")
        in_ch = 3
        img_size = 32

    vae = SmallVAE(in_ch=in_ch, z_dim=z_dim, img_size=img_size).to(device)
    if path:
        state = th.load(path, map_location=device)
        vae.load_state_dict(state, strict=True)
    vae.eval()
    for p in vae.parameters(): p.requires_grad_(False)
    return vae
