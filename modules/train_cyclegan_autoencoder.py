import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from PIL import Image
import os

class CycleGANGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(CycleGANGenerator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.transformer = nn.Sequential(
            *[self.residual_block(256) for _ in range(6)]
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, out_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def residual_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        enc = self.encoder(x)
        trans = self.transformer(enc)
        dec = self.decoder(trans + enc)
        return dec

class CycleGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(CycleGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, padding=1),
        )

    def forward(self, x):
        return self.model(x)

class Autoencoder(nn.Module):
    def __init__(self, in_channels=3):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, in_channels, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return dec

class CustomDataset(Dataset):
    def __init__(self, domain_a_data, domain_b_data, transform=None):
        self.domain_a_data = domain_a_data
        self.domain_b_data = domain_b_data
        self.transform = transform

    def __len__(self):
        return min(len(self.domain_a_data), len(self.domain_b_data))

    def __getitem__(self, idx):
        img_a = self.domain_a_data[idx]['image'].convert('RGB')
        img_b = self.domain_b_data[idx]['image'].convert('RGB')
        if self.transform:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)
        return img_a, img_b

img_size = 128
batch_size = 16
epochs = 100
lr = 0.0002
beta1 = 0.5
lambda_cycle = 10.0

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

try:
    data_horses = load_dataset("gigant/horse2zebra", name="horse", split="train")
    data_zebras = load_dataset("gigant/horse2zebra", name="zebra", split="train")
except Exception as e:
    raise Exception(f"خطا در لود دیتاست horse2zebra: {str(e)}. لطفاً مطمئن شوید که اینترنت متصل است و کتابخانه datasets نصب شده است.")

dataset = CustomDataset(
    domain_a_data=data_horses,
    domain_b_data=data_zebras,
    transform=transform
)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G_A2B = CycleGANGenerator().to(device)
G_B2A = CycleGANGenerator().to(device)
D_A = CycleGANDiscriminator().to(device)
D_B = CycleGANDiscriminator().to(device)
autoencoder = Autoencoder().to(device)

adversarial_loss = nn.MSELoss()
cycle_loss = nn.L1Loss()
reconstruction_loss = nn.L1Loss()
optimizer_G = optim.Adam(list(G_A2B.parameters()) + list(G_B2A.parameters()), lr=lr, betas=(beta1, 0.999))
optimizer_D_A = optim.Adam(D_A.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D_B = optim.Adam(D_B.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_AE = optim.Adam(autoencoder.parameters(), lr=lr, betas=(beta1, 0.999))

for epoch in range(epochs):
    for i, (real_A, real_B) in enumerate(dataloader):
        real_A = real_A.to(device)
        real_B = real_B.to(device)
        batch_size = real_A.size(0)

        optimizer_D_A.zero_grad()
        optimizer_D_B.zero_grad()

        fake_B = G_A2B(real_A)
        fake_A = G_B2A(real_B)

        real_A_validity = D_A(real_A)
        fake_A_validity = D_A(fake_A.detach())
        real_B_validity = D_B(real_B)
        fake_B_validity = D_B(fake_B.detach())

        d_A_loss = (adversarial_loss(real_A_validity, torch.ones_like(real_A_validity)) +
                    adversarial_loss(fake_A_validity, torch.zeros_like(fake_A_validity))) / 2
        d_B_loss = (adversarial_loss(real_B_validity, torch.ones_like(real_B_validity)) +
                    adversarial_loss(fake_B_validity, torch.zeros_like(fake_B_validity))) / 2

        d_A_loss.backward()
        d_B_loss.backward()
        optimizer_D_A.step()
        optimizer_D_B.step()

        optimizer_G.zero_grad()
        fake_B = G_A2B(real_A)
        fake_A = G_B2A(real_B)
        rec_A = G_B2A(fake_B)
        rec_B = G_A2B(fake_A)

        g_A2B_loss = adversarial_loss(D_B(fake_B), torch.ones_like(D_B(fake_B)))
        g_B2A_loss = adversarial_loss(D_A(fake_A), torch.ones_like(D_A(fake_A)))
        cycle_A_loss = cycle_loss(rec_A, real_A) * lambda_cycle
        cycle_B_loss = cycle_loss(rec_B, real_B) * lambda_cycle

        g_loss = g_A2B_loss + g_B2A_loss + cycle_A_loss + cycle_B_loss
        g_loss.backward()
        optimizer_G.step()

        optimizer_AE.zero_grad()
        reconstructed_A = autoencoder(real_A)
        reconstructed_B = autoencoder(real_B)
        ae_loss = (reconstruction_loss(reconstructed_A, real_A) + reconstruction_loss(reconstructed_B, real_B)) / 2
        ae_loss.backward()
        optimizer_AE.step()

        if i % 100 == 0:
            print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] "
                  f"[D_A loss: {d_A_loss.item()}] [D_B loss: {d_B_loss.item()}] "
                  f"[G loss: {g_loss.item()}] [AE loss: {ae_loss.item()}]")

os.makedirs("../assets/models", exist_ok=True)
torch.save(G_A2B.state_dict(), "../assets/models/cyclegan_g_a2b.pth")
torch.save(G_B2A.state_dict(), "../assets/models/cyclegan_g_b2a.pth")
torch.save(autoencoder.state_dict(), "../assets/models/autoencoder.pth")