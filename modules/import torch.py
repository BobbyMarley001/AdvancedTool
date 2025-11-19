import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datasets import load_dataset
from PIL import Image
import os

# تعریف مدل Generator
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_channels=3, img_size=64):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, img_channels * img_size * img_size),
            nn.Tanh()
        )
        self.img_size = img_size
        self.img_channels = img_channels

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), self.img_channels, self.img_size, self.img_size)
        return img

# تعریف مدل Discriminator
class Discriminator(nn.Module):
    def __init__(self, img_channels=3, img_size=64):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_channels * img_size * img_size, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

# تنظیمات
latent_dim = 100
img_size = 64
img_channels = 3
batch_size = 64
epochs = 50
lr = 0.0002
beta1 = 0.5

# لود دیتاست (مثلاً CIFAR-10)
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

try:
    dataset = load_dataset("cifar10", split="train")
    images = [transform(Image.fromarray(img['img'])) for img in dataset]
    dataloader = DataLoader(images, batch_size=batch_size, shuffle=True)
except Exception as e:
    raise Exception(f"خطا در لود دیتاست CIFAR-10: {str(e)}. لطفاً اینترنت رو چک کن و مطمئن شو که datasets نصب شده.")

# مقداردهی اولیه مدل‌ها
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator(latent_dim, img_channels, img_size).to(device)
discriminator = Discriminator(img_channels, img_size).to(device)

# تعریف لاس و بهینه‌سازها
adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# آموزش مدل
for epoch in range(epochs):
    for i, real_imgs in enumerate(dataloader):
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0)

        # لیبل‌ها
        real_label = torch.ones(batch_size, 1).to(device)
        fake_label = torch.zeros(batch_size, 1).to(device)

        # آموزش Discriminator
        optimizer_D.zero_grad()
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_imgs = generator(z)
        real_validity = discriminator(real_imgs)
        fake_validity = discriminator(fake_imgs.detach())
        d_loss = (adversarial_loss(real_validity, real_label) + adversarial_loss(fake_validity, fake_label)) / 2
        d_loss.backward()
        optimizer_D.step()

        # آموزش Generator
        optimizer_G.zero_grad()
        fake_validity = discriminator(fake_imgs)
        g_loss = adversarial_loss(fake_validity, real_label)
        g_loss.backward()
        optimizer_G.step()

        if i % 100 == 0:
            print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] "
                  f"[D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

# ذخیره مدل
os.makedirs("../assets/models", exist_ok=True)
torch.save(generator.state_dict(), "../assets/models/gan_model.pth")
print("مدل GAN ذخیره شد: ../assets/models/gan_model.pth")