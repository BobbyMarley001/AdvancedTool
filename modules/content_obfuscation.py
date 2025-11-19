import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
import exiftool
import numpy as np

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

class ContentObfuscator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = 128
        self.img_channels = 3

        cyclegan_path = "../assets/models/cyclegan_g_a2b.pth"
        if not os.path.exists(cyclegan_path):
            raise FileNotFoundError(f"مدل CycleGAN در مسیر {cyclegan_path} یافت نشد! ابتدا فایل train_cyclegan_autoencoder.py را اجرا کنید.")
        self.cyclegan = CycleGANGenerator().to(self.device)
        self.cyclegan.load_state_dict(torch.load(cyclegan_path, map_location=self.device))
        self.cyclegan.eval()

        autoencoder_path = "../assets/models/autoencoder.pth"
        if not os.path.exists(autoencoder_path):
            raise FileNotFoundError(f"مدل Autoencoder در مسیر {autoencoder_path} یافت نشد! ابتدا فایل train_cyclegan_autoencoder.py را اجرا کنید.")
        self.autoencoder = Autoencoder().to(self.device)
        self.autoencoder.load_state_dict(torch.load(autoencoder_path, map_location=self.device))
        self.autoencoder.eval()

        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.reverse_transform = transforms.Compose([
            transforms.Normalize([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]),
            transforms.Normalize([-0.5, -0.5, -0.5], [1.0, 1.0, 1.0]),
            transforms.ToPILImage()
        ])

    def obfuscate_image(self, image_path, output_path):
        if not os.path.exists(image_path):
            return f"خطا: فایل تصویر {image_path} یافت نشد!"
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                obfuscated_img = self.cyclegan(img_tensor)
                obfuscated_img = self.reverse_transform(obfuscated_img.squeeze(0))
            obfuscated_img.save(output_path)
            return f"تصویر تغییر یافت و در {output_path} ذخیره شد"
        except Exception as e:
            return f"خطا در تغییر تصویر: {str(e)}"

    def obfuscate_video(self, video_path, output_path):
        if not os.path.exists(video_path):
            return f"خطا: فایل ویدئو {video_path} یافت نشد!"
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return f"خطا: نمی‌توان ویدئو {video_path} را باز کرد!"

            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (self.img_size, self.img_size), isColor=True)

            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.img_size, self.img_size))
                frame_pil = Image.fromarray(frame)
                frame_tensor = self.transform(frame_pil)
                frames.append(frame_tensor)

            if not frames:
                cap.release()
                out.release()
                return f"خطا: ویدئو {video_path} هیچ فریمی ندارد!"

            frames_tensor = torch.stack(frames).to(self.device)
            with torch.no_grad():
                obfuscated_frames = self.cyclegan(frames_tensor)
                for obfuscated_frame in obfuscated_frames:
                    obfuscated_frame = self.reverse_transform(obfuscated_frame)
                    obfuscated_frame = np.array(obfuscated_frame)
                    obfuscated_frame = cv2.cvtColor(obfuscated_frame, cv2.COLOR_RGB2BGR)
                    out.write(obfuscated_frame)

            cap.release()
            out.release()
            cv2.destroyAllWindows()
            return f"ویدئو تغییر یافت و در {output_path} ذخیره شد"
        except Exception as e:
            return f"خطا در تغییر ویدئو: {str(e)}"

    def reconstruct_content(self, file_path):
        if not os.path.exists(file_path):
            return f"خطا: فایل {file_path} یافت نشد!"
        try:
            img = Image.open(file_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                reconstructed_img = self.autoencoder(img_tensor)
                reconstructed_img = self.reverse_transform(reconstructed_img.squeeze(0))
            output_path = "reconstructed_" + os.path.basename(file_path)
            reconstructed_img.save(output_path)
            return f"محتوا بازسازی شد و در {output_path} ذخیره شد"
        except Exception as e:
            return f"خطا در بازسازی محتوا: {str(e)}"

    def obfuscate_metadata(self, file_path):
        if not os.path.exists(file_path):
            return f"خطا: فایل {file_path} یافت نشد!"
        try:
            with exiftool.ExifToolHelper() as et:
                et.execute("-all=", file_path)
                et.set_tags(file_path, {
                    "EXIF:Artist": "Anonymous User",
                    "EXIF:Copyright": "Obfuscated Content 2025",
                    "EXIF:Software": "AdvancedTool Pro",
                    "EXIF:CreateDate": "2025:01:01 00:00:00",
                    "EXIF:GPSLatitude": "0.0",
                    "EXIF:GPSamp;GPSLongitude": "0.0",
                    "IPTC:Keywords": "obfuscated, anonymous, secure",
                    "XMP:Creator": "Unknown"
                })
            return f"متادیتا تغییر یافت: {file_path}"
        except Exception as e:
            return f"خطا در تغییر متادیتا: {str(e)}"