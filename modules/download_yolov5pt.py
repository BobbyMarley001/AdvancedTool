import torch
import os


save_path = "../assets/models/yolov5_model.pt"

try:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"مدل YOLOv5 دانلود و در {save_path} ذخیره شد.")
except Exception as e:
    print(f"خطا در دانلود مدل YOLOv5: {str(e)}. لطفاً اینترنت رو چک کن.")