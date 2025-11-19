from PIL import Image
import os

class Steganography:
    def hide_data(self, image_path, data):
        if not os.path.exists(image_path):
            return "خطا: فایل تصویر یافت نشد!"
        img = Image.open(image_path).convert("RGB")  
        binary_data = ''.join(format(ord(i), '08b') for i in data) + '11111111' 
        pixels = img.load()
        data_index = 0
        for i in range(img.size[0]):
            for j in range(img.size[1]):
                if data_index < len(binary_data):
                    r, g, b = pixels[i, j]
                    r = (r & ~1) | int(binary_data[data_index])
                    data_index += 1
                    if data_index < len(binary_data):
                        g = (g & ~1) | int(binary_data[data_index])
                        data_index += 1
                    if data_index < len(binary_data):
                        b = (b & ~1) | int(binary_data[data_index])
                        data_index += 1
                    pixels[i, j] = (r, g, b)
                else:
                    break
            if data_index >= len(binary_data):
                break
        output_path = "stego_" + os.path.basename(image_path)
        img.save(output_path)
        return f"داده در {output_path} مخفی شد"

    def extract_data(self, image_path):
        if not os.path.exists(image_path):
            return "خطا: فایل تصویر یافت نشد!"
        img = Image.open(image_path).convert("RGB")
        pixels = img.load()
        binary_data = ""
        for i in range(img.size[0]):
            for j in range(img.size[1]):
                r, g, b = pixels[i, j]
                binary_data += str(r & 1)
                binary_data += str(g & 1)
                binary_data += str(b & 1)
        all_bytes = [binary_data[i:i+8] for i in range(0, len(binary_data), 8)]
        decoded_data = ""
        for byte in all_bytes:
            if byte == "11111111":  
                break
            decoded_data += chr(int(byte, 2))
        return f"داده استخراج‌شده: {decoded_data}"