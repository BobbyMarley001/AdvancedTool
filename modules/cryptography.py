from cryptography.fernet import Fernet
import os

class Cryptography:
    def __init__(self):
        self.key_path = "key.key"
        if os.path.exists(self.key_path):
            with open(self.key_path, "rb") as key_file:
                self.key = key_file.read()
        else:
            self.key = Fernet.generate_key()
            with open(self.key_path, "wb") as key_file:
                key_file.write(self.key)
        self.cipher = Fernet(self.key)

    def encrypt_file(self, input_file, output_file):
        if not os.path.exists(input_file):
            return f"خطا: فایل ورودی {input_file} یافت نشد!"
        try:
            with open(input_file, "rb") as f:
                data = f.read()
            encrypted_data = self.cipher.encrypt(data)
            with open(output_file, "wb") as f:
                f.write(encrypted_data)
            return f"فایل رمزنگاری شد و در {output_file} ذخیره شد"
        except Exception as e:
            return f"خطا در رمزنگاری: {str(e)}"

    def decrypt_file(self, input_file, output_file):
        if not os.path.exists(input_file):
            return f"خطا: فایل ورودی {input_file} یافت نشد!"
        if not os.path.exists(self.key_path):
            return f"خطا: فایل کلید {self.key_path} یافت نشد!"
        try:
            with open(input_file, "rb") as f:
                encrypted_data = f.read()
            decrypted_data = self.cipher.decrypt(encrypted_data)
            with open(output_file, "wb") as f:
                f.write(decrypted_data)
            return f"فایل رمزگشایی شد و در {output_file} ذخیره شد"
        except Exception as e:
            return f"خطا در رمزگشایی: {str(e)}"