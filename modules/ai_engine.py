class AIEngine:
    def suggest_format(self, file_path):
        if not os.path.exists(file_path):
            return "خطا: فایل یافت نشد!"
        ext = os.path.splitext(file_path)[1].lower()
        if ext in [".mp4", ".avi", ".mov"]:
            return "mp4"
        elif ext in [".jpg", ".png", ".jpeg"]:
            return "png"
        return "فرمت پیشنهادی: mp4"

    def suggest_encryption(self, file_path):
        if not os.path.exists(file_path):
            return "خطا: فایل یافت نشد!"
        return "AES-256"

    def analyze_cve(self, cve_data):
        if "critical" in cve_data.lower():
            return "پیشنهاد: به‌روزرسانی فوری نرم‌افزار"
        return "پیشنهاد: بررسی و به‌روزرسانی نرم‌افزار"