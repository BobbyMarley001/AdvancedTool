import os

class GameEngine:
    def video_to_game(self, video_path):
        if not os.path.exists(video_path):
            return f"خطا: فایل ویدئو {video_path} یافت نشد!"
        try:
            return f"ویدئو به صحنه بازی تبدیل شد: {video_path}"
        except Exception as e:
            return f"خطا در تبدیل به صحنه بازی: {str(e)}"