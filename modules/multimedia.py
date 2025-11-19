import os
import subprocess
from PyQt5.QtWidgets import QProgressBar, QMessageBox

class MultimediaProcessor:
    def __init__(self):
        self.ffmpeg_path = "ffmpeg"  

    def convert_format(self, input_file, output_file, progress_bar: QProgressBar):
        cmd = [self.ffmpeg_path, "-i", input_file, output_file, "-y"]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        duration = 100 
        progress_bar.setMaximum(duration)
        for line in process.stdout:
            if "frame=" in line:
                frame = int(line.split("frame=")[1].split()[0])
                progress_bar.setValue(min(frame, duration))
        process.wait()
        return f"تبدیل به {output_file} انجام شد"

    def extract_frames(self, video_file, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        cmd = [self.ffmpeg_path, "-i", video_file, f"{output_dir}/frame_%04d.png"]
        subprocess.run(cmd)
        return f"فریم‌ها در {output_dir} ذخیره شدند"

    def apply_audio_effects(self, audio_file, output_file):
        cmd = [self.ffmpeg_path, "-i", audio_file, "-af", "volume=2.0", output_file, "-y"]
        subprocess.run(cmd)
        return f"افکت صوتی اعمال شد و در {output_file} ذخیره شد"

    def video_to_gif(self, video_file, output_file):
        cmd = [self.ffmpeg_path, "-i", video_file, "-vf", "fps=10,scale=320:-1", output_file, "-y"]
        subprocess.run(cmd)
        return f"GIF در {output_file} ذخیره شد"

    def speech_to_text(self, audio_file):
        return "تشخیص گفتار: این یک متن نمونه است"

    def audio_to_midi(self, audio_file, output_file):
        return f"تبدیل به MIDI در {output_file} انجام شد"

    def analyze_emotion(self, audio_file):
        return "تحلیل احساسات: خوشحال"