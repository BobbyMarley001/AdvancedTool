import speech_recognition as sr
from PyQt5.QtWidgets import QMessageBox
import os

class VoiceControl:
    def __init__(self, parent):
        self.parent = parent
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)

    def start_listening(self):
        try:
            with self.microphone as source:
                self.parent.log_text.append("در حال گوش دادن... لطفاً صحبت کنید!")
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)


            command = self.recognizer.recognize_google(audio, language="fa-IR").lower()
            self.parent.log_text.append(f"دستور صوتی شناسایی شد: {command}")

            return self.process_command(command)

        except sr.WaitTimeoutError:
            self.parent.log_text.append("خطا: زمان انتظار برای صحبت کردن به پایان رسید!")
            return "خطا: زمان انتظار برای صحبت کردن به پایان رسید!"
        except sr.UnknownValueError:
            self.parent.log_text.append("خطا: گفتار قابل درک نبود!")
            return "خطا: گفتار قابل درک نبود!"
        except sr.RequestError as e:
            self.parent.log_text.append(f"خطا در اتصال به سرویس تشخیص گفتار: {str(e)}")
            return f"خطا در اتصال به سرویس تشخیص گفتار: {str(e)}"
        except Exception as e:
            self.parent.log_text.append(f"خطا در کنترل صوتی: {str(e)}")
            return f"خطا در کنترل صوتی: {str(e)}"

    def process_command(self, command):

        if "چندرسانه‌ای" in command or "multimedia" in command:
            self.parent.tabs.setCurrentIndex(0)
            return "تب چندرسانه‌ای باز شد"
        elif "استگانوگرافی" in command or "steganography" in command:
            self.parent.tabs.setCurrentIndex(1)
            return "تب استگانوگرافی باز شد"
        elif "کریپتوگرافی" in command or "cryptography" in command:
            self.parent.tabs.setCurrentIndex(2)
            return "تب کریپتوگرافی باز شد"
        elif "عملیات فایل" in command or "file operations" in command:
            self.parent.tabs.setCurrentIndex(3)
            return "تب عملیات فایل باز شد"
        elif "امنیت" in command or "security" in command:
            self.parent.tabs.setCurrentIndex(4)
            return "تب امنیت باز شد"
        elif "دور زدن تشخیص" in command or "obfuscation" in command:
            self.parent.tabs.setCurrentIndex(5)
            return "تب دور زدن تشخیص باز شد"
        elif "شبکه" in command or "network" in command:
            self.parent.tabs.setCurrentIndex(6)
            return "تب شبکه باز شد"
        elif "ابری" in command or "cloud" in command:
            self.parent.tabs.setCurrentIndex(7)
            return "تب ابری باز شد"
        elif "بلاکچین" in command or "blockchain" in command:
            self.parent.tabs.setCurrentIndex(8)
            return "تب بلاکچین باز شد"
        elif "پروژه‌ها" in command or "projects" in command:
            self.parent.tabs.setCurrentIndex(9)
            return "تب پروژه‌ها باز شد"
        elif "واقعیت افزوده" in command or "augmented reality" in command:
            self.parent.tabs.setCurrentIndex(10)
            return "تب واقعیت افزوده باز شد"
        elif "کوانتومی" in command or "quantum" in command:
            self.parent.tabs.setCurrentIndex(11)
            return "تب کوانتومی باز شد"
        elif "بازی‌سازی" in command or "game development" in command:
            self.parent.tabs.setCurrentIndex(12)
            return "تب بازی‌سازی باز شد"

        elif "تم تیره" in command or "dark theme" in command:
            self.parent.theme_combo.setCurrentText("تم تیره")
            self.parent.change_theme()
            return "تم تیره اعمال شد"
        elif "تم روشن" in command or "light theme" in command:
            self.parent.theme_combo.setCurrentText("تم روشن")
            self.parent.change_theme()
            return "تم روشن اعمال شد"
        elif "تم گیمینگ" in command or "gaming theme" in command:
            self.parent.theme_combo.setCurrentText("تم گیمینگ")
            self.parent.change_theme()
            return "تم گیمینگ اعمال شد"

        elif "فارسی" in command or "persian" in command:
            self.parent.language_combo.setCurrentText("فارسی")
            self.parent.change_language()
            return "زبان به فارسی تغییر کرد"
        elif "انگلیسی" in command or "english" in command:
            self.parent.language_combo.setCurrentText("English")
            self.parent.change_language()
            return "زبان به انگلیسی تغییر کرد"
        elif "عربی" in command or "arabic" in command:
            self.parent.language_combo.setCurrentText("العربية")
            self.parent.change_language()
            return "زبان به عربی تغییر کرد"

        elif "ذخیره پروژه" in command or "save project" in command:
            self.parent.save_project()
            return "پروژه ذخیره شد"
        elif "بارگذاری پروژه" in command or "load project" in command:
            self.parent.load_project()
            return "پروژه بارگذاری شد"
        elif "تولید گزارش" in command or "generate report" in command:
            self.parent.generate_report()
            return "گزارش تولید شد"

        else:
            return f"دستور ناشناخته: {command}"