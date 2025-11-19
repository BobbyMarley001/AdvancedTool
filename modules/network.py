import paramiko
import ftplib
import os

class NetworkManager:
    def ftp_upload(self, file_path, host, username, password):
        if not os.path.exists(file_path):
            return f"خطا: فایل {file_path} یافت نشد!"
        try:
            ftp = ftplib.FTP(host)
            ftp.login(username, password)
            with open(file_path, "rb") as f:
                ftp.storbinary(f"STOR {os.path.basename(file_path)}", f)
            ftp.quit()
            return f"فایل از طریق FTP آپلود شد: {file_path}"
        except Exception as e:
            return f"خطا در آپلود FTP: {str(e)}"

    def ssh_exec(self, host, username, password, command):
        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(host, username=username, password=password)
            stdin, stdout, stderr = client.exec_command(command)
            output = stdout.read().decode()
            client.close()
            return f"دستور از طریق SSH اجرا شد: {command}\nخروجی: {output}"
        except Exception as e:
            return f"خطا در اجرای SSH: {str(e)}"