import os
import pickle
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

class CloudManager:
    def __init__(self):

        self.SCOPES = ['https://www.googleapis.com/auth/drive.file']
        self.creds = None
        self.token_path = 'token.pickle'
        self.credentials_path = 'credentials.json'

        self._authenticate()
        self.service = build('drive', 'v3', credentials=self.creds)

    def _authenticate(self):

        if os.path.exists(self.token_path):
            with open(self.token_path, 'rb') as token:
                self.creds = pickle.load(token)


        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                if not os.path.exists(self.credentials_path):
                    raise FileNotFoundError(f"فایل {self.credentials_path} یافت نشد! لطفاً فایل را در مسیر پروژه قرار دهید.")
                flow = InstalledAppFlow.from_client_secrets_file(self.credentials_path, self.SCOPES)
                self.creds = flow.run_local_server(port=0)


            with open(self.token_path, 'wb') as token:
                pickle.dump(self.creds, token)

    def upload_to_drive(self, file_path):
        if not os.path.exists(file_path):
            return f"خطا: فایل {file_path} یافت نشد!"
        try:

            file_metadata = {
                'name': os.path.basename(file_path),  
                'mimeType': 'application/octet-stream'
            }
            media = MediaFileUpload(file_path)

           
            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id, webViewLink'
            ).execute()

            
            file_link = file.get('webViewLink')
            return f"فایل به Google Drive آپلود شد: {file_path}\nلینک فایل: {file_link}"
        except Exception as e:
            return f"خطا در آپلود به Google Drive: {str(e)}"