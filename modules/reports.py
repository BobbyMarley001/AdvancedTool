from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os

class ReportGenerator:
    def generate_pdf(self, logs):
        output_path = "report.pdf"
        try:
            c = canvas.Canvas(output_path, pagesize=letter)
            c.drawString(100, 750, "گزارش لاگ‌ها")
            y = 700
            for log in logs:
                c.drawString(100, y, f"{log['action']}: {log['file']} - {log['timestamp']}")
                y -= 20
                if y < 50:
                    c.showPage()
                    y = 750
            c.save()
            return f"گزارش PDF تولید شد: {output_path}"
        except Exception as e:
            return f"خطا در تولید PDF: {str(e)}"