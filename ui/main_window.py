import sys
import os
import json
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QFileDialog, QLineEdit, QLabel, QComboBox, QTextEdit)
from ..modules.content_obfuscation import ContentObfuscator
from ..modules.security_tools import SecurityTools
from ..modules.file_operations import FileOperations

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Security Tool")
        self.setGeometry(100, 100, 800, 600)

        self.content_obfuscator = ContentObfuscator()
        self.security_tools = SecurityTools()
        self.file_ops = FileOperations()

        self.load_language("fa")
        self.init_ui()

    def load_language(self, lang):
        with open(f"../languages/{lang}.json", "r", encoding="utf-8") as f:
            self.lang = json.load(f)

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        lang_layout = QHBoxLayout()
        lang_label = QLabel(self.lang["select_language"])
        lang_combo = QComboBox()
        lang_combo.addItem(self.lang["english"], "en")
        lang_combo.addItem(self.lang["persian"], "fa")
        lang_combo.currentIndexChanged.connect(self.change_language)
        lang_layout.addWidget(lang_label)
        lang_layout.addWidget(lang_combo)
        layout.addLayout(lang_layout)

        welcome_label = QLabel(self.lang["welcome_message"])
        layout.addWidget(welcome_label)

        obfuscate_image_layout = QHBoxLayout()
        self.obfuscate_image_btn = QPushButton(self.lang["obfuscate_image"])
        self.obfuscate_image_btn.clicked.connect(self.obfuscate_image)
        self.obfuscate_image_input = QLineEdit()
        self.obfuscate_image_input.setPlaceholderText(self.lang["select_file"])
        self.obfuscate_image_output = QLineEdit()
        self.obfuscate_image_output.setPlaceholderText(self.lang["output_path"])
        obfuscate_image_layout.addWidget(self.obfuscate_image_btn)
        obfuscate_image_layout.addWidget(self.obfuscate_image_input)
        obfuscate_image_layout.addWidget(self.obfuscate_image_output)
        layout.addLayout(obfuscate_image_layout)

        obfuscate_video_layout = QHBoxLayout()
        self.obfuscate_video_btn = QPushButton(self.lang["obfuscate_video"])
        self.obfuscate_video_btn.clicked.connect(self.obfuscate_video)
        self.obfuscate_video_input = QLineEdit()
        self.obfuscate_video_input.setPlaceholderText(self.lang["select_file"])
        self.obfuscate_video_output = QLineEdit()
        self.obfuscate_video_output.setPlaceholderText(self.lang["output_path"])
        obfuscate_video_layout.addWidget(self.obfuscate_video_btn)
        obfuscate_video_layout.addWidget(self.obfuscate_video_input)
        obfuscate_video_layout.addWidget(self.obfuscate_video_output)
        layout.addLayout(obfuscate_video_layout)

        reconstruct_layout = QHBoxLayout()
        self.reconstruct_btn = QPushButton(self.lang["reconstruct_content"])
        self.reconstruct_btn.clicked.connect(self.reconstruct_content)
        self.reconstruct_input = QLineEdit()
        self.reconstruct_input.setPlaceholderText(self.lang["select_file"])
        reconstruct_layout.addWidget(self.reconstruct_btn)
        reconstruct_layout.addWidget(self.reconstruct_input)
        layout.addLayout(reconstruct_layout)

        metadata_layout = QHBoxLayout()
        self.metadata_btn = QPushButton(self.lang["obfuscate_metadata"])
        self.metadata_btn.clicked.connect(self.obfuscate_metadata)
        self.metadata_input = QLineEdit()
        self.metadata_input.setPlaceholderText(self.lang["select_file"])
        metadata_layout.addWidget(self.metadata_btn)
        metadata_layout.addWidget(self.metadata_input)
        layout.addLayout(metadata_layout)

        cve_layout = QHBoxLayout()
        self.cve_btn = QPushButton(self.lang["search_cve"])
        self.cve_btn.clicked.connect(self.search_cve)
        self.cve_input = QLineEdit()
        self.cve_input.setPlaceholderText(self.lang["cve_id_label"])
        cve_layout.addWidget(self.cve_btn)
        cve_layout.addWidget(self.cve_input)
        layout.addLayout(cve_layout)

        payload_layout = QHBoxLayout()
        self.payload_btn = QPushButton(self.lang["inject_payload"])
        self.payload_btn.clicked.connect(self.inject_payload)
        self.payload_file_input = QLineEdit()
        self.payload_file_input.setPlaceholderText(self.lang["select_file"])
        self.payload_input = QLineEdit()
        self.payload_input.setPlaceholderText(self.lang["payload_label"])
        payload_layout.addWidget(self.payload_btn)
        payload_layout.addWidget(self.payload_file_input)
        payload_layout.addWidget(self.payload_input)
        layout.addLayout(payload_layout)

        malware_layout = QHBoxLayout()
        self.malware_btn = QPushButton(self.lang["scan_malware"])
        self.malware_btn.clicked.connect(self.scan_malware)
        self.malware_input = QLineEdit()
        self.malware_input.setPlaceholderText(self.lang["select_file"])
        malware_layout.addWidget(self.malware_btn)
        malware_layout.addWidget(self.malware_input)
        layout.addLayout(malware_layout)

        resize_layout = QHBoxLayout()
        self.resize_btn = QPushButton(self.lang["resize_image"])
        self.resize_btn.clicked.connect(self.resize_image)
        self.resize_input = QLineEdit()
        self.resize_input.setPlaceholderText(self.lang["select_file"])
        self.resize_size = QLineEdit()
        self.resize_size.setPlaceholderText(self.lang["size_label"])
        resize_layout.addWidget(self.resize_btn)
        resize_layout.addWidget(self.resize_input)
        resize_layout.addWidget(self.resize_size)
        layout.addLayout(resize_layout)

        ocr_layout = QHBoxLayout()
        self.ocr_btn = QPushButton(self.lang["ocr_pdf"])
        self.ocr_btn.clicked.connect(self.ocr_pdf)
        self.ocr_input = QLineEdit()
        self.ocr_input.setPlaceholderText(self.lang["select_file"])
        ocr_layout.addWidget(self.ocr_btn)
        ocr_layout.addWidget(self.ocr_input)
        layout.addLayout(ocr_layout)

        doc_layout = QHBoxLayout()
        self.doc_btn = QPushButton(self.lang["process_doc"])
        self.doc_btn.clicked.connect(self.process_doc)
        self.doc_input = QLineEdit()
        self.doc_input.setPlaceholderText(self.lang["select_file"])
        doc_layout.addWidget(self.doc_btn)
        doc_layout.addWidget(self.doc_input)
        layout.addLayout(doc_layout)

        zip_layout = QHBoxLayout()
        self.zip_btn = QPushButton(self.lang["extract_zip"])
        self.zip_btn.clicked.connect(self.extract_zip)
        self.zip_input = QLineEdit()
        self.zip_input.setPlaceholderText(self.lang["select_file"])
        zip_layout.addWidget(self.zip_btn)
        zip_layout.addWidget(self.zip_input)
        layout.addLayout(zip_layout)

        detection_layout = QHBoxLayout()
        self.detection_btn = QPushButton(self.lang["object_detection"])
        self.detection_btn.clicked.connect(self.object_detection)
        self.detection_input = QLineEdit()
        self.detection_input.setPlaceholderText(self.lang["select_file"])
        detection_layout.addWidget(self.detection_btn)
        detection_layout.addWidget(self.detection_input)
        layout.addLayout(detection_layout)

        threed_layout = QHBoxLayout()
        self.threed_btn = QPushButton(self.lang["process_3d_file"])
        self.threed_btn.clicked.connect(self.process_3d_file)
        self.threed_input = QLineEdit()
        self.threed_input.setPlaceholderText(self.lang["select_file"])
        threed_layout.addWidget(self.threed_btn)
        threed_layout.addWidget(self.threed_input)
        layout.addLayout(threed_layout)

        edit_metadata_layout = QHBoxLayout()
        self.edit_metadata_btn = QPushButton(self.lang["edit_metadata"])
        self.edit_metadata_btn.clicked.connect(self.edit_metadata)
        self.edit_metadata_input = QLineEdit()
        self.edit_metadata_input.setPlaceholderText(self.lang["select_file"])
        edit_metadata_layout.addWidget(self.edit_metadata_btn)
        edit_metadata_layout.addWidget(self.edit_metadata_input)
        layout.addLayout(edit_metadata_layout)

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        layout.addWidget(self.output_text)

    def change_language(self, index):
        lang = self.sender().itemData(index)
        self.load_language(lang)
        self.update_ui_texts()

    def update_ui_texts(self):
        self.setWindowTitle(self.lang["welcome_message"])
        self.obfuscate_image_btn.setText(self.lang["obfuscate_image"])
        self.obfuscate_video_btn.setText(self.lang["obfuscate_video"])
        self.reconstruct_btn.setText(self.lang["reconstruct_content"])
        self.metadata_btn.setText(self.lang["obfuscate_metadata"])
        self.cve_btn.setText(self.lang["search_cve"])
        self.payload_btn.setText(self.lang["inject_payload"])
        self.malware_btn.setText(self.lang["scan_malware"])
        self.resize_btn.setText(self.lang["resize_image"])
        self.ocr_btn.setText(self.lang["ocr_pdf"])
        self.doc_btn.setText(self.lang["process_doc"])
        self.zip_btn.setText(self.lang["extract_zip"])
        self.detection_btn.setText(self.lang["object_detection"])
        self.threed_btn.setText(self.lang["process_3d_file"])
        self.edit_metadata_btn.setText(self.lang["edit_metadata"])

    def obfuscate_image(self):
        input_path = QFileDialog.getOpenFileName(self, self.lang["select_file"], "", "Images (*.png *.jpg *.jpeg)")[0]
        if not input_path:
            return
        output_path = QFileDialog.getSaveFileName(self, self.lang["output_path"], "", "Images (*.png *.jpg *.jpeg)")[0]
        if not output_path:
            return
        self.obfuscate_image_input.setText(input_path)
        self.obfuscate_image_output.setText(output_path)
        result = self.content_obfuscator.obfuscate_image(input_path, output_path)
        self.output_text.append(result)

    def obfuscate_video(self):
        input_path = QFileDialog.getOpenFileName(self, self.lang["select_file"], "", "Videos (*.mp4 *.avi)")[0]
        if not input_path:
            return
        output_path = QFileDialog.getSaveFileName(self, self.lang["output_path"], "", "Videos (*.mp4 *.avi)")[0]
        if not output_path:
            return
        self.obfuscate_video_input.setText(input_path)
        self.obfuscate_video_output.setText(output_path)
        result = self.content_obfuscator.obfuscate_video(input_path, output_path)
        self.output_text.append(result)

    def reconstruct_content(self):
        input_path = QFileDialog.getOpenFileName(self, self.lang["select_file"], "", "Images (*.png *.jpg *.jpeg)")[0]
        if not input_path:
            return
        self.reconstruct_input.setText(input_path)
        result = self.content_obfuscator.reconstruct_content(input_path)
        self.output_text.append(result)

    def obfuscate_metadata(self):
        input_path = QFileDialog.getOpenFileName(self, self.lang["select_file"], "")[0]
        if not input_path:
            return
        self.metadata_input.setText(input_path)
        result = self.content_obfuscator.obfuscate_metadata(input_path)
        self.output_text.append(result)

    def search_cve(self):
        cve_id = self.cve_input.text()
        if not cve_id:
            self.output_text.append(f"{self.lang['error']}: لطفاً یک CVE ID وارد کنید!")
            return
        result = self.security_tools.search_cve(cve_id)
        self.output_text.append(result)

    def inject_payload(self):
        input_path = QFileDialog.getOpenFileName(self, self.lang["select_file"], "")[0]
        if not input_path:
            return
        payload = self.payload_input.text()
        if not payload:
            self.output_text.append(f"{self.lang['error']}: لطفاً یک پیلود وارد کنید!")
            return
        self.payload_file_input.setText(input_path)
        result = self.security_tools.inject_payload(input_path, payload)
        self.output_text.append(result)

    def scan_malware(self):
        input_path = QFileDialog.getOpenFileName(self, self.lang["select_file"], "")[0]
        if not input_path:
            return
        self.malware_input.setText(input_path)
        result = self.security_tools.scan_malware(input_path)
        self.output_text.append(result)

    def resize_image(self):
        input_path = QFileDialog.getOpenFileName(self, self.lang["select_file"], "", "Images (*.png *.jpg *.jpeg)")[0]
        if not input_path:
            return
        size_text = self.resize_size.text()
        try:
            width, height = map(int, size_text.split(","))
        except:
            self.output_text.append(f"{self.lang['error']}: اندازه باید به فرمت 'عرض,ارتفاع' باشد!")
            return
        self.resize_input.setText(input_path)
        result = self.file_ops.resize_image(input_path, (width, height))
        self.output_text.append(result)

    def ocr_pdf(self):
        input_path = QFileDialog.getOpenFileName(self, self.lang["select_file"], "", "PDF Files (*.pdf)")[0]
        if not input_path:
            return
        self.ocr_input.setText(input_path)
        result = self.file_ops.ocr_pdf(input_path)
        self.output_text.append(result)

    def process_doc(self):
        input_path = QFileDialog.getOpenFileName(self, self.lang["select_file"], "", "Word Files (*.docx)")[0]
        if not input_path:
            return
        self.doc_input.setText(input_path)
        result = self.file_ops.process_doc(input_path)
        self.output_text.append(result)

    def extract_zip(self):
        input_path = QFileDialog.getOpenFileName(self, self.lang["select_file"], "", "ZIP Files (*.zip)")[0]
        if not input_path:
            return
        self.zip_input.setText(input_path)
        result = self.file_ops.extract_zip(input_path)
        self.output_text.append(result)

    def object_detection(self):
        input_path = QFileDialog.getOpenFileName(self, self.lang["select_file"], "", "Images (*.png *.jpg *.jpeg)")[0]
        if not input_path:
            return
        self.detection_input.setText(input_path)
        result = self.file_ops.object_detection(input_path)
        self.output_text.append(result)

    def process_3d_file(self):
        input_path = QFileDialog.getOpenFileName(self, self.lang["select_file"], "", "3D Files (*.obj)")[0]
        if not input_path:
            return
        self.threed_input.setText(input_path)
        result = self.file_ops.process_3d_file(input_path)
        self.output_text.append(result)

    def edit_metadata(self):
        input_path = QFileDialog.getOpenFileName(self, self.lang["select_file"], "")[0]
        if not input_path:
            return
        self.edit_metadata_input.setText(input_path)
        result = self.file_ops.edit_metadata(input_path)
        self.output_text.append(result)