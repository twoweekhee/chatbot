import os
import ssl

from chromadb import Client, Settings
from easyocr import easyocr
from fastapi import UploadFile
from ultralytics import YOLO

ssl._create_default_https_context = ssl._create_unverified_context

chroma_client = Client(Settings(
    persist_directory="./vector_db",  # 저장 위치
    anonymized_telemetry=False))

yolo_model = YOLO("yolov8n.pt")
ocr_reader = easyocr.Reader(['ko', 'en'])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(BASE_DIR, "detected_bubbles")

def ocr_with_yolo(file: UploadFile):
    results = yolo_model(file)

    texts_yolo = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop_img = file[y1:y2, x1:x2]
        ocr_crop = ocr_reader.readtext(crop_img)
        for _, text, _ in ocr_crop:
            texts_yolo.append(text)

    print("\n=== YOLO + OCR ===")
    for idx, t in enumerate(texts_yolo, 1):
        print(f"{idx}. {t}")

def ocr_with_nothing(file: UploadFile):
    ocr_full = ocr_reader.readtext(file)

    texts_full = [text for _, text, conf in ocr_full]

    print("\n=== YOLO 없이 OCR ===")
    for idx, t in enumerate(texts_full, 1):
        print(f"{idx}. {t}")

# def save_text_vector():