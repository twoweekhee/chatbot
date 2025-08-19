import os
import ssl
from typing import Dict, Any

import io

from PIL import Image
from chromadb import Client, Settings
from easyocr import easyocr
import pytesseract
from fastapi import UploadFile

ssl._create_default_https_context = ssl._create_unverified_context

chroma_client = Client(Settings(
    persist_directory="./vector_db",  # 저장 위치
    anonymized_telemetry=False))

ocr_reader = easyocr.Reader(['ko', 'en'])

# Tesseract 경로 설정 (Mac의 경우)
import platform
if platform.system() == 'Darwin':
    # M1/M2 Mac
    if os.path.exists('/opt/homebrew/bin/tesseract'):
        pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
    # Intel Mac
    elif os.path.exists('/usr/local/bin/tesseract'):
        pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(BASE_DIR, "detected_bubbles")


async def ocr_with_tesseract(file: UploadFile) -> Dict[str, Any]:
    """
    YOLO 없이 전체 이미지에서 바로 OCR 수행 (Tesseract) - 문장 단위
    """
    print("\n=== 전체 이미지 OCR 시작 (Tesseract) ===")

    await file.seek(0)
    contents = await file.read()

    # PIL Image로 변환
    pil_image = Image.open(io.BytesIO(contents))

    if pil_image.mode != 'L':
        pil_image = pil_image.convert('L')

    custom_config = r'''
        -l kor+eng 
        --oem 3 
        --psm 11 
        -c preserve_interword_spaces=0
        -c tosp_min_sane_kn_sp=0.5
        '''

    # 방법 1: image_to_string으로 전체 텍스트 가져오기 (라인 단위)
    full_text = pytesseract.image_to_string(
        pil_image,
        lang='kor+eng',
        config=custom_config  # PSM 11: 희소 텍스트 (채팅에 적합)
    )

    # 라인별로 분리 (빈 줄 제거)
    lines = [line.strip() for line in full_text.split('\n') if line.strip()]

    # 방법 2: 상세 데이터로 라인별 그룹화
    data = pytesseract.image_to_data(
        pil_image,
        lang='kor+eng',
        output_type=pytesseract.Output.DICT,
        config='--oem 3 --psm 11'
    )

    # 라인별로 텍스트 그룹화
    line_dict = {}
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 30 and data['text'][i].strip():
            # block_num과 line_num을 키로 사용
            line_key = (data['block_num'][i], data['line_num'][i])

            if line_key not in line_dict:
                line_dict[line_key] = {
                    'words': [],
                    'confidences': [],
                    'top': data['top'][i]
                }

            line_dict[line_key]['words'].append(data['text'][i])
            line_dict[line_key]['confidences'].append(int(data['conf'][i]))

    # 라인별 텍스트 조합
    grouped_texts = []
    for line_key in sorted(line_dict.keys()):
        line_data = line_dict[line_key]
        # 단어들을 공백으로 연결
        full_line = ' '.join(line_data['words'])
        # 평균 신뢰도 계산
        avg_conf = sum(line_data['confidences']) / len(line_data['confidences']) / 100

        grouped_texts.append({
            'text': full_line,
            'confidence': avg_conf
        })
        print(f"  '{full_line}' (신뢰도: {avg_conf:.2f})")

    print(f"\n총 추출된 텍스트: {len(grouped_texts)}개")

    return {
        "method": "Direct OCR (Tesseract)",
        "text_count": len(grouped_texts),
        "texts": [item['text'] for item in grouped_texts],
        "detailed_texts": grouped_texts
    }