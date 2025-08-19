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


async def ocr_with_tesseract_grouped(file: UploadFile) -> Dict[str, Any]:
    """
    YOLO 없이 전체 이미지에서 바로 OCR 수행 (Tesseract) - 좌우 위치별 그룹화
    """
    print("\n=== 전체 이미지 OCR 시작 (Tesseract) - 위치별 그룹화 ===")

    await file.seek(0)
    contents = await file.read()

    # PIL Image로 변환
    pil_image = Image.open(io.BytesIO(contents))
    image_width = pil_image.width

    if pil_image.mode != 'L':
        pil_image = pil_image.convert('L')

    custom_config = r'''
        -l kor+eng 
        --oem 3 
        --psm 11 
        -c preserve_interword_spaces=0
        -c tosp_min_sane_kn_sp=0.5
        '''

    # 상세 데이터로 위치 정보까지 추출
    data = pytesseract.image_to_data(
        pil_image,
        lang='kor+eng',
        output_type=pytesseract.Output.DICT,
        config='--oem 3 --psm 11'
    )

    # 라인별로 텍스트 그룹화 + 위치 정보 포함
    line_dict = {}
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 30 and data['text'][i].strip():
            # block_num과 line_num을 키로 사용
            line_key = (data['block_num'][i], data['line_num'][i])

            if line_key not in line_dict:
                line_dict[line_key] = {
                    'words': [],
                    'confidences': [],
                    'left_positions': [],  # x 좌표들
                    'top': data['top'][i],
                    'width': 0,
                    'height': data['height'][i]
                }

            line_dict[line_key]['words'].append(data['text'][i])
            line_dict[line_key]['confidences'].append(int(data['conf'][i]))
            line_dict[line_key]['left_positions'].append(data['left'][i])

            # 전체 width 계산 (가장 오른쪽 끝까지)
            right_edge = data['left'][i] + data['width'][i]
            current_width = right_edge - min(line_dict[line_key]['left_positions'])
            line_dict[line_key]['width'] = max(line_dict[line_key]['width'], current_width)

    # 이미지 중앙선 기준으로 좌우 구분
    center_x = image_width / 2
    left_messages = []  # 왼쪽 메시지 (상대방)
    right_messages = []  # 오른쪽 메시지 (내 메시지)

    print(f"이미지 너비: {image_width}px, 중앙선: {center_x}px")
    print("\n=== 메시지별 위치 분석 ===")

    for line_key in sorted(line_dict.keys(), key=lambda x: line_dict[x]['top']):
        line_data = line_dict[line_key]

        # 텍스트 블록의 시작 위치 (가장 왼쪽)
        block_left = min(line_data['left_positions'])
        # 텍스트 블록의 중앙 위치
        block_center = block_left + (line_data['width'] / 2)
        # 텍스트 블록의 끝 위치 (가장 오른쪽)
        block_right = block_left + line_data['width']

        # 전체 라인 텍스트
        full_line = ' '.join(line_data['words'])
        avg_conf = sum(line_data['confidences']) / len(line_data['confidences']) / 100

        message_data = {
            'text': full_line,
            'confidence': avg_conf,
            'position': {
                'left': block_left,
                'center': block_center,
                'right': block_right,
                'top': line_data['top'],
                'width': line_data['width'],
                'height': line_data['height']
            }
        }

        # 위치 기반 분류 (여러 기준 적용)
        is_right_aligned = False

        # 기준 1: 블록 중앙이 이미지 중앙보다 오른쪽에 있는가
        if block_center > center_x:
            is_right_aligned = True

        # 기준 2: 블록이 이미지 오른쪽 1/3 영역에 시작하는가 (추가 검증)
        if block_left > (image_width * 2 / 3):
            is_right_aligned = True

        # 기준 3: 블록이 이미지 왼쪽 1/3 영역에서 끝나는가 (왼쪽 확실)
        if block_right < (image_width * 1 / 3):
            is_right_aligned = False

        if is_right_aligned:
            right_messages.append(message_data)
            align_status = "오른쪽 (내 메시지)"
        else:
            left_messages.append(message_data)
            align_status = "왼쪽 (상대방)"

        print(f"  '{full_line}' -> {align_status}")
        print(f"    위치: left={block_left}, center={block_center}, right={block_right}")
        print(f"    신뢰도: {avg_conf:.2f}")
        print()

    print(f"\n=== 분류 결과 ===")
    print(f"왼쪽 메시지 (상대방): {len(left_messages)}개")
    print(f"오른쪽 메시지 (내 메시지): {len(right_messages)}개")
    print(f"총 메시지: {len(left_messages) + len(right_messages)}개")

    return {
        "method": "OCR with Position Grouping (Tesseract)",
        "image_info": {
            "width": image_width,
            "height": pil_image.height,
            "center_line": center_x
        },
        "message_counts": {
            "left_messages": len(left_messages),
            "right_messages": len(right_messages),
            "total": len(left_messages) + len(right_messages)
        },
        "left_messages": left_messages,  # 상대방 메시지
        "right_messages": right_messages,  # 내 메시지
        "all_texts": [msg['text'] for msg in left_messages + right_messages],
        "grouped_conversations": {
            "other_person": [msg['text'] for msg in left_messages],
            "me": [msg['text'] for msg in right_messages]
        }
    }
