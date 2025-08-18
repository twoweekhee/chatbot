import os
import ssl
from typing import Dict, Any
import warnings

import torch

from PIL import Image
import cv2
import numpy as np
from chromadb import Client, Settings
from easyocr import easyocr
from fastapi import UploadFile

ssl._create_default_https_context = ssl._create_unverified_context

chroma_client = Client(Settings(
    persist_directory="./vector_db",  # 저장 위치
    anonymized_telemetry=False))

ocr_reader = easyocr.Reader(['ko', 'en'])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(BASE_DIR, "detected_bubbles")

async def ocr_with_nothing(file: UploadFile) -> Dict[str, Any]:
    """
    YOLO 없이 전체 이미지에서 바로 OCR 수행
    """
    print("\n=== 전체 이미지 OCR 시작 ===")

    await file.seek(0)

    # 파일 읽기
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    # 전체 이미지에서 바로 OCR
    ocr_results = ocr_reader.readtext(img)

    all_texts = []
    for bbox, text, conf in ocr_results:
        all_texts.append(text)
        print(f"  '{text}' (신뢰도: {conf:.2f})")

    print(f"\n총 추출된 텍스트: {len(all_texts)}개")

    return {
        "method": "Direct OCR",
        "text_count": len(all_texts),
        "texts": all_texts
    }

async def quick_improvement_ocr_mps(file: UploadFile) -> Dict[str, Any]:
    """
    Mac MPS(Metal Performance Shaders) 활용 버전 - 안전한 언패킹 적용
    """
    print("\n=== Mac MPS 최적화 OCR 시작 ===")

    # 디바이스 설정
    if torch.backends.mps.is_available():
        device = 'mps'
        print("MPS 디바이스 사용")
    else:
        device = 'cpu'
        print("CPU 사용")

    await file.seek(0)
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    # 전처리 (동일)
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)

    if np.mean(img) < 127:
        print("다크모드 감지 - 이미지 반전 적용")
        img = cv2.bitwise_not(img)

    print("OCR 시작...")

    # Reader 초기화 시 경고 무시
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # MPS 사용 시 gpu=True로 설정
        reader = easyocr.Reader(
            ['ko', 'en'],
            gpu=(device == 'mps'),
            verbose=False
        )

    # readtext 호출 시에도 경고 무시
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        raw_results = reader.readtext(
            img,
            detail=1,
            paragraph=True,
            width_ths=0.7,
            height_ths=0.2,
            y_ths=0.5,
            x_ths=1.0,
            decoder='greedy',
            beamWidth=5,
            batch_size=1,
            workers=0,  # Mac에서는 0 권장
            allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz가-힣 .,!?-_()[]{}/@#$%&*+=":;'
        )

    # 안전한 언패킹 처리
    results = []
    for i, result in enumerate(raw_results):
        try:
            if isinstance(result, (list, tuple)):
                if len(result) == 3:
                    bbox, text, conf = result
                elif len(result) == 2:
                    bbox, text = result
                    conf = 1.0  # 기본 confidence 값
                    print(f"  [경고] {i}번째 결과에 confidence 누락, 기본값 1.0 사용")
                else:
                    print(f"  [경고] {i}번째 결과 형식 오류: 길이 {len(result)}")
                    continue

                results.append((bbox, text, conf))
            else:
                print(f"  [경고] {i}번째 결과 타입 오류: {type(result)}")
                continue

        except Exception as e:
            print(f"  [오류] {i}번째 결과 처리 실패: {e}")
            print(f"    원본 데이터: {result}")
            continue

    texts = []
    confidences = []
    excluded_texts = []

    print("\n추출된 텍스트:")
    for bbox, text, conf in results:
        if conf > 0.5:
            texts.append(text)
            confidences.append(conf)
            print(f"  '{text}' (신뢰도: {conf:.2f})")
        else:
            excluded_texts.append({'text': text, 'confidence': conf})
            print(f"  [제외] '{text}' (신뢰도: {conf:.2f})")

    # 통계 정보
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0

    print(f"\n=== OCR 결과 요약 ===")
    print(f"총 검출: {len(raw_results)}개")
    print(f"정상 처리: {len(results)}개")
    print(f"채택된 텍스트: {len(texts)}개")
    print(f"제외된 텍스트: {len(excluded_texts)}개")
    print(f"평균 신뢰도: {avg_confidence:.2f}")

    return {
        "method": "Quick Improvement (MPS)",
        "text_count": len(texts),
        "texts": texts,
        "device": device,
        "statistics": {
            "total_detected": len(raw_results),
            "successfully_processed": len(results),
            "accepted_texts": len(texts),
            "excluded_texts": len(excluded_texts),
            "average_confidence": avg_confidence
        },
        "excluded": excluded_texts if excluded_texts else None
    }