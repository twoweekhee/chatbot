import base64
from typing import TypedDict

from dotenv import load_dotenv
from fastapi import UploadFile
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.constants import END
from langgraph.graph import StateGraph

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)

class ImageAnalysisState(TypedDict):
    image_base64: str
    analysis_result: str

def analyze_image(state: ImageAnalysisState) -> dict:
    print("🔍 이미지 분석 중 ")
    try:
        base64_image = state['image_base64']

        # 🔥 GPT에게 원하는 양식으로 답변하라고 명령
        prompt = "다음 이미지를 보낸 사람과 받는 사람으로 구분하여 채팅을 분석해서 채팅 내용을 뽑아줘"


        messages = [
            HumanMessage(content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ])
        ]

        response = llm.invoke(messages)
        print(f"response : {response}")

        return {
            "analysis_result": response.content
        }

    except Exception as e:
        print(f"❌ 이미지 분석 오류: {e}")
        return {
            "analysis_result": f"이미지 분석 중 오류가 발생했습니다: {str(e)}"
        }

builder = StateGraph(ImageAnalysisState)
builder.add_node("analyze", analyze_image)
builder.set_entry_point("analyze")
builder.add_edge("analyze", END)

image_analysis_graph = builder.compile()

async def run_graph(uploadFile: UploadFile) -> dict:
    # 1. UploadFile을 base64로 변환
    content = await uploadFile.read()
    base64_image = base64.b64encode(content).decode('utf-8')

    initial_content = ImageAnalysisState(image_base64=base64_image, analysis_result='')

    result = image_analysis_graph.invoke(initial_content)
    return result