from typing import List

from fastapi import FastAPI, UploadFile
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from chatBot import graph  # generate_response 대신 graph를 import 합니다.
import imageUtils

app = FastAPI()


class RequestText(BaseModel):
    text: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # React 개발 서버
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/api/upload-multiple")
async def upload_image_vector(files: List[UploadFile]):
    for file in files:
        imageUtils.ocr_with_yolo(file)
        imageUtils.ocr_with_nothing(file)

@app.post("/hello")
async def request_gpt(request_text: RequestText):
    state = {
        "messages": [HumanMessage(content=request_text.text)]
    }
    print(f"DEBUG - Initial state: {state}")

    # generate_response(state) 대신 graph.invoke(state)를 사용합니다.
    result = graph.invoke(state)
    answer = result["messages"][-1].content  # 마지막 메시지 (GPT 응답)

    return {"response": answer}
