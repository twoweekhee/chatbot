from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from chatBot import graph  # generate_response 대신 graph를 import 합니다.

app = FastAPI()


class RequestText(BaseModel):
    text: str


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


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
