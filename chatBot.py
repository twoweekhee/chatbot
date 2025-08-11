from __future__ import annotations

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain.schema import HumanMessage, AIMessage
from dotenv import load_dotenv
from typing import TypedDict, List, Annotated
from langgraph.graph.message import add_messages

load_dotenv()

# LLM 준비
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


# 🔥 LangGraph 표준 방식으로 상태 정의
class ChatState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], add_messages]


# 🔥 완전히 수정된 응답 생성 함수
def generate_response(state: ChatState) -> ChatState:
    print(f"🔍 DEBUG - Input state: {state}")

    # 메시지 가져오기
    messages = state.get("messages", [])

    print(f"🔍 DEBUG - Messages count: {len(messages)}")
    print(f"🔍 DEBUG - Messages: {messages}")

    # 빈 배열 체크
    if not messages:
        print("⚠️ WARNING: No messages found!")
        return {"messages": [AIMessage(content="안녕하세요! 무엇을 도와드릴까요?")]}

    try:
        # 🔥 llm.invoke() 사용 (새로운 방식)
        response = llm.invoke(messages)
        print(f"✅ SUCCESS - LLM response: {response.content}")

        # 기존 메시지 + 새 응답 반환
        return {"messages": [response]}  # add_messages가 자동으로 합쳐줌

    except Exception as e:
        print(f"❌ ERROR in generate_response: {e}")
        return {"messages": [AIMessage(content="죄송합니다. 오류가 발생했습니다.")]}


# 종료 조건 함수 (단순화)
def should_continue(state: ChatState) -> str:
    return END  # 한 번만 실행하고 끝


# 🔥 LangGraph 구성 (더 간단하게)
builder = StateGraph(ChatState)
builder.add_node("chat", generate_response)
builder.set_entry_point("chat")
builder.add_edge("chat", END)  # 단순한 엣지
graph = builder.compile()

# 테스트 코드 (선택사항)
if __name__ == "__main__":
    test_state = {"messages": [HumanMessage(content="안녕!")]}
    print("🧪 Testing...")
    result = graph.invoke(test_state)
    print(f"✅ Test result: {result}")