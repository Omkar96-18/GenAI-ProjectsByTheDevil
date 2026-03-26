from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages

llm = ChatOllama(
    model='llama3:8b-instruct-q4_k_m'
)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)

    return {'messages': [response]}

checkpointer = InMemorySaver()
_config = {'configurable': {'thread_id': 'thread-1'}}

graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpointer)

response = chatbot.invoke({'messages': [HumanMessage(content='What is the capital of India')]}, config=_config)

print(response)

