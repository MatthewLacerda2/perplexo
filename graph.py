from pydantic import BaseModel

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph
from langgraph.types import Send
from tavily import TavilyClient

from schemas import *
from prompts import *

from dotenv import load_dotenv
load_dotenv()

llm = ChatOllama(model="llama3.1")
reasoning_llm = ChatOllama(model="llama3.1")

def build_first_query(state: ReportState):
    class QueryList(BaseModel):
        queries: List[str]
    
    user_input = state.user_input
    prompt = build_queries.format(user_input=user_input)
    query_llm = llm.with_structured_output(QueryList)
    result = query_llm.invoke(prompt)
    return {"queries": result.queries}

builder = StateGraph(ReportState)

# Add the node
builder.add_node("build_first_query", build_first_query)

# Set the entry point instead of using "START"
builder.set_entry_point("build_first_query")

graph = builder.compile()

if __name__ == "__main__":
    
    user_input = "Explain the thought process of reasoning LLMs"
    graph.invoke({"user_input": user_input})