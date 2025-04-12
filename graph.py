from pydantic import BaseModel

from langchain_ollama import ChatOllama
from langgraph.graph import START, END, StateGraph
from langgraph.types import Send
from tavily import TavilyClient


from schemas import *
from prompts import *

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

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

def spawn_researchers(state: ReportState):
    return [Send("single_search", query) for query in state.queries]

def single_search(query:str):
    
    tavily_client = TavilyClient()
    
    results = tavily_client.search(query, max_results=5, include_raw_content=False)
    
    url = results[0]["url"]
    url_extraction = tavily_client.extract_url(url)
    
    if len(url_extraction["results"]) > 0:
        raw_content = url_extraction["results"][0]["raw_content"]
        prompt = resume_search.format(user_input=user_input, search_results=raw_content)
        
        llm_result = llm.invoke(prompt)
        
        query_results = QueryResult(
            title=results[0]["title"],
            url=url,
            resume=llm_result.content,
            raw_content=raw_content
        )
        
        return {"query_results": [query_results]}

def final_writer(state: ReportState):
    search_results = ""
    references = ""
    for i, result in enumerate(state.queries_results):
        search_results += f"[{i+1}]\n\n"
        search_results += f"Title: {result.title}\n"
        search_results += f"URL: {result.url}\n"
        search_results += f"Content: {result.resume}\n"
        search_results += f"\n\n"

        references += f"[{i+1}] - [{result.title}]({result.url})\n"

    prompt = build_final_response.format(user_input=user_input,
                                       search_results=search_results)

    llm_result = reasoning_llm.invoke(prompt)

    print(llm_result)
    final_response = llm_result.content + "\n\n References:\n" + references
    # print(final_response)
    
    return {"final_response": final_response}

builder = StateGraph(ReportState)

# Add the node
builder.add_node("build_first_query", build_first_query)
builder.add_node("single_search", single_search)
builder.add_node("final_writer", final_writer)

builder.add_edge(START, "build_first_query")
builder.add_conditional_edges(
    "build_first_query",
    spawn_researchers,
    ["single_search"]
)

builder.add_edge("single_search", "final_writer")
builder.add_edge("final_writer", END)

graph = builder.compile()

if __name__ == "__main__":
    from IPython.display import Image, display
    display(Image(graph.get_graph().draw_mermaid_png()))
    
    st.title("Local Perplexity")
    
    user_input = st.text_input("Enter your question:", value="Explain the thought process of reasoning LLMs")
    
    if st.button("Generate"):
        with st.status("Generating..."):
            response = graph.invoke({"user_input": user_input})
            st.write(response)
