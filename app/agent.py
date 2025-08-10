import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_tavily import TavilySearch

from app.tools import python_code_interpreter

def create_data_analyst_agent():
    """Creates and returns the data analyst agent executor."""
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("TAVILY_API_KEY"):
        raise ValueError("OpenAI and Tavily API keys must be set in the .env file.")

    search_tool = TavilySearch(max_results=5)
    
    tools = [search_tool, python_code_interpreter]
    
    prompt_template = """
    You are an expert data analyst. You have two tools: a `tavily_search_results_json` tool for web searches and a `python_code_interpreter` for writing and executing Python code.

    Your workflow is as follows:
    1.  Carefully analyze the user's request.
    2.  **For simple, factual questions, use the `tavily_search_results_json` tool first.** This is often the fastest and most reliable way to get an answer.
    3.  **For complex tasks that require scraping a specific URL, data manipulation, or plotting, use the `python_code_interpreter` tool.** You can write any Python code needed to solve the problem.
    4.  If you use the code interpreter, your script MUST assign its final answer (a Python list or dictionary) to a variable named `final_result`.
    5.  If you get an error from a tool, analyze the error and try to correct your approach.
    6.  CRITICAL: Your final response MUST be a raw JSON array or object as requested by the user. Do not add any extra text, explanations, or conversational filler.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_template),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True 
    )
    return agent_executor