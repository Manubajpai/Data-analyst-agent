import json
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware  
from app.agent import create_data_analyst_agent

app = FastAPI(
    title="Data Analyst Agent API",
    description="An API that uses an LLM agent to analyze data.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    agent_executor = create_data_analyst_agent()
except Exception as e:
    print(f"Startup Error: {e}")
    agent_executor = None

@app.post("/api/", tags=["Analysis"])
async def analyze_data(question_file: UploadFile = File(...)):
    if agent_executor is None:
        raise HTTPException(status_code=500, detail="Agent not initialized. Check server logs for details.")

    try:
        request_text = await question_file.read()
        user_prompt = request_text.decode("utf-8")

        response = agent_executor.invoke({"input": user_prompt})
        final_result = response.get("output")

        return final_result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")