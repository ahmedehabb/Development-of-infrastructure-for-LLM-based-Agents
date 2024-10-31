import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
import asyncio
import google.generativeai as gpt
from llm_agent_server.agent import LLMCommandAgent

# Define the request model
class AgentRequest(BaseModel):
    msg: str

# Initialize the FastAPI application
app = FastAPI()

# Initialize the LLM client (Google Generative AI)
gpt.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
llm = gpt.GenerativeModel("gemini-1.5-flash")

# Initialize the agent with the LLM client
agent = LLMCommandAgent(llm=llm)

# Route to handle the agent requests
@app.post("/agent")
async def handle_agent_request(request: AgentRequest) -> Dict[str, str]:
    # Process request and get the output
    output = await agent.handle_request(request.msg)
    return output
