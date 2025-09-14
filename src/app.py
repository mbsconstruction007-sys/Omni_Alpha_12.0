from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Omni Alpha API",
    description="24-Step Analysis Process API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Pydantic models
class AnalysisStep(BaseModel):
    step_id: int
    title: str
    description: str
    status: str = "pending"  # pending, in_progress, completed
    data: Optional[Dict[str, Any]] = None

class AnalysisRequest(BaseModel):
    analysis_type: str
    parameters: Optional[Dict[str, Any]] = None

class WebhookPayload(BaseModel):
    event_type: str
    data: Dict[str, Any]
    timestamp: Optional[str] = None

# In-memory storage for demo purposes
analysis_steps = []
current_analysis = None

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/api")
async def api_info():
    return {"message": "Omni Alpha API - 24-Step Analysis Process", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "omni-alpha-api"}

@app.get("/steps")
async def get_analysis_steps():
    """Get all 24 analysis steps"""
    if not analysis_steps:
        # Initialize the 24 steps
        for i in range(1, 25):
            analysis_steps.append(AnalysisStep(
                step_id=i,
                title=f"Step {i}",
                description=f"Analysis step {i} description",
                status="pending"
            ))
    return {"steps": analysis_steps}

@app.get("/steps/{step_id}")
async def get_analysis_step(step_id: int):
    """Get specific analysis step"""
    if step_id < 1 or step_id > 24:
        raise HTTPException(status_code=404, detail="Step not found")
    
    step = next((s for s in analysis_steps if s.step_id == step_id), None)
    if not step:
        raise HTTPException(status_code=404, detail="Step not found")
    
    return step

@app.post("/analysis/start")
async def start_analysis(request: AnalysisRequest):
    """Start a new analysis process"""
    global current_analysis
    current_analysis = {
        "type": request.analysis_type,
        "parameters": request.parameters,
        "current_step": 1,
        "started_at": "2024-01-01T00:00:00Z"
    }
    
    # Reset all steps to pending
    for step in analysis_steps:
        step.status = "pending"
    
    return {"message": "Analysis started", "analysis": current_analysis}

@app.post("/steps/{step_id}/complete")
async def complete_step(step_id: int, data: Optional[Dict[str, Any]] = None):
    """Mark a step as completed"""
    if step_id < 1 or step_id > 24:
        raise HTTPException(status_code=404, detail="Step not found")
    
    step = next((s for s in analysis_steps if s.step_id == step_id), None)
    if not step:
        raise HTTPException(status_code=404, detail="Step not found")
    
    step.status = "completed"
    if data:
        step.data = data
    
    return {"message": f"Step {step_id} completed", "step": step}

@app.post("/webhook")
async def webhook_endpoint(payload: WebhookPayload):
    """Webhook endpoint for bot integration"""
    # Process webhook payload
    print(f"Received webhook: {payload.event_type}")
    
    # Here you would implement your webhook logic
    return {"message": "Webhook received", "event_type": payload.event_type}

@app.get("/advice")
async def get_advice():
    """Get analysis advice and recommendations"""
    completed_steps = [s for s in analysis_steps if s.status == "completed"]
    
    advice = {
        "total_steps": 24,
        "completed_steps": len(completed_steps),
        "progress_percentage": (len(completed_steps) / 24) * 100,
        "recommendations": [
            "Continue with the next step in your analysis",
            "Review completed steps for consistency",
            "Consider external factors that might affect your analysis"
        ]
    }
    
    return advice

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
