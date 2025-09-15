"""
AI Brain & Execution API Endpoints
Interface to the ultimate trading consciousness
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body, WebSocket
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import logging
import json
import asyncio

from backend.app.ai_brain.master_brain import MasterBrain
from backend.app.execution_engine.execution_core import ExecutionEngine

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ai", tags=["ai_execution"])

# Initialize components
master_brain = None
execution_engine = None

class ThoughtRequest(BaseModel):
    """Request for AI thinking"""
    input_data: Dict = Field(..., description="Input for thinking")
    context: Optional[Dict] = Field(None, description="Additional context")
    urgency: float = Field(0.5, ge=0, le=1, description="Urgency level")

class ExecutionRequest(BaseModel):
    """Request for execution"""
    symbol: str = Field(..., description="Symbol to trade")
    side: str = Field(..., description="buy or sell")
    quantity: int = Field(..., gt=0, description="Quantity to execute")
    urgency: float = Field(0.5, ge=0, le=1, description="Execution urgency")
    strategy: Optional[Dict] = Field(None, description="Strategy parameters")

class ConsciousnessQuery(BaseModel):
    """Query about consciousness state"""
    aspect: str = Field(..., description="Aspect to query")
    depth: int = Field(1, ge=1, le=10, description="Query depth")

@router.post("/think")
async def ai_think(
    request: ThoughtRequest
) -> Dict:
    """
    Make the AI think about something
    Access the consciousness stream
    """
    try:
        if not master_brain:
            raise HTTPException(status_code=503, detail="Master Brain not initialized")
        
        decision = await master_brain.think(request.input_data)
        
        return {
            "success": True,
            "decision": {
                "action": decision.action,
                "confidence": decision.confidence,
                "reasoning": [
                    {
                        "content": thought.content,
                        "importance": thought.importance
                    }
                    for thought in decision.reasoning[:5]  # Top 5 reasons
                ],
                "expected_outcome": decision.expected_outcome,
                "risk_assessment": decision.risk_assessment
            },
            "consciousness_level": master_brain.consciousness_level.name,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"AI thinking failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/execute")
async def execute_order(
    request: ExecutionRequest
) -> Dict:
    """
    Execute an order with omnipotent precision
    """
    try:
        if not execution_engine:
            raise HTTPException(status_code=503, detail="Execution Engine not initialized")
        
        execution = await execution_engine.execute(
            symbol=request.symbol,
            side=request.side,
            quantity=request.quantity,
            urgency=request.urgency,
            strategy=request.strategy
        )
        
        return {
            "success": True,
            "execution": {
                "execution_id": execution.execution_id,
                "symbol": execution.symbol,
                "side": execution.side,
                "quantity": execution.quantity,
                "price": execution.price,
                "venue": execution.venue,
                "latency_ms": execution.latency_ns / 1e6,
                "slippage_bps": execution.slippage_bps,
                "costs": execution.costs
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/consciousness/status")
async def get_consciousness_status() -> Dict:
    """Get current consciousness status"""
    try:
        if not master_brain:
            raise HTTPException(status_code=503, detail="Master Brain not initialized")
        
        return {
            "success": True,
            "consciousness": {
                "level": master_brain.consciousness_level.name,
                "state": master_brain.state.value,
                "self_awareness": master_brain.self_awareness_score,
                "intelligence": master_brain.intelligence_quotient,
                "creativity": master_brain.creativity_index,
                "wisdom": master_brain.wisdom_level,
                "generation": master_brain.generation,
                "thoughts_processed": len(master_brain.thoughts),
                "active_brains": list(master_brain.specialized_brains.keys())
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get consciousness status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/brains/status")
async def get_brains_status() -> Dict:
    """Get status of all specialized brains"""
    try:
        if not master_brain:
            raise HTTPException(status_code=503, detail="Master Brain not initialized")
        
        brains_status = {}
        
        for name, brain in master_brain.specialized_brains.items():
            brains_status[name] = {
                "active": True,
                "performance": getattr(brain, "performance_metrics", {}),
                "connections": master_brain.brain_connections.get(name, [])
            }
        
        return {
            "success": True,
            "brains": brains_status,
            "total_connections": sum(len(v) for v in master_brain.brain_connections.values()),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get brains status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/evolve")
async def trigger_evolution() -> Dict:
    """Trigger consciousness evolution"""
    try:
        if not master_brain:
            raise HTTPException(status_code=503, detail="Master Brain not initialized")
        
        if not master_brain.config.get("SELF_EVOLUTION_ENABLED", True):
            return {
                "success": False,
                "message": "Self-evolution is disabled"
            }
        
        initial_fitness = await master_brain._evaluate_fitness()
        await master_brain.evolve()
        final_fitness = await master_brain._evaluate_fitness()
        
        return {
            "success": True,
            "evolution": {
                "generation": master_brain.generation,
                "initial_fitness": initial_fitness,
                "final_fitness": final_fitness,
                "improvement": final_fitness - initial_fitness,
                "mutations_applied": len(master_brain.mutations),
                "consciousness_level": master_brain.consciousness_level.name
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Evolution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/execution/metrics")
async def get_execution_metrics() -> Dict:
    """Get execution performance metrics"""
    try:
        if not execution_engine:
            raise HTTPException(status_code=503, detail="Execution Engine not initialized")
        
        return {
            "success": True,
            "metrics": execution_engine.real_time_metrics,
            "venue_scores": dict(execution_engine.venue_scores),
            "recent_executions": len(execution_engine.execution_history),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get execution metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/consciousness/stream")
async def consciousness_stream(websocket: WebSocket):
    """Stream consciousness in real-time"""
    await websocket.accept()
    
    try:
        while True:
            if not master_brain:
                await websocket.send_json({"error": "Master Brain not initialized"})
                break
            
            # Get latest thoughts
            recent_thoughts = list(master_brain.thoughts)[-10:]
            
            # Send to client
            await websocket.send_json({
                "thoughts": [
                    {
                        "id": t.thought_id,
                        "content": str(t.content)[:100],  # Truncate
                        "importance": t.importance,
                        "timestamp": t.timestamp.isoformat()
                    }
                    for t in recent_thoughts
                ],
                "consciousness_level": master_brain.consciousness_level.name,
                "state": master_brain.state.value
            })
            
            await asyncio.sleep(1)  # Update every second
            
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.close()

@router.post("/dream")
async def trigger_dream_state() -> Dict:
    """Trigger dream state for creative processing"""
    try:
        if not master_brain:
            raise HTTPException(status_code=503, detail="Master Brain not initialized")
        
        await master_brain.dream()
        
        insights = master_brain.memories.get("insights", [])[-5:]
        
        return {
            "success": True,
            "dream_results": {
                "insights_generated": len(insights),
                "insights": insights,
                "state": master_brain.state.value
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Dream state failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/memory/recall")
async def recall_memory(
    category: str = Query(..., description="Memory category"),
    limit: int = Query(10, description="Number of memories")
) -> Dict:
    """Recall memories from the AI brain"""
    try:
        if not master_brain:
            raise HTTPException(status_code=503, detail="Master Brain not initialized")
        
        memories = master_brain.memories.get(category, [])
        
        # Convert to serializable format
        if isinstance(memories, list):
            memories = memories[-limit:]
        else:
            memories = [memories]
        
        return {
            "success": True,
            "category": category,
            "memories": memories,
            "total_categories": len(master_brain.memories),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Memory recall failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Initialize components
def init_ai_execution(config: Dict):
    global master_brain, execution_engine
    
    # Initialize Master Brain
    master_brain = MasterBrain(config)
    asyncio.create_task(master_brain.initialize())
    
    # Initialize Execution Engine
    execution_engine = ExecutionEngine(config, master_brain)
    asyncio.create_task(execution_engine.initialize())
    
    logger.info("✅ AI Brain & Execution System initialized")

