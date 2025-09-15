"""
Integration Manager - Connects all components together
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import aiohttp
import grpc
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class ServiceEndpoint:
    """Service endpoint configuration"""
    name: str
    host: str
    port: int
    protocol: str  # http, grpc, websocket
    health_check_path: str
    auth_required: bool

class IntegrationManager:
    """
    Manages integration between all system components
    """
    
    def __init__(self):
        self.services: Dict[str, ServiceEndpoint] = {}
        self.connections: Dict[str, Any] = {}
        self.health_status: Dict[str, bool] = {}
        
        # Define all service endpoints
        self._register_services()
    
    def _register_services(self):
        """Register all service endpoints"""
        self.services = {
            "data_pipeline": ServiceEndpoint(
                name="Data Pipeline",
                host="localhost",
                port=8001,
                protocol="http",
                health_check_path="/health",
                auth_required=False
            ),
            "strategy_engine": ServiceEndpoint(
                name="Strategy Engine",
                host="localhost",
                port=8002,
                protocol="grpc",
                health_check_path="/health",
                auth_required=True
            ),
            "risk_management": ServiceEndpoint(
                name="Risk Management",
                host="localhost",
                port=8003,
                protocol="http",
                health_check_path="/health",
                auth_required=True
            ),
            "execution_system": ServiceEndpoint(
                name="Execution System",
                host="localhost",
                port=8004,
                protocol="websocket",
                health_check_path="/health",
                auth_required=True
            ),
            "ml_platform": ServiceEndpoint(
                name="ML Platform",
                host="localhost",
                port=8005,
                protocol="http",
                health_check_path="/health",
                auth_required=False
            ),
            "monitoring": ServiceEndpoint(
                name="Monitoring System",
                host="localhost",
                port=8006,
                protocol="http",
                health_check_path="/health",
                auth_required=False
            ),
            "analytics": ServiceEndpoint(
                name="Analytics Engine",
                host="localhost",
                port=8007,
                protocol="http",
                health_check_path="/health",
                auth_required=False
            ),
            "ai_brain": ServiceEndpoint(
                name="AI Brain",
                host="localhost",
                port=8008,
                protocol="grpc",
                health_check_path="/health",
                auth_required=True
            )
        }
    
    async def connect_all(self):
        """Connect to all services"""
        for service_name, endpoint in self.services.items():
            try:
                await self.connect_service(service_name)
                logger.info(f"Connected to {service_name}")
            except Exception as e:
                logger.error(f"Failed to connect to {service_name}: {str(e)}")
    
    async def connect_service(self, service_name: str):
        """Connect to a specific service"""
        endpoint = self.services[service_name]
        
        if endpoint.protocol == "http":
            self.connections[service_name] = aiohttp.ClientSession()
        elif endpoint.protocol == "grpc":
            channel = grpc.aio.insecure_channel(
                f"{endpoint.host}:{endpoint.port}"
            )
            self.connections[service_name] = channel
        elif endpoint.protocol == "websocket":
            import websockets
            self.connections[service_name] = await websockets.connect(
                f"ws://{endpoint.host}:{endpoint.port}"
            )
    
    async def call_service(
        self, 
        service_name: str, 
        method: str, 
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call a service method"""
        if service_name not in self.connections:
            await self.connect_service(service_name)
        
        endpoint = self.services[service_name]
        connection = self.connections[service_name]
        
        if endpoint.protocol == "http":
            async with connection.post(
                f"http://{endpoint.host}:{endpoint.port}/{method}",
                json=data
            ) as response:
                return await response.json()
        
        elif endpoint.protocol == "grpc":
            # GRPC call implementation
            return {"status": "grpc_not_implemented", "data": data}
        
        elif endpoint.protocol == "websocket":
            await connection.send(json.dumps({
                "method": method,
                "data": data
            }))
            response = await connection.recv()
            return json.loads(response)
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all services"""
        health_status = {}
        
        for service_name, endpoint in self.services.items():
            try:
                if endpoint.protocol == "http":
                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                            f"http://{endpoint.host}:{endpoint.port}{endpoint.health_check_path}"
                        ) as response:
                            health_status[service_name] = response.status == 200
                else:
                    # Implement health checks for other protocols
                    health_status[service_name] = True
                    
            except Exception as e:
                logger.error(f"Health check failed for {service_name}: {str(e)}")
                health_status[service_name] = False
        
        self.health_status = health_status
        return health_status
    
    async def broadcast_event(self, event: Dict[str, Any]):
        """Broadcast an event to all services"""
        tasks = []
        for service_name in self.services:
            task = asyncio.create_task(
                self.call_service(service_name, "event", event)
            )
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def shutdown(self):
        """Shutdown all connections"""
        for service_name, connection in self.connections.items():
            try:
                endpoint = self.services[service_name]
                
                if endpoint.protocol == "http":
                    await connection.close()
                elif endpoint.protocol == "grpc":
                    await connection.close()
                elif endpoint.protocol == "websocket":
                    await connection.close()
                    
                logger.info(f"Disconnected from {service_name}")
            except Exception as e:
                logger.error(f"Error disconnecting from {service_name}: {str(e)}")
