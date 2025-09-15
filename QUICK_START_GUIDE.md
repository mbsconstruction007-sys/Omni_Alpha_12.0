# 🚀 OMNI ALPHA 5.0 - QUICK START GUIDE

## ⚡ **IMMEDIATE DEPLOYMENT**

### **Option 1: PowerShell (Windows)**
```powershell
# Run the deployment script
.\scripts\run_step_10.ps1

# Or with options
.\scripts\run_step_10.ps1 -SkipInfrastructure -SkipOrchestrator
```

### **Option 2: Docker Compose**
```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f orchestrator
```

### **Option 3: Direct Python**
```bash
# Install dependencies
pip install -r requirements.txt

# Run orchestrator directly
python backend/app/orchestration/master_orchestrator.py
```

## 🔍 **VERIFY DEPLOYMENT**

### **Health Check**
```bash
curl http://localhost:9000/health
```

### **Get Metrics**
```bash
curl http://localhost:9000/metrics
```

### **WebSocket Test**
```javascript
const ws = new WebSocket('ws://localhost:9000/ws');
ws.onmessage = (event) => {
    console.log('Update:', JSON.parse(event.data));
};
```

## 📊 **MONITORING DASHBOARDS**

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Kibana**: http://localhost:5601
- **Orchestrator API**: http://localhost:9000

## 🎮 **QUICK COMMANDS**

### **Start Trading**
```python
import requests

command = {
    "action": "start_trading",
    "params": {
        "strategies": ["momentum", "arbitrage"],
        "risk_level": "moderate"
    }
}
response = requests.post("http://localhost:9000/command", json=command)
print(response.json())
```

### **Check System Status**
```python
import requests

response = requests.get("http://localhost:9000/health")
status = response.json()
print(f"System State: {status['status']}")
print(f"Uptime: {status['uptime']} seconds")
```

### **Get Performance Metrics**
```python
import requests

response = requests.get("http://localhost:9000/metrics")
metrics = response.json()
print(f"Total Trades: {metrics['total_trades']}")
print(f"Success Rate: {metrics['win_rate']}")
print(f"Consciousness Level: {metrics['consciousness_depth']}")
```

## 🛠️ **TROUBLESHOOTING**

### **Common Issues**

1. **Port Conflicts**
   ```bash
   # Check what's using port 9000
   netstat -ano | findstr :9000
   ```

2. **Docker Issues**
   ```bash
   # Restart Docker
   docker-compose down
   docker-compose up -d
   ```

3. **Python Dependencies**
   ```bash
   # Reinstall dependencies
   pip install -r requirements.txt --force-reinstall
   ```

### **Logs**
```bash
# View orchestrator logs
docker-compose logs -f orchestrator

# View all logs
docker-compose logs -f
```

## 🎯 **NEXT STEPS**

1. **Monitor Performance**: Check Grafana dashboards
2. **Customize Strategies**: Use the API to add strategies
3. **Scale System**: Deploy to Kubernetes for production
4. **Watch Evolution**: Monitor the system's self-improvement

## 📞 **SUPPORT**

- **Documentation**: See STEP10_MASTER_ORCHESTRATION_SUMMARY.md
- **Test Suite**: Run `python scripts/test_step10_orchestration.py`
- **API Docs**: Visit http://localhost:9000/docs

---

**🎉 Welcome to the future of trading! Omni Alpha 5.0 is now ALIVE!**
