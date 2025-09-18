# 🚨 OMNI ALPHA 5.0 - INCIDENT RESPONSE RUNBOOK
## Production Incident Response & Emergency Procedures

---

## 📋 **SEVERITY LEVELS & RESPONSE TIMES**

### **P0 - CRITICAL (5 minutes)**
- **Definition**: Complete system down, no trading possible, financial loss imminent
- **Examples**: Database complete failure, trading engine crash, security breach
- **Response**: Immediate escalation, all hands on deck
- **Actions**: Emergency shutdown, failover to DR, notify regulators if required

### **P1 - HIGH (15 minutes)**
- **Definition**: Critical functionality impaired, major financial risk
- **Examples**: Order execution failures, risk system down, data feed loss
- **Response**: Senior engineers, trading desk notification
- **Actions**: Implement workarounds, activate backup systems

### **P2 - MEDIUM (1 hour)**
- **Definition**: Partial functionality loss, degraded performance
- **Examples**: Slow response times, non-critical service failures
- **Response**: On-call engineer, standard escalation
- **Actions**: Performance optimization, service restart

### **P3 - LOW (4 hours)**
- **Definition**: Minor issues, no immediate impact
- **Examples**: Monitoring alerts, non-critical warnings
- **Response**: Business hours support
- **Actions**: Schedule maintenance, documentation updates

---

## 🎯 **IMMEDIATE RESPONSE CHECKLIST**

### **First 5 Minutes:**
```bash
# 1. Acknowledge alert and join incident channel
slack-cli join #incident-$(date +%Y%m%d)-001

# 2. Quick system status check
curl -s http://monitoring.omni-alpha.com/health | jq '.'

# 3. Check critical services
kubectl get pods -n production -l app=trading-engine
kubectl get svc -n production

# 4. Check database connectivity
psql -h db-primary.omni-alpha.com -U admin -d trading -c "SELECT NOW();"

# 5. Check Redis cache
redis-cli -h redis.omni-alpha.com ping

# 6. Verify trading status
curl -s http://api.omni-alpha.com/v1/trading/status
```

---

## 🔧 **SYSTEM DIAGNOSTICS**

### **Infrastructure Health Check:**
```bash
#!/bin/bash
# infrastructure_health.sh

echo "=== OMNI ALPHA 5.0 HEALTH CHECK ==="
echo "Timestamp: $(date)"
echo ""

# Check orchestrator
echo "🔍 Orchestrator Status:"
python -c "
import asyncio
from orchestrator import OmniAlphaOrchestrator
async def check():
    o = OmniAlphaOrchestrator()
    status = o.get_system_status()
    print(f'Running: {status[\"is_running\"]}')
    print(f'Uptime: {status[\"uptime_seconds\"]:.1f}s')
    print(f'Components: {len([c for c in status[\"components\"].values() if c])}/{len(status[\"components\"])} healthy')
asyncio.run(check())
"

# Check databases
echo ""
echo "🗄️ Database Status:"
echo "PostgreSQL Primary:"
pg_isready -h ${DB_PRIMARY_HOST:-localhost} -p ${DB_PRIMARY_PORT:-5432} -U ${DB_USER:-postgres}

echo "Redis:"
redis-cli -h ${REDIS_HOST:-localhost} -p ${REDIS_PORT:-6379} ping

echo "InfluxDB:"
curl -s "${INFLUXDB_URL:-http://localhost:8086}/ping"

# Check Kafka
echo ""
echo "📨 Message Queue Status:"
if command -v kafka-topics &> /dev/null; then
    kafka-topics --bootstrap-server ${KAFKA_BROKERS:-localhost:9092} --list
else
    echo "Kafka CLI not available"
fi

# Check Consul
echo ""
echo "🔍 Service Discovery:"
curl -s "${CONSUL_HOST:-localhost}:${CONSUL_PORT:-8500}/v1/status/leader"

# Check API endpoints
echo ""
echo "🌐 API Endpoints:"
curl -s -o /dev/null -w "Health endpoint: %{http_code} (%{time_total}s)\n" \
    http://localhost:8000/health

curl -s -o /dev/null -w "Metrics endpoint: %{http_code} (%{time_total}s)\n" \
    http://localhost:8001/metrics

# Check disk space
echo ""
echo "💾 Disk Usage:"
df -h / /var/log /tmp

# Check memory
echo ""
echo "🧠 Memory Usage:"
free -h

# Check load
echo ""
echo "⚡ System Load:"
uptime

echo ""
echo "=== HEALTH CHECK COMPLETE ==="
```

### **Trading System Diagnostics:**
```bash
#!/bin/bash
# trading_diagnostics.sh

echo "=== TRADING SYSTEM DIAGNOSTICS ==="

# Check Alpaca connectivity
echo "🔌 Alpaca API Status:"
python -c "
import asyncio
from data_collection.providers.alpaca_collector import get_alpaca_collector
async def check():
    collector = get_alpaca_collector()
    health = await collector.health_check()
    print(f'Status: {health[\"status\"]}')
    print(f'Message: {health[\"message\"]}')
    if 'metrics' in health:
        print(f'API Available: {health[\"metrics\"].get(\"api_available\", False)}')
asyncio.run(check())
"

# Check risk engine
echo ""
echo "🛡️ Risk Engine Status:"
python -c "
import asyncio
from risk_management.risk_engine import get_risk_engine
async def check():
    engine = get_risk_engine()
    health = await engine.health_check()
    print(f'Status: {health[\"status\"]}')
    print(f'Message: {health[\"message\"]}')
    if 'metrics' in health:
        print(f'Positions: {health[\"metrics\"].get(\"positions_count\", 0)}')
        print(f'Daily P&L: ${health[\"metrics\"].get(\"daily_pnl\", 0):.2f}')
asyncio.run(check())
"

# Check circuit breakers
echo ""
echo "⚡ Circuit Breaker Status:"
python -c "
from infrastructure.circuit_breaker import get_circuit_breaker_manager
manager = get_circuit_breaker_manager()
status = manager.get_all_status()
for name, breaker_status in status.items():
    print(f'{name}: {breaker_status[\"state\"]} (failures: {breaker_status[\"failure_count\"]})')
"

# Check active positions
echo ""
echo "📊 Current Positions:"
python -c "
from risk_management.risk_engine import get_risk_engine
engine = get_risk_engine()
summary = engine.get_risk_summary()
print(f'Portfolio Value: ${summary[\"risk_metrics\"][\"portfolio_value\"]:,.2f}')
print(f'Total Exposure: ${summary[\"risk_metrics\"][\"total_exposure\"]:,.2f}')
print(f'Leverage: {summary[\"risk_metrics\"][\"leverage\"]:.2f}x')
print(f'Risk Score: {summary[\"overall_risk_score\"]:.2f}')
"

echo ""
echo "=== DIAGNOSTICS COMPLETE ==="
```

---

## 🚨 **EMERGENCY PROCEDURES**

### **EMERGENCY SHUTDOWN (P0):**
```bash
#!/bin/bash
# emergency_shutdown.sh

echo "🚨 EMERGENCY SHUTDOWN INITIATED"
echo "Timestamp: $(date)"

# 1. Stop all trading immediately
echo "🛑 Stopping all trading..."
curl -X POST http://api.omni-alpha.com/admin/trading/emergency-stop \
    -H "Authorization: Bearer ${EMERGENCY_TOKEN}" \
    -H "Content-Type: application/json" \
    -d '{"reason": "emergency_shutdown", "initiated_by": "incident_response"}'

# 2. Close all positions (if safe to do so)
echo "📊 Initiating position closure..."
python scripts/emergency_close_positions.py --confirm

# 3. Enable all circuit breakers
echo "⚡ Activating circuit breakers..."
curl -X POST http://api.omni-alpha.com/admin/circuit-breakers/enable-all

# 4. Stop market data feeds
echo "📡 Stopping data feeds..."
curl -X POST http://api.omni-alpha.com/admin/data/stop-feeds

# 5. Notify stakeholders
echo "📢 Sending notifications..."
python scripts/emergency_notifications.py --level=P0 --message="Emergency shutdown executed"

# 6. Create incident record
echo "📝 Creating incident record..."
python scripts/create_incident.py --severity=P0 --title="Emergency Shutdown" --auto-assign

echo "✅ Emergency shutdown complete"
```

### **DATABASE FAILOVER (P0/P1):**
```bash
#!/bin/bash
# database_failover.sh

echo "🔄 DATABASE FAILOVER INITIATED"

# 1. Check primary database status
echo "🔍 Checking primary database..."
if ! pg_isready -h ${DB_PRIMARY_HOST} -p ${DB_PRIMARY_PORT}; then
    echo "❌ Primary database unreachable"
    
    # 2. Promote replica to primary
    echo "⬆️ Promoting replica to primary..."
    
    # Update service discovery
    consul kv put database/primary/host ${DB_REPLICA_HOST}
    consul kv put database/primary/port ${DB_REPLICA_PORT}
    
    # Update Kubernetes service
    kubectl patch service postgres-primary -p '{
        "spec": {
            "selector": {
                "role": "replica",
                "instance": "replica-1"
            }
        }
    }'
    
    # 3. Update application configuration
    echo "⚙️ Updating application config..."
    kubectl patch configmap trading-config -p '{
        "data": {
            "DB_PRIMARY_HOST": "'${DB_REPLICA_HOST}'",
            "DB_PRIMARY_PORT": "'${DB_REPLICA_PORT}'"
        }
    }'
    
    # 4. Restart affected pods
    echo "🔄 Restarting affected services..."
    kubectl rollout restart deployment/trading-engine
    kubectl rollout restart deployment/risk-engine
    
    # 5. Verify failover
    echo "✅ Verifying failover..."
    sleep 30
    kubectl get pods -l app=trading-engine
    
    # 6. Test database connectivity
    psql -h ${DB_REPLICA_HOST} -p ${DB_REPLICA_PORT} -U ${DB_USER} -d ${DB_NAME} -c "SELECT NOW();"
    
    echo "✅ Database failover complete"
else
    echo "✅ Primary database is healthy"
fi
```

### **PERFORMANCE DEGRADATION (P2):**
```bash
#!/bin/bash
# performance_recovery.sh

echo "⚡ PERFORMANCE RECOVERY INITIATED"

# 1. Check system resources
echo "📊 Checking system resources..."
echo "CPU Usage:"
top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1

echo "Memory Usage:"
free -h

echo "Disk I/O:"
iostat -x 1 1

# 2. Check database performance
echo ""
echo "🗄️ Database Performance:"
psql -h ${DB_PRIMARY_HOST} -U ${DB_USER} -d ${DB_NAME} -c "
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    rows
FROM pg_stat_statements 
ORDER BY total_time DESC 
LIMIT 10;
"

# 3. Clear caches if needed
echo ""
echo "🧹 Cache Management:"
echo "Redis memory usage:"
redis-cli -h ${REDIS_HOST} info memory | grep used_memory_human

echo "Clearing expired keys:"
redis-cli -h ${REDIS_HOST} --scan --pattern "*:expired:*" | xargs redis-cli del

# 4. Scale services if needed
echo ""
echo "📈 Scaling Services:"
current_replicas=$(kubectl get deployment trading-engine -o jsonpath='{.spec.replicas}')
echo "Current trading-engine replicas: $current_replicas"

if [ "$current_replicas" -lt 5 ]; then
    echo "Scaling up trading-engine..."
    kubectl scale deployment trading-engine --replicas=5
fi

# 5. Check for memory leaks
echo ""
echo "🔍 Memory Leak Detection:"
kubectl top pods -n production --sort-by=memory

# 6. Restart problematic services
echo ""
echo "🔄 Service Restart (if needed):"
# Only restart if memory usage > 80%
kubectl get pods -n production -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.containerStatuses[0].restartCount}{"\n"}{end}' | \
while read pod restarts; do
    if [ "$restarts" -gt 5 ]; then
        echo "Restarting high-restart pod: $pod"
        kubectl delete pod "$pod" -n production
    fi
done

echo "✅ Performance recovery complete"
```

---

## 📊 **MONITORING & ALERTING**

### **Key Metrics to Monitor:**
```yaml
# Prometheus alerts configuration
groups:
  - name: omni_alpha_critical
    rules:
      - alert: TradingSystemDown
        expr: up{job="trading-engine"} == 0
        for: 1m
        labels:
          severity: P0
        annotations:
          summary: "Trading system is down"
          
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 2m
        labels:
          severity: P1
        annotations:
          summary: "High error rate detected"
          
      - alert: DatabaseConnectionFailure
        expr: db_connection_errors_total > 10
        for: 1m
        labels:
          severity: P1
        annotations:
          summary: "Database connection failures"
          
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
        for: 5m
        labels:
          severity: P2
        annotations:
          summary: "High response latency"
          
      - alert: MemoryUsageHigh
        expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes > 0.8
        for: 5m
        labels:
          severity: P2
        annotations:
          summary: "High memory usage"
```

### **Alert Escalation Matrix:**
```
P0: PagerDuty → SMS → Voice Call → Slack #critical
P1: Slack #alerts → Email → PagerDuty (after 10min)
P2: Slack #monitoring → Email
P3: Email → Daily digest
```

---

## 🛠️ **TROUBLESHOOTING GUIDES**

### **Database Issues:**
```bash
# Check connection pool status
SELECT 
    state,
    count(*) 
FROM pg_stat_activity 
GROUP BY state;

# Kill long-running queries
SELECT 
    pg_terminate_backend(pid),
    query_start,
    state,
    query
FROM pg_stat_activity 
WHERE state = 'active' 
AND query_start < now() - interval '5 minutes'
AND query NOT LIKE '%pg_stat_activity%';

# Check locks
SELECT 
    blocked_locks.pid AS blocked_pid,
    blocked_activity.usename AS blocked_user,
    blocking_locks.pid AS blocking_pid,
    blocking_activity.usename AS blocking_user,
    blocked_activity.query AS blocked_statement,
    blocking_activity.query AS current_statement_in_blocking_process
FROM pg_catalog.pg_locks blocked_locks
JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
JOIN pg_catalog.pg_locks blocking_locks 
    ON blocking_locks.locktype = blocked_locks.locktype
    AND blocking_locks.transactionid = blocked_locks.transactionid
JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
WHERE NOT blocked_locks.granted;

# Failover to replica
kubectl patch service postgres-primary -p '{"spec":{"selector":{"role":"replica"}}}'
```

### **High Latency Issues:**
```bash
# Check network latency to external APIs
echo "🌐 Network Latency Check:"
curl -w "@curl-format.txt" -o /dev/null -s "https://paper-api.alpaca.markets/v2/account"

# Check internal service latency
echo "🔗 Internal Service Latency:"
for service in trading-engine risk-engine data-collector; do
    echo "$service:"
    curl -w "Response time: %{time_total}s\n" -o /dev/null -s "http://$service:8080/health"
done

# Check database query performance
echo "🗄️ Slow Queries:"
psql -h ${DB_PRIMARY_HOST} -U ${DB_USER} -d ${DB_NAME} -c "
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    (total_time/calls) as avg_time
FROM pg_stat_statements 
WHERE calls > 100
ORDER BY mean_time DESC 
LIMIT 20;
"

# Check Redis performance
echo "📦 Redis Performance:"
redis-cli -h ${REDIS_HOST} --latency-history -i 1

# Scale up if needed
echo "📈 Auto-scaling check:"
current_cpu=$(kubectl top pods -n production --no-headers | awk '{sum+=$3} END {print sum/NR}' | cut -d'm' -f1)
if [ "$current_cpu" -gt 80 ]; then
    echo "High CPU detected, scaling up..."
    kubectl scale deployment trading-engine --replicas=$(($(kubectl get deployment trading-engine -o jsonpath='{.spec.replicas}') + 2))
fi
```

### **Security Incident Response:**
```bash
# Security incident response
echo "🔒 SECURITY INCIDENT RESPONSE"

# 1. Immediate containment
echo "🚨 Immediate containment..."

# Block suspicious IPs
suspicious_ips=$(tail -1000 /var/log/nginx/access.log | grep -E "(SELECT|UNION|<script|javascript:)" | awk '{print $1}' | sort -u)
for ip in $suspicious_ips; do
    echo "Blocking IP: $ip"
    iptables -A INPUT -s $ip -j DROP
done

# 2. Check for data exfiltration
echo "🔍 Checking for data exfiltration..."
tail -1000 /var/log/nginx/access.log | grep -E "(download|export|backup)" | head -20

# 3. Verify authentication logs
echo "🔐 Authentication audit..."
python -c "
from security.enterprise.security_manager import get_enterprise_security_manager
manager = get_enterprise_security_manager()
summary = manager.get_security_summary()
print(f'Security Status: {summary[\"security_status\"]}')
print(f'Blocked IPs: {summary[\"blocked_ips\"]}')
print(f'Recent Events: {summary[\"recent_events\"]}')
"

# 4. Check for privilege escalation
echo "👑 Privilege escalation check..."
grep -i "sudo\|su\|admin\|root" /var/log/auth.log | tail -20

# 5. Network traffic analysis
echo "🌐 Network analysis..."
netstat -tulpn | grep ESTABLISHED | head -20

# 6. File integrity check
echo "📁 File integrity check..."
find /opt/omni-alpha -name "*.py" -newer /tmp/last_known_good -ls

echo "✅ Security incident response complete"
```

---

## 📞 **ESCALATION PROCEDURES**

### **Contact Information:**
```
PRIMARY ON-CALL: +1-XXX-XXX-XXXX (PagerDuty)
BACKUP ON-CALL: +1-XXX-XXX-XXXX
TRADING DESK: +1-XXX-XXX-XXXX
COMPLIANCE: compliance@omni-alpha.com
SECURITY TEAM: security@omni-alpha.com
```

### **Escalation Timeline:**
```
0-5 min:    On-call engineer
10-15 min:  Senior engineer + Trading desk (P0/P1)
30 min:     Engineering manager + CTO (P0)
1 hour:     Executive team (P0 only)
```

---

## 🔄 **RECOVERY VERIFICATION**

### **System Recovery Checklist:**
```bash
#!/bin/bash
# recovery_verification.sh

echo "✅ RECOVERY VERIFICATION CHECKLIST"

# 1. All services healthy
echo "🔍 Service Health Check:"
kubectl get pods -n production | grep -v Running || echo "❌ Unhealthy pods found"

# 2. Database connectivity
echo "🗄️ Database Check:"
psql -h ${DB_PRIMARY_HOST} -U ${DB_USER} -d ${DB_NAME} -c "SELECT 'DB OK';" || echo "❌ Database issue"

# 3. External API connectivity
echo "🌐 External API Check:"
curl -s https://paper-api.alpaca.markets/v2/account > /dev/null || echo "❌ Alpaca API issue"

# 4. Trading functionality
echo "📈 Trading Test:"
python scripts/test_paper_trade.py || echo "❌ Trading issue"

# 5. Risk controls active
echo "🛡️ Risk Controls:"
python -c "
from risk_management.risk_engine import get_risk_engine
engine = get_risk_engine()
can_trade, msg, level = engine.check_pre_trade_risk('AAPL', 100, 150.0, 'buy')
print('✅ Risk controls active' if can_trade else '❌ Risk controls issue')
"

# 6. Monitoring active
echo "📊 Monitoring Check:"
curl -s http://localhost:8001/metrics > /dev/null || echo "❌ Metrics issue"
curl -s http://localhost:8000/health > /dev/null || echo "❌ Health endpoint issue"

# 7. Performance baseline
echo "⚡ Performance Check:"
response_time=$(curl -w "%{time_total}" -o /dev/null -s http://localhost:8000/health)
if (( $(echo "$response_time > 1.0" | bc -l) )); then
    echo "⚠️ High response time: ${response_time}s"
else
    echo "✅ Response time OK: ${response_time}s"
fi

echo ""
echo "🎉 Recovery verification complete"
echo "⏰ Monitor system for 15 minutes before declaring incident resolved"
```

---

## 📚 **POST-INCIDENT PROCEDURES**

### **1. Incident Documentation:**
```markdown
# Incident Report Template

## Incident Summary
- **Incident ID**: INC-YYYYMMDD-XXX
- **Severity**: P0/P1/P2/P3
- **Start Time**: YYYY-MM-DD HH:MM:SS UTC
- **End Time**: YYYY-MM-DD HH:MM:SS UTC
- **Duration**: X hours Y minutes
- **Impact**: Brief description

## Timeline
- **HH:MM** - Initial alert received
- **HH:MM** - Incident acknowledged
- **HH:MM** - Root cause identified
- **HH:MM** - Mitigation applied
- **HH:MM** - Service restored
- **HH:MM** - Incident resolved

## Root Cause Analysis
- **Primary Cause**: 
- **Contributing Factors**: 
- **Detection Method**: 

## Impact Assessment
- **Financial Impact**: $X,XXX
- **Customer Impact**: X users affected
- **Reputation Impact**: 
- **Regulatory Impact**: 

## Actions Taken
1. Immediate response actions
2. Mitigation steps
3. Recovery procedures

## Lessons Learned
- What went well
- What could be improved
- Process gaps identified

## Follow-up Actions
- [ ] Update runbooks
- [ ] Improve monitoring
- [ ] Code changes required
- [ ] Infrastructure improvements
- [ ] Training needs
```

### **2. Post-Mortem Schedule:**
```
Within 24 hours: Initial incident report
Within 72 hours: Detailed post-mortem meeting
Within 1 week: Action items assigned and tracked
Within 1 month: Preventive measures implemented
```

---

## 🔧 **MAINTENANCE PROCEDURES**

### **Planned Maintenance:**
```bash
#!/bin/bash
# planned_maintenance.sh

echo "🔧 PLANNED MAINTENANCE INITIATED"

# 1. Notify users
echo "📢 Sending maintenance notifications..."
python scripts/maintenance_notification.py --start-time="$(date -d '+10 minutes')"

# 2. Enable maintenance mode
echo "🚧 Enabling maintenance mode..."
kubectl patch configmap trading-config -p '{"data":{"MAINTENANCE_MODE":"true"}}'

# 3. Gracefully stop trading
echo "🛑 Graceful trading stop..."
curl -X POST http://api.omni-alpha.com/admin/trading/graceful-stop

# 4. Wait for position closure
echo "⏳ Waiting for positions to close..."
while [ $(curl -s http://api.omni-alpha.com/v1/portfolio/positions/count) -gt 0 ]; do
    echo "Waiting for positions to close..."
    sleep 30
done

# 5. Perform maintenance
echo "🔄 Performing maintenance..."
# Add maintenance commands here

# 6. Verify system health
echo "✅ Post-maintenance verification..."
./scripts/recovery_verification.sh

# 7. Disable maintenance mode
echo "🚀 Disabling maintenance mode..."
kubectl patch configmap trading-config -p '{"data":{"MAINTENANCE_MODE":"false"}}'

# 8. Resume trading
echo "📈 Resuming trading..."
curl -X POST http://api.omni-alpha.com/admin/trading/resume

echo "✅ Planned maintenance complete"
```

---

## 📋 **QUICK REFERENCE**

### **Emergency Contacts:**
- **Incident Commander**: +1-XXX-XXX-XXXX
- **Technical Lead**: +1-XXX-XXX-XXXX  
- **Trading Desk**: +1-XXX-XXX-XXXX
- **Compliance**: +1-XXX-XXX-XXXX

### **Key URLs:**
- **Monitoring**: http://monitoring.omni-alpha.com
- **Metrics**: http://prometheus.omni-alpha.com
- **Logs**: http://kibana.omni-alpha.com
- **Tracing**: http://jaeger.omni-alpha.com

### **Common Commands:**
```bash
# System status
kubectl get all -n production

# Service logs
kubectl logs -f deployment/trading-engine -n production

# Database status
pg_isready -h db.omni-alpha.com

# Cache status
redis-cli -h redis.omni-alpha.com ping

# Trading status
curl http://api.omni-alpha.com/v1/trading/status
```

---

**🚨 REMEMBER: In P0 incidents, prioritize system stability over data collection. Stop trading first, investigate second! 🚨**
