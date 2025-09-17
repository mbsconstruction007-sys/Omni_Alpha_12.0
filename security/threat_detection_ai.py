"""
LAYER 2: AI-Powered Threat Detection System
Advanced machine learning for anomaly detection
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import asyncio
import logging
import json
import hashlib
from collections import defaultdict, deque

# Machine Learning imports
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

class AIThreatDetectionSystem:
    """
    Machine Learning based threat detection and anomaly analysis
    """
    
    def __init__(self):
        self.models = self._initialize_models()
        self.threat_patterns = self._load_threat_patterns()
        self.behavioral_baselines = {}
        self.activity_buffer = deque(maxlen=10000)
        self.threat_intelligence = {}
        self.real_time_monitoring = True
        
    def _initialize_models(self) -> Dict:
        """Initialize ML models for threat detection"""
        
        models = {}
        
        if SKLEARN_AVAILABLE:
            # Behavioral anomaly detection
            models['behavioral'] = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            
            # Network traffic analysis
            models['network'] = IsolationForest(
                contamination=0.05,
                random_state=42
            )
            
            # Transaction fraud detection
            models['transaction'] = MLPClassifier(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                alpha=0.001,
                random_state=42
            )
            
            # Scaler for feature normalization
            models['scaler'] = StandardScaler()
        else:
            # Fallback implementations
            models['behavioral'] = self._simple_anomaly_detector
            models['network'] = self._simple_network_detector
            models['transaction'] = self._simple_fraud_detector
        
        return models
    
    def _load_threat_patterns(self) -> Dict:
        """Load known threat patterns and signatures"""
        
        return {
            'apt_patterns': [
                'lateral_movement',
                'privilege_escalation', 
                'data_exfiltration',
                'persistence_mechanisms',
                'command_control_communication'
            ],
            'malware_signatures': [
                'suspicious_file_operations',
                'registry_modifications',
                'network_scanning',
                'crypto_mining_behavior'
            ],
            'attack_vectors': [
                'sql_injection',
                'xss_attempts',
                'brute_force',
                'credential_stuffing',
                'session_hijacking'
            ]
        }
    
    async def detect_threats(self, activity_stream: Dict) -> List[Dict]:
        """
        Real-time threat detection using multiple AI models
        """
        
        threats_detected = []
        
        try:
            # Add activity to buffer for analysis
            self.activity_buffer.append({
                'timestamp': datetime.now().isoformat(),
                'activity': activity_stream
            })
            
            # 1. Behavioral Analysis
            behavioral_anomaly = await self._detect_behavioral_anomaly(activity_stream)
            if behavioral_anomaly['score'] > 0.8:
                threats_detected.append({
                    'type': 'BEHAVIORAL_ANOMALY',
                    'severity': 'HIGH',
                    'confidence': behavioral_anomaly['score'],
                    'details': behavioral_anomaly
                })
            
            # 2. Network Traffic Analysis
            network_threat = await self._detect_network_threat(activity_stream)
            if network_threat['threat_level'] > 0.7:
                threats_detected.append({
                    'type': 'NETWORK_INTRUSION',
                    'severity': 'CRITICAL',
                    'confidence': network_threat['threat_level'],
                    'details': network_threat
                })
            
            # 3. Transaction Pattern Analysis
            transaction_anomaly = await self._detect_transaction_fraud(activity_stream)
            if transaction_anomaly['fraud_probability'] > 0.6:
                threats_detected.append({
                    'type': 'FRAUDULENT_TRANSACTION',
                    'severity': 'HIGH',
                    'confidence': transaction_anomaly['fraud_probability'],
                    'details': transaction_anomaly
                })
            
            # 4. Advanced Persistent Threat (APT) Detection
            apt_indicators = await self._detect_apt_activity(activity_stream)
            if apt_indicators['confidence'] > 0.75:
                threats_detected.append({
                    'type': 'APT_DETECTED',
                    'severity': 'CRITICAL',
                    'confidence': apt_indicators['confidence'],
                    'details': apt_indicators
                })
            
            # 5. Zero-Day Exploit Detection
            zero_day = await self._detect_zero_day_patterns(activity_stream)
            if zero_day['probability'] > 0.5:
                threats_detected.append({
                    'type': 'ZERO_DAY_EXPLOIT',
                    'severity': 'CRITICAL',
                    'confidence': zero_day['probability'],
                    'details': zero_day
                })
            
            # 6. Insider Threat Detection
            insider_threat = await self._detect_insider_threat(activity_stream)
            if insider_threat['risk_score'] > 0.7:
                threats_detected.append({
                    'type': 'INSIDER_THREAT',
                    'severity': 'HIGH',
                    'confidence': insider_threat['risk_score'],
                    'details': insider_threat
                })
            
            # Trigger automatic response for critical threats
            for threat in threats_detected:
                if threat['severity'] == 'CRITICAL':
                    await self._trigger_automated_response(threat)
            
            return threats_detected
            
        except Exception as e:
            logger.error(f"Threat detection error: {e}")
            return []
    
    async def _detect_behavioral_anomaly(self, activity: Dict) -> Dict:
        """
        Detect abnormal user behavior using machine learning
        """
        
        try:
            # Extract behavioral features
            features = self._extract_behavioral_features(activity)
            
            if SKLEARN_AVAILABLE and isinstance(self.models['behavioral'], IsolationForest):
                # Use Isolation Forest for anomaly detection
                feature_vector = np.array(features).reshape(1, -1)
                
                # Normalize features
                if hasattr(self.models['scaler'], 'transform'):
                    feature_vector = self.models['scaler'].transform(feature_vector)
                
                anomaly_score = self.models['behavioral'].decision_function(feature_vector)[0]
                is_anomaly = self.models['behavioral'].predict(feature_vector)[0] == -1
                
                return {
                    'score': abs(anomaly_score),
                    'is_anomaly': is_anomaly,
                    'features_analyzed': len(features),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                # Fallback simple anomaly detection
                return await self._simple_behavioral_analysis(activity)
                
        except Exception as e:
            logger.error(f"Behavioral analysis error: {e}")
            return {'score': 0.0, 'is_anomaly': False, 'error': str(e)}
    
    async def _detect_network_threat(self, activity: Dict) -> Dict:
        """
        Detect network-based threats
        """
        
        try:
            network_features = {
                'connection_count': activity.get('connection_count', 0),
                'data_volume': activity.get('data_volume', 0),
                'port_scan_indicators': activity.get('port_scan_indicators', 0),
                'suspicious_protocols': activity.get('suspicious_protocols', 0),
                'geographic_anomalies': activity.get('geographic_anomalies', 0)
            }
            
            # Calculate threat level
            threat_indicators = 0
            
            if network_features['connection_count'] > 1000:
                threat_indicators += 0.3
            
            if network_features['data_volume'] > 1024 * 1024 * 100:  # 100MB
                threat_indicators += 0.2
            
            if network_features['port_scan_indicators'] > 0:
                threat_indicators += 0.4
            
            if network_features['suspicious_protocols'] > 0:
                threat_indicators += 0.3
            
            return {
                'threat_level': min(1.0, threat_indicators),
                'indicators': network_features,
                'analysis_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Network threat detection error: {e}")
            return {'threat_level': 0.0, 'error': str(e)}
    
    async def _detect_transaction_fraud(self, activity: Dict) -> Dict:
        """
        Detect fraudulent trading transactions
        """
        
        try:
            transaction_features = {
                'amount': activity.get('transaction_amount', 0),
                'frequency': activity.get('transaction_frequency', 0),
                'time_of_day': datetime.now().hour,
                'day_of_week': datetime.now().weekday(),
                'location_change': activity.get('location_change', False),
                'device_change': activity.get('device_change', False)
            }
            
            # Fraud indicators
            fraud_score = 0.0
            
            # Large transaction amount
            if transaction_features['amount'] > 10000000:  # 1 Crore
                fraud_score += 0.3
            
            # High frequency trading (potential wash trading)
            if transaction_features['frequency'] > 100:
                fraud_score += 0.4
            
            # Unusual timing
            if transaction_features['time_of_day'] < 6 or transaction_features['time_of_day'] > 22:
                fraud_score += 0.2
            
            # Location/device changes
            if transaction_features['location_change']:
                fraud_score += 0.3
            
            if transaction_features['device_change']:
                fraud_score += 0.2
            
            return {
                'fraud_probability': min(1.0, fraud_score),
                'features': transaction_features,
                'risk_factors': self._identify_risk_factors(transaction_features)
            }
            
        except Exception as e:
            logger.error(f"Transaction fraud detection error: {e}")
            return {'fraud_probability': 0.0, 'error': str(e)}
    
    async def _detect_apt_activity(self, activity: Dict) -> Dict:
        """
        Detect Advanced Persistent Threats using pattern matching
        """
        
        try:
            apt_indicators = {
                'lateral_movement': False,
                'data_exfiltration': False,
                'privilege_escalation': False,
                'persistence_mechanisms': False,
                'command_control': False,
                'reconnaissance': False
            }
            
            # Check for lateral movement patterns
            if self._check_lateral_movement(activity):
                apt_indicators['lateral_movement'] = True
            
            # Check for data exfiltration
            if self._check_data_exfiltration(activity):
                apt_indicators['data_exfiltration'] = True
            
            # Check for privilege escalation attempts
            if self._check_privilege_escalation(activity):
                apt_indicators['privilege_escalation'] = True
            
            # Check for persistence mechanisms
            if self._check_persistence_mechanisms(activity):
                apt_indicators['persistence_mechanisms'] = True
            
            # Check for command and control communication
            if self._check_command_control(activity):
                apt_indicators['command_control'] = True
            
            # Check for reconnaissance
            if self._check_reconnaissance(activity):
                apt_indicators['reconnaissance'] = True
            
            # Calculate confidence score
            indicators_found = sum(apt_indicators.values())
            confidence = indicators_found / len(apt_indicators)
            
            return {
                'confidence': confidence,
                'indicators': apt_indicators,
                'risk_level': 'CRITICAL' if confidence > 0.6 else 'HIGH' if confidence > 0.3 else 'LOW',
                'threat_actor_profile': self._profile_threat_actor(apt_indicators)
            }
            
        except Exception as e:
            logger.error(f"APT detection error: {e}")
            return {'confidence': 0.0, 'error': str(e)}
    
    async def _detect_zero_day_patterns(self, activity: Dict) -> Dict:
        """
        Detect potential zero-day exploits using heuristic analysis
        """
        
        try:
            zero_day_indicators = {
                'unknown_attack_pattern': False,
                'exploit_characteristics': False,
                'payload_analysis': False,
                'system_vulnerability': False
            }
            
            # Check for unknown attack patterns
            if self._check_unknown_patterns(activity):
                zero_day_indicators['unknown_attack_pattern'] = True
            
            # Analyze for exploit characteristics
            if self._analyze_exploit_characteristics(activity):
                zero_day_indicators['exploit_characteristics'] = True
            
            # Payload analysis
            if self._analyze_payload(activity):
                zero_day_indicators['payload_analysis'] = True
            
            # System vulnerability assessment
            if self._check_system_vulnerabilities(activity):
                zero_day_indicators['system_vulnerability'] = True
            
            probability = sum(zero_day_indicators.values()) / len(zero_day_indicators)
            
            return {
                'probability': probability,
                'indicators': zero_day_indicators,
                'confidence': probability * 0.8,  # Conservative confidence
                'recommended_action': 'IMMEDIATE_ISOLATION' if probability > 0.7 else 'MONITOR'
            }
            
        except Exception as e:
            logger.error(f"Zero-day detection error: {e}")
            return {'probability': 0.0, 'error': str(e)}
    
    async def _detect_insider_threat(self, activity: Dict) -> Dict:
        """
        Detect insider threats using behavioral analysis
        """
        
        try:
            user_id = activity.get('user_id')
            if not user_id:
                return {'risk_score': 0.0, 'reason': 'No user ID provided'}
            
            # Get user's baseline behavior
            baseline = self.behavioral_baselines.get(user_id, {})
            
            insider_indicators = {
                'unusual_access_patterns': False,
                'data_hoarding': False,
                'privilege_abuse': False,
                'after_hours_activity': False,
                'large_data_transfers': False,
                'policy_violations': False
            }
            
            # Check for unusual access patterns
            if self._check_unusual_access(activity, baseline):
                insider_indicators['unusual_access_patterns'] = True
            
            # Check for data hoarding
            if self._check_data_hoarding(activity):
                insider_indicators['data_hoarding'] = True
            
            # Check for privilege abuse
            if self._check_privilege_abuse(activity):
                insider_indicators['privilege_abuse'] = True
            
            # Check for after-hours activity
            current_hour = datetime.now().hour
            if current_hour < 6 or current_hour > 22:
                insider_indicators['after_hours_activity'] = True
            
            # Check for large data transfers
            if activity.get('data_transfer_size', 0) > 1024 * 1024 * 50:  # 50MB
                insider_indicators['large_data_transfers'] = True
            
            risk_score = sum(insider_indicators.values()) / len(insider_indicators)
            
            return {
                'risk_score': risk_score,
                'indicators': insider_indicators,
                'user_id': user_id,
                'risk_level': 'HIGH' if risk_score > 0.6 else 'MEDIUM' if risk_score > 0.3 else 'LOW'
            }
            
        except Exception as e:
            logger.error(f"Insider threat detection error: {e}")
            return {'risk_score': 0.0, 'error': str(e)}
    
    def _extract_behavioral_features(self, activity: Dict) -> List[float]:
        """Extract behavioral features for ML analysis"""
        
        features = [
            activity.get('typing_speed', 100) / 200.0,  # Normalize to 0-1
            activity.get('mouse_velocity', 50) / 100.0,
            activity.get('session_duration', 3600) / 7200.0,  # Normalize to 0-1 for 2 hours max
            activity.get('command_count', 10) / 100.0,
            activity.get('error_rate', 0.05),
            len(activity.get('unique_commands', [])) / 50.0,
            activity.get('pause_frequency', 5) / 20.0,
            activity.get('navigation_speed', 1.0),
            activity.get('multitasking_score', 0.5),
            activity.get('focus_score', 0.8)
        ]
        
        return features
    
    async def _simple_behavioral_analysis(self, activity: Dict) -> Dict:
        """Simple behavioral analysis fallback"""
        
        # Rule-based anomaly detection
        anomaly_score = 0.0
        
        # Check typing speed
        typing_speed = activity.get('typing_speed', 100)
        if typing_speed < 20 or typing_speed > 300:
            anomaly_score += 0.3
        
        # Check session duration
        session_duration = activity.get('session_duration', 3600)
        if session_duration > 14400:  # More than 4 hours
            anomaly_score += 0.2
        
        # Check error rate
        error_rate = activity.get('error_rate', 0.05)
        if error_rate > 0.2:
            anomaly_score += 0.3
        
        # Check command patterns
        commands = activity.get('commands_executed', [])
        if len(commands) > 100:
            anomaly_score += 0.2
        
        return {
            'score': min(1.0, anomaly_score),
            'is_anomaly': anomaly_score > 0.5,
            'method': 'rule_based'
        }
    
    def _check_lateral_movement(self, activity: Dict) -> bool:
        """Check for lateral movement indicators"""
        
        indicators = [
            activity.get('multiple_system_access', False),
            activity.get('credential_reuse', False),
            activity.get('network_scanning', False),
            activity.get('remote_access_tools', False)
        ]
        
        return sum(indicators) >= 2
    
    def _check_data_exfiltration(self, activity: Dict) -> bool:
        """Check for data exfiltration patterns"""
        
        # Large data transfers
        data_transfer = activity.get('data_transfer_size', 0)
        if data_transfer > 1024 * 1024 * 100:  # 100MB
            return True
        
        # Unusual file access patterns
        files_accessed = activity.get('files_accessed', [])
        if len(files_accessed) > 1000:
            return True
        
        # Compression activities
        if activity.get('compression_activity', False):
            return True
        
        return False
    
    def _check_privilege_escalation(self, activity: Dict) -> bool:
        """Check for privilege escalation attempts"""
        
        escalation_indicators = [
            activity.get('admin_access_attempts', 0) > 0,
            activity.get('system_file_modifications', False),
            activity.get('service_manipulations', False),
            activity.get('registry_modifications', False)
        ]
        
        return any(escalation_indicators)
    
    def _check_persistence_mechanisms(self, activity: Dict) -> bool:
        """Check for persistence mechanism installation"""
        
        persistence_indicators = [
            activity.get('startup_modifications', False),
            activity.get('scheduled_task_creation', False),
            activity.get('service_installation', False),
            activity.get('dll_injection', False)
        ]
        
        return any(persistence_indicators)
    
    def _check_command_control(self, activity: Dict) -> bool:
        """Check for command and control communication"""
        
        c2_indicators = [
            activity.get('suspicious_network_connections', False),
            activity.get('encrypted_communications', False),
            activity.get('dns_tunneling', False),
            activity.get('beacon_activity', False)
        ]
        
        return any(c2_indicators)
    
    def _check_reconnaissance(self, activity: Dict) -> bool:
        """Check for reconnaissance activities"""
        
        recon_indicators = [
            activity.get('system_enumeration', False),
            activity.get('network_discovery', False),
            activity.get('user_enumeration', False),
            activity.get('vulnerability_scanning', False)
        ]
        
        return any(recon_indicators)
    
    def _check_unknown_patterns(self, activity: Dict) -> bool:
        """Check for unknown attack patterns"""
        
        # Compare against known good patterns
        activity_hash = self._calculate_activity_hash(activity)
        
        # Check if pattern is in known good patterns
        known_patterns = getattr(self, 'known_good_patterns', set())
        
        return activity_hash not in known_patterns
    
    def _analyze_exploit_characteristics(self, activity: Dict) -> bool:
        """Analyze for exploit characteristics"""
        
        exploit_indicators = [
            activity.get('buffer_overflow_attempts', False),
            activity.get('memory_corruption', False),
            activity.get('code_injection', False),
            activity.get('return_oriented_programming', False)
        ]
        
        return any(exploit_indicators)
    
    def _analyze_payload(self, activity: Dict) -> bool:
        """Analyze payload for malicious content"""
        
        payload = activity.get('payload', '')
        if not payload:
            return False
        
        # Check for common exploit payloads
        malicious_patterns = [
            'shellcode',
            'metasploit',
            'cobalt_strike',
            'powershell_empire',
            'mimikatz'
        ]
        
        return any(pattern in payload.lower() for pattern in malicious_patterns)
    
    def _check_system_vulnerabilities(self, activity: Dict) -> bool:
        """Check if activity targets known vulnerabilities"""
        
        # In production, integrate with vulnerability databases
        return activity.get('targets_known_vulnerability', False)
    
    def _calculate_activity_hash(self, activity: Dict) -> str:
        """Calculate hash of activity pattern"""
        
        # Create normalized representation
        normalized = json.dumps(activity, sort_keys=True)
        return hashlib.sha256(normalized.encode()).hexdigest()
    
    def _identify_risk_factors(self, features: Dict) -> List[str]:
        """Identify specific risk factors"""
        
        risk_factors = []
        
        if features.get('amount', 0) > 5000000:
            risk_factors.append('Large transaction amount')
        
        if features.get('frequency', 0) > 50:
            risk_factors.append('High transaction frequency')
        
        if features.get('location_change', False):
            risk_factors.append('Unusual location')
        
        if features.get('device_change', False):
            risk_factors.append('New device')
        
        return risk_factors
    
    def _profile_threat_actor(self, indicators: Dict) -> Dict:
        """Profile threat actor based on indicators"""
        
        if indicators.get('lateral_movement') and indicators.get('data_exfiltration'):
            return {
                'type': 'NATION_STATE',
                'sophistication': 'HIGH',
                'motivation': 'ESPIONAGE'
            }
        elif indicators.get('privilege_escalation') and indicators.get('persistence_mechanisms'):
            return {
                'type': 'CYBERCRIMINAL',
                'sophistication': 'MEDIUM',
                'motivation': 'FINANCIAL'
            }
        else:
            return {
                'type': 'SCRIPT_KIDDIE',
                'sophistication': 'LOW',
                'motivation': 'UNKNOWN'
            }
    
    async def _trigger_automated_response(self, threat: Dict):
        """
        Automated threat response system
        """
        
        try:
            response_actions = []
            
            if threat['type'] == 'NETWORK_INTRUSION':
                response_actions.extend([
                    self._isolate_network_segment(),
                    self._block_suspicious_connections(),
                    self._enable_enhanced_monitoring()
                ])
            
            elif threat['type'] == 'APT_DETECTED':
                response_actions.extend([
                    self._initiate_incident_response(),
                    self._preserve_forensic_evidence(),
                    self._notify_security_team()
                ])
            
            elif threat['type'] == 'ZERO_DAY_EXPLOIT':
                response_actions.extend([
                    self._emergency_system_isolation(),
                    self._activate_backup_systems(),
                    self._alert_vendor_security_team()
                ])
            
            # Execute all response actions
            for action in response_actions:
                try:
                    await action
                except Exception as e:
                    logger.error(f"Response action failed: {e}")
            
            logger.info(f"Automated response triggered for {threat['type']}")
            
        except Exception as e:
            logger.error(f"Automated response error: {e}")
    
    async def _isolate_network_segment(self):
        """Isolate affected network segment"""
        logger.info("Network segment isolation initiated")
    
    async def _block_suspicious_connections(self):
        """Block suspicious network connections"""
        logger.info("Suspicious connections blocked")
    
    async def _enable_enhanced_monitoring(self):
        """Enable enhanced monitoring mode"""
        logger.info("Enhanced monitoring enabled")
    
    async def _initiate_incident_response(self):
        """Initiate formal incident response"""
        logger.critical("Incident response initiated")
    
    async def _preserve_forensic_evidence(self):
        """Preserve forensic evidence"""
        logger.info("Forensic evidence preservation started")
    
    async def _notify_security_team(self):
        """Notify security team"""
        logger.critical("Security team notified")
    
    async def _emergency_system_isolation(self):
        """Emergency system isolation"""
        logger.critical("Emergency system isolation activated")
    
    async def _activate_backup_systems(self):
        """Activate backup systems"""
        logger.info("Backup systems activated")
    
    async def _alert_vendor_security_team(self):
        """Alert vendor security team"""
        logger.critical("Vendor security team alerted")
    
    def update_behavioral_baseline(self, user_id: str, activity: Dict):
        """Update user's behavioral baseline"""
        
        if user_id not in self.behavioral_baselines:
            self.behavioral_baselines[user_id] = {
                'typing_speed': [],
                'session_duration': [],
                'command_patterns': [],
                'error_rates': [],
                'last_updated': datetime.now()
            }
        
        baseline = self.behavioral_baselines[user_id]
        
        # Update metrics
        baseline['typing_speed'].append(activity.get('typing_speed', 100))
        baseline['session_duration'].append(activity.get('session_duration', 3600))
        baseline['error_rates'].append(activity.get('error_rate', 0.05))
        baseline['last_updated'] = datetime.now()
        
        # Keep only recent data (last 100 sessions)
        for key in ['typing_speed', 'session_duration', 'error_rates']:
            baseline[key] = baseline[key][-100:]
    
    def get_security_dashboard_data(self) -> Dict:
        """Get data for security dashboard"""
        
        return {
            'total_events': len(self.security_events),
            'critical_events': len([e for e in self.security_events if e.get('severity') == 'CRITICAL']),
            'high_events': len([e for e in self.security_events if e.get('severity') == 'HIGH']),
            'medium_events': len([e for e in self.security_events if e.get('severity') == 'MEDIUM']),
            'low_events': len([e for e in self.security_events if e.get('severity') == 'LOW']),
            'last_24h_events': len([
                e for e in self.security_events 
                if datetime.fromisoformat(e['timestamp']) > datetime.now() - timedelta(days=1)
            ]),
            'threat_trends': self._calculate_threat_trends(),
            'security_score': self._calculate_security_score(),
            'active_monitoring': self.real_time_monitoring
        }
    
    def _calculate_threat_trends(self) -> Dict:
        """Calculate threat trends over time"""
        
        # Group events by day
        events_by_day = defaultdict(int)
        
        for event in self.security_events:
            event_date = datetime.fromisoformat(event['timestamp']).date()
            events_by_day[event_date.isoformat()] += 1
        
        # Calculate trend
        dates = sorted(events_by_day.keys())
        if len(dates) >= 2:
            recent_avg = sum(events_by_day[d] for d in dates[-7:]) / 7  # Last 7 days
            previous_avg = sum(events_by_day[d] for d in dates[-14:-7]) / 7  # Previous 7 days
            
            if previous_avg > 0:
                trend = (recent_avg - previous_avg) / previous_avg * 100
            else:
                trend = 0
        else:
            trend = 0
        
        return {
            'daily_events': dict(events_by_day),
            'trend_percentage': trend,
            'trend_direction': 'UP' if trend > 5 else 'DOWN' if trend < -5 else 'STABLE'
        }
    
    def _calculate_security_score(self) -> float:
        """Calculate overall security score"""
        
        base_score = 95.0  # Start with high score
        
        # Deduct for security events
        critical_events = len([e for e in self.security_events if e.get('severity') == 'CRITICAL'])
        high_events = len([e for e in self.security_events if e.get('severity') == 'HIGH'])
        
        base_score -= critical_events * 10  # 10 points per critical event
        base_score -= high_events * 5      # 5 points per high event
        
        # Add points for security measures
        if self.redis_available:
            base_score += 2
        
        if len(self.trusted_devices) > 0:
            base_score += 1
        
        # Ensure score is between 0 and 100
        return max(0, min(100, base_score))
    
    def _simple_anomaly_detector(self, features: List[float]) -> Dict:
        """Simple rule-based anomaly detector"""
        
        anomaly_score = 0.0
        
        # Check each feature for anomalies
        for i, feature in enumerate(features):
            if feature < 0.1 or feature > 0.9:  # Outside normal range
                anomaly_score += 0.2
        
        return {
            'score': min(1.0, anomaly_score),
            'is_anomaly': anomaly_score > 0.5
        }
    
    def _simple_network_detector(self, activity: Dict) -> Dict:
        """Simple network threat detector"""
        
        threat_level = 0.0
        
        # Check for suspicious network activity
        if activity.get('connection_count', 0) > 100:
            threat_level += 0.3
        
        if activity.get('data_volume', 0) > 1024 * 1024 * 10:  # 10MB
            threat_level += 0.2
        
        return {'threat_level': min(1.0, threat_level)}
    
    def _simple_fraud_detector(self, activity: Dict) -> Dict:
        """Simple fraud detector"""
        
        fraud_probability = 0.0
        
        # Check transaction amount
        amount = activity.get('transaction_amount', 0)
        if amount > 1000000:  # 10 lakhs
            fraud_probability += 0.3
        
        # Check timing
        hour = datetime.now().hour
        if hour < 6 or hour > 22:
            fraud_probability += 0.2
        
        return {'fraud_probability': min(1.0, fraud_probability)}
    
    def _check_unusual_access(self, activity: Dict, baseline: Dict) -> bool:
        """Check for unusual access patterns"""
        
        if not baseline:
            return False
        
        # Compare current activity to baseline
        current_hour = datetime.now().hour
        typical_hours = baseline.get('typical_hours', [9, 10, 11, 14, 15, 16])
        
        return current_hour not in typical_hours
    
    def _check_data_hoarding(self, activity: Dict) -> bool:
        """Check for data hoarding behavior"""
        
        return activity.get('bulk_data_access', False) or activity.get('systematic_data_collection', False)
    
    def _check_privilege_abuse(self, activity: Dict) -> bool:
        """Check for privilege abuse"""
        
        return activity.get('unauthorized_access_attempts', 0) > 0

# Note: zero_trust instance will be created by security_manager to avoid circular imports
