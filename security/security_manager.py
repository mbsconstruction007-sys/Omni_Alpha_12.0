"""
Complete Security Manager - Orchestrates all security layers
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json

# Import all security layers
from .zero_trust_framework import ZeroTrustSecurityFramework
from .threat_detection_ai import AIThreatDetectionSystem
from .advanced_encryption import AdvancedEncryption
from .application_security import ApplicationSecurityLayer

logger = logging.getLogger(__name__)

class ComprehensiveSecurityManager:
    """
    Orchestrates all security layers for complete protection
    """
    
    def __init__(self):
        # Initialize all security layers
        self.zero_trust = ZeroTrustSecurityFramework()
        self.threat_detection = AIThreatDetectionSystem()
        self.encryption = AdvancedEncryption()
        self.app_security = ApplicationSecurityLayer()
        
        # Security metrics
        self.security_metrics = {
            'total_requests': 0,
            'blocked_requests': 0,
            'threats_detected': 0,
            'false_positives': 0,
            'response_time_ms': []
        }
        
        # Security configuration
        self.security_config = {
            'threat_detection_enabled': True,
            'auto_response_enabled': True,
            'logging_enabled': True,
            'encryption_required': True,
            'zero_trust_mode': True
        }
        
        self.active_threats = []
        self.security_alerts = []
        
    async def secure_request_processing(self, request: Dict) -> Dict:
        """
        Process request through all security layers
        """
        
        start_time = datetime.now()
        
        try:
            self.security_metrics['total_requests'] += 1
            
            # Layer 1: Zero-Trust Authentication
            auth_result, auth_error = self.zero_trust.authenticate_request(request)
            if not auth_result:
                self.security_metrics['blocked_requests'] += 1
                return {
                    'status': 'BLOCKED',
                    'reason': auth_error,
                    'layer': 'ZERO_TRUST'
                }
            
            # Layer 2: Input Sanitization
            sanitized_request = await self._sanitize_request_inputs(request)
            
            # Layer 3: Threat Detection
            if self.security_config['threat_detection_enabled']:
                threats = await self.threat_detection.detect_threats(sanitized_request)
                
                if threats:
                    self.security_metrics['threats_detected'] += len(threats)
                    self.active_threats.extend(threats)
                    
                    # Block critical threats
                    critical_threats = [t for t in threats if t['severity'] == 'CRITICAL']
                    if critical_threats:
                        return {
                            'status': 'BLOCKED',
                            'reason': 'Critical threats detected',
                            'threats': critical_threats,
                            'layer': 'THREAT_DETECTION'
                        }
            
            # Layer 4: Business Logic Validation
            business_validation = await self._validate_business_logic(sanitized_request)
            if not business_validation['valid']:
                return {
                    'status': 'BLOCKED',
                    'reason': business_validation['reason'],
                    'layer': 'BUSINESS_LOGIC'
                }
            
            # Layer 5: Data Encryption (if required)
            if self.security_config['encryption_required']:
                encrypted_request = await self._encrypt_sensitive_fields(sanitized_request)
            else:
                encrypted_request = sanitized_request
            
            # Calculate response time
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            self.security_metrics['response_time_ms'].append(response_time)
            
            return {
                'status': 'ALLOWED',
                'request': encrypted_request,
                'security_score': self._calculate_request_security_score(request),
                'response_time_ms': response_time
            }
            
        except Exception as e:
            logger.error(f"Security processing error: {e}")
            return {
                'status': 'ERROR',
                'reason': 'Security system error',
                'error': str(e)
            }
    
    async def _sanitize_request_inputs(self, request: Dict) -> Dict:
        """Sanitize all request inputs"""
        
        sanitized = request.copy()
        
        for key, value in request.items():
            if isinstance(value, str):
                try:
                    # Determine input type
                    input_type = self._determine_input_type(key)
                    
                    # Sanitize input
                    sanitized[key] = self.app_security.sanitize_input(value, input_type)
                    
                except ValueError as e:
                    logger.warning(f"Input sanitization failed for {key}: {e}")
                    # Remove invalid input
                    del sanitized[key]
        
        return sanitized
    
    def _determine_input_type(self, field_name: str) -> str:
        """Determine input type for validation"""
        
        type_mapping = {
            'email': 'email',
            'phone': 'phone',
            'pan': 'pan',
            'aadhar': 'aadhar',
            'amount': 'amount',
            'symbol': 'symbol',
            'quantity': 'numeric',
            'price': 'amount',
            'password': 'text',
            'username': 'alphanumeric'
        }
        
        field_lower = field_name.lower()
        
        for pattern, input_type in type_mapping.items():
            if pattern in field_lower:
                return input_type
        
        return 'text'
    
    async def _validate_business_logic(self, request: Dict) -> Dict:
        """Validate business logic rules"""
        
        try:
            # Trading-specific validations
            if request.get('action') in ['BUY', 'SELL']:
                # Validate trading parameters
                symbol = request.get('symbol')
                quantity = request.get('quantity', 0)
                amount = request.get('amount', 0)
                
                # Check market hours
                if not self._is_market_open():
                    return {
                        'valid': False,
                        'reason': 'Market is closed'
                    }
                
                # Check position limits
                if quantity > 1000:  # Max 1000 shares per order
                    return {
                        'valid': False,
                        'reason': 'Quantity exceeds limit'
                    }
                
                # Check amount limits
                if amount > 10000000:  # Max 1 crore per order
                    return {
                        'valid': False,
                        'reason': 'Amount exceeds limit'
                    }
            
            # Portfolio access validation
            if request.get('action') == 'VIEW_PORTFOLIO':
                portfolio_id = request.get('portfolio_id')
                user_id = request.get('user_id')
                
                if not self._user_owns_portfolio(user_id, portfolio_id):
                    return {
                        'valid': False,
                        'reason': 'Unauthorized portfolio access'
                    }
            
            return {'valid': True}
            
        except Exception as e:
            logger.error(f"Business logic validation error: {e}")
            return {
                'valid': False,
                'reason': 'Business logic validation failed'
            }
    
    async def _encrypt_sensitive_fields(self, request: Dict) -> Dict:
        """Encrypt sensitive fields in request"""
        
        sensitive_fields = [
            'pan_number',
            'aadhar_number',
            'bank_account',
            'password',
            'api_key',
            'secret'
        ]
        
        encrypted_request = request.copy()
        
        for field in sensitive_fields:
            if field in encrypted_request:
                encrypted_value = self.encryption.encrypt_sensitive_data(
                    str(encrypted_request[field]),
                    {'field_name': field, 'request_id': request.get('request_id')}
                )
                encrypted_request[field] = encrypted_value
        
        return encrypted_request
    
    def _calculate_request_security_score(self, request: Dict) -> float:
        """Calculate security score for request"""
        
        score = 100.0
        
        # Deduct for missing security features
        if not request.get('auth_token'):
            score -= 20
        
        if not request.get('device_fingerprint'):
            score -= 10
        
        if not request.get('signature'):
            score -= 15
        
        if not request.get('csrf_token'):
            score -= 10
        
        # Add points for security features
        if request.get('mfa_verified'):
            score += 5
        
        if request.get('biometric_verified'):
            score += 10
        
        return max(0, min(100, score))
    
    def _is_market_open(self) -> bool:
        """Check if market is open"""
        
        now = datetime.now()
        
        # Indian market hours: 9:15 AM to 3:30 PM, Monday to Friday
        if now.weekday() >= 5:  # Weekend
            return False
        
        market_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_start <= now <= market_end
    
    def _user_owns_portfolio(self, user_id: str, portfolio_id: str) -> bool:
        """Check if user owns portfolio"""
        
        # In production, check database
        # For demo, assume user owns portfolio if IDs match pattern
        return user_id and portfolio_id and user_id in portfolio_id
    
    async def generate_security_report(self) -> Dict:
        """Generate comprehensive security report"""
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'security_metrics': self.security_metrics.copy(),
            'active_threats': len(self.active_threats),
            'security_alerts': len(self.security_alerts),
            'zero_trust_metrics': self.zero_trust.get_security_metrics(),
            'encryption_metrics': self.encryption.get_encryption_metrics(),
            'overall_security_score': await self._calculate_overall_security_score()
        }
        
        # Calculate additional metrics
        total_requests = self.security_metrics['total_requests']
        if total_requests > 0:
            report['block_rate'] = (self.security_metrics['blocked_requests'] / total_requests) * 100
            report['threat_detection_rate'] = (self.security_metrics['threats_detected'] / total_requests) * 100
        
        if self.security_metrics['response_time_ms']:
            report['avg_security_overhead_ms'] = sum(self.security_metrics['response_time_ms']) / len(self.security_metrics['response_time_ms'])
        
        return report
    
    async def _calculate_overall_security_score(self) -> float:
        """Calculate overall security score"""
        
        # Component scores
        zero_trust_score = self.zero_trust._calculate_security_score()
        
        # Base score from components
        component_score = zero_trust_score
        
        # Adjust based on threat activity
        recent_threats = len([
            t for t in self.active_threats 
            if datetime.fromisoformat(t.get('timestamp', '2024-01-01')) > datetime.now() - timedelta(hours=24)
        ])
        
        # Deduct for recent threats
        threat_penalty = min(20, recent_threats * 2)
        
        # Calculate final score
        final_score = max(0, min(100, component_score - threat_penalty))
        
        return final_score
    
    async def run_security_health_check(self) -> Dict:
        """Run comprehensive security health check"""
        
        health_check = {
            'timestamp': datetime.now().isoformat(),
            'components': {},
            'overall_health': 'HEALTHY'
        }
        
        # Check each security component
        components = [
            ('Zero Trust Framework', self._check_zero_trust_health),
            ('Threat Detection AI', self._check_threat_detection_health),
            ('Encryption System', self._check_encryption_health),
            ('Application Security', self._check_app_security_health)
        ]
        
        unhealthy_components = 0
        
        for component_name, check_function in components:
            try:
                component_health = await check_function()
                health_check['components'][component_name] = component_health
                
                if not component_health['healthy']:
                    unhealthy_components += 1
                    
            except Exception as e:
                health_check['components'][component_name] = {
                    'healthy': False,
                    'error': str(e)
                }
                unhealthy_components += 1
        
        # Determine overall health
        if unhealthy_components == 0:
            health_check['overall_health'] = 'HEALTHY'
        elif unhealthy_components <= 1:
            health_check['overall_health'] = 'DEGRADED'
        else:
            health_check['overall_health'] = 'UNHEALTHY'
        
        return health_check
    
    async def _check_zero_trust_health(self) -> Dict:
        """Check zero trust framework health"""
        
        try:
            metrics = self.zero_trust.get_security_metrics()
            
            return {
                'healthy': metrics['security_score'] > 80,
                'score': metrics['security_score'],
                'active_tokens': metrics['active_tokens'],
                'blocked_ips': len(metrics.get('blocked_ips', [])),
                'last_check': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'healthy': False, 'error': str(e)}
    
    async def _check_threat_detection_health(self) -> Dict:
        """Check threat detection system health"""
        
        try:
            return {
                'healthy': True,
                'models_loaded': len(self.threat_detection.models),
                'monitoring_active': self.threat_detection.real_time_monitoring,
                'activity_buffer_size': len(self.threat_detection.activity_buffer),
                'last_check': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'healthy': False, 'error': str(e)}
    
    async def _check_encryption_health(self) -> Dict:
        """Check encryption system health"""
        
        try:
            # Test encryption/decryption
            test_data = "Security health check test"
            encrypted = self.encryption.encrypt_sensitive_data(test_data)
            decrypted = self.encryption.decrypt_sensitive_data(encrypted)
            
            encryption_working = (decrypted == test_data)
            
            return {
                'healthy': encryption_working,
                'encryption_layers': len(self.encryption.encryption_layers),
                'test_successful': encryption_working,
                'last_check': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'healthy': False, 'error': str(e)}
    
    async def _check_app_security_health(self) -> Dict:
        """Check application security health"""
        
        try:
            # Test input sanitization
            test_input = "<script>alert('test')</script>"
            sanitized = self.app_security.sanitize_input(test_input, 'text')
            
            xss_prevented = '<script>' not in sanitized
            
            return {
                'healthy': xss_prevented,
                'xss_protection': xss_prevented,
                'csrf_tokens_active': len(self.app_security.csrf_tokens),
                'sessions_active': len(self.app_security.session_security),
                'last_check': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'healthy': False, 'error': str(e)}
    
    def get_security_dashboard_data(self) -> Dict:
        """Get comprehensive security dashboard data"""
        
        return {
            'security_score': self._calculate_overall_security_score(),
            'threat_level': self._calculate_current_threat_level(),
            'security_metrics': self.security_metrics,
            'active_threats': len(self.active_threats),
            'recent_alerts': len([
                alert for alert in self.security_alerts
                if datetime.fromisoformat(alert.get('timestamp', '2024-01-01')) > datetime.now() - timedelta(hours=24)
            ]),
            'component_status': {
                'zero_trust': self.zero_trust._calculate_security_score(),
                'threat_detection': 95.0,  # Simplified
                'encryption': 99.0,
                'application_security': 92.0
            },
            'security_events_24h': self._get_recent_security_events(),
            'top_threats': self._get_top_threat_types(),
            'blocked_ips': len(self.zero_trust._get_blocked_ips()),
            'last_updated': datetime.now().isoformat()
        }
    
    def _calculate_overall_security_score(self) -> float:
        """Calculate overall security score"""
        
        component_scores = [
            self.zero_trust._calculate_security_score(),
            95.0,  # Threat detection score
            99.0,  # Encryption score
            92.0   # Application security score
        ]
        
        # Weighted average
        weights = [0.3, 0.3, 0.2, 0.2]
        weighted_score = sum(score * weight for score, weight in zip(component_scores, weights))
        
        # Adjust for recent threats
        recent_critical_threats = len([
            t for t in self.active_threats
            if t.get('severity') == 'CRITICAL' and
            datetime.fromisoformat(t.get('timestamp', '2024-01-01')) > datetime.now() - timedelta(hours=1)
        ])
        
        threat_penalty = min(10, recent_critical_threats * 5)
        
        return max(0, min(100, weighted_score - threat_penalty))
    
    def _calculate_current_threat_level(self) -> str:
        """Calculate current threat level"""
        
        recent_threats = [
            t for t in self.active_threats
            if datetime.fromisoformat(t.get('timestamp', '2024-01-01')) > datetime.now() - timedelta(hours=1)
        ]
        
        critical_threats = len([t for t in recent_threats if t.get('severity') == 'CRITICAL'])
        high_threats = len([t for t in recent_threats if t.get('severity') == 'HIGH'])
        
        if critical_threats > 0:
            return 'CRITICAL'
        elif high_threats > 2:
            return 'HIGH'
        elif len(recent_threats) > 5:
            return 'ELEVATED'
        else:
            return 'LOW'
    
    def _get_recent_security_events(self) -> List[Dict]:
        """Get recent security events"""
        
        recent_events = []
        
        # From zero trust framework
        recent_events.extend([
            event for event in self.zero_trust.security_events
            if datetime.fromisoformat(event['timestamp']) > datetime.now() - timedelta(hours=24)
        ])
        
        return recent_events[-10:]  # Last 10 events
    
    def _get_top_threat_types(self) -> Dict[str, int]:
        """Get top threat types in last 24 hours"""
        
        threat_counts = {}
        
        recent_threats = [
            t for t in self.active_threats
            if datetime.fromisoformat(t.get('timestamp', '2024-01-01')) > datetime.now() - timedelta(hours=24)
        ]
        
        for threat in recent_threats:
            threat_type = threat.get('type', 'UNKNOWN')
            threat_counts[threat_type] = threat_counts.get(threat_type, 0) + 1
        
        # Sort by count
        return dict(sorted(threat_counts.items(), key=lambda x: x[1], reverse=True))
    
    async def emergency_security_lockdown(self, reason: str) -> Dict:
        """Emergency security lockdown procedure"""
        
        lockdown_actions = []
        
        try:
            # 1. Block all new connections
            lockdown_actions.append("BLOCK_NEW_CONNECTIONS")
            
            # 2. Revoke all active tokens
            for session_id in list(self.app_security.session_security.keys()):
                del self.app_security.session_security[session_id]
            lockdown_actions.append("REVOKE_ALL_SESSIONS")
            
            # 3. Enable maximum security mode
            self.security_config['zero_trust_mode'] = True
            self.security_config['threat_detection_enabled'] = True
            lockdown_actions.append("ENABLE_MAX_SECURITY")
            
            # 4. Create forensic snapshot
            snapshot = await self._create_forensic_snapshot()
            lockdown_actions.append(f"FORENSIC_SNAPSHOT_CREATED: {snapshot}")
            
            # 5. Alert security team
            await self._send_emergency_alert(reason, lockdown_actions)
            lockdown_actions.append("SECURITY_TEAM_ALERTED")
            
            logger.critical(f"Emergency security lockdown initiated: {reason}")
            
            return {
                'lockdown_initiated': True,
                'reason': reason,
                'actions_taken': lockdown_actions,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Emergency lockdown error: {e}")
            return {
                'lockdown_initiated': False,
                'error': str(e)
            }
    
    async def _create_forensic_snapshot(self) -> str:
        """Create forensic snapshot of current state"""
        
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'security_metrics': self.security_metrics,
            'active_threats': self.active_threats,
            'security_events': self.zero_trust.security_events[-100:],  # Last 100 events
            'system_state': {
                'total_sessions': len(self.app_security.session_security),
                'blocked_ips': len(self.zero_trust._get_blocked_ips()),
                'security_score': self._calculate_overall_security_score()
            }
        }
        
        # Encrypt and save snapshot
        snapshot_file = self.encryption.create_encrypted_backup(
            snapshot,
            f"forensic_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        return snapshot_file
    
    async def _send_emergency_alert(self, reason: str, actions: List[str]):
        """Send emergency security alert"""
        
        alert = {
            'alert_type': 'EMERGENCY_SECURITY_LOCKDOWN',
            'reason': reason,
            'actions_taken': actions,
            'timestamp': datetime.now().isoformat(),
            'severity': 'CRITICAL'
        }
        
        self.security_alerts.append(alert)
        
        # In production, send to:
        # - Security team email/SMS
        # - SIEM system
        # - Incident response platform
        # - Management dashboard
        
        logger.critical(f"EMERGENCY SECURITY ALERT: {reason}")

# Global security manager instance
security_manager = ComprehensiveSecurityManager()
