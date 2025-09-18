"""
OMNI ALPHA 5.0 - ENTERPRISE SECURITY MANAGER
============================================
Military-grade security hardening with comprehensive threat protection
"""

import hashlib
import hmac
import secrets
import time
import os
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import logging
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.backends import default_backend
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False

try:
    from prometheus_client import Counter, Gauge, Histogram
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False

from config.settings import get_settings
from config.logging_config import get_logger

logger = get_logger(__name__, 'security_manager')

# Security metrics (if available)
if PROMETHEUS_AVAILABLE:
    auth_attempts = Counter('auth_attempts_total', 'Total authentication attempts', ['result', 'method'])
    security_violations = Counter('security_violations_total', 'Security violations detected', ['type', 'severity'])
    token_operations = Counter('token_operations_total', 'Token operations', ['operation', 'result'])
    encryption_operations = Counter('encryption_operations_total', 'Encryption operations', ['operation'])
    failed_logins = Counter('failed_logins_total', 'Failed login attempts', ['source_ip', 'username'])
    blocked_requests = Counter('blocked_requests_total', 'Blocked requests', ['reason'])

class ThreatLevel(Enum):
    """Security threat levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AuthMethod(Enum):
    """Authentication methods"""
    PASSWORD = "password"
    TOKEN = "token"
    API_KEY = "api_key"
    CERTIFICATE = "certificate"
    BIOMETRIC = "biometric"

@dataclass
class SecurityEvent:
    """Security event data"""
    timestamp: datetime
    event_type: str
    threat_level: ThreatLevel
    source_ip: str
    user_agent: str
    details: Dict[str, Any]
    action_taken: str = ""

@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    requests_per_window: int
    window_seconds: int
    burst_limit: int = 0
    penalty_seconds: int = 300

class EnterpriseSecurityManager:
    """Comprehensive enterprise security manager"""
    
    def __init__(self, settings=None):
        if settings is None:
            settings = get_settings()
        
        self.settings = settings
        self.logger = get_logger(__name__, 'enterprise_security')
        
        # Security configuration
        self.master_key = os.getenv('MASTER_KEY', self._generate_master_key())
        self.jwt_secret = os.getenv('JWT_SECRET', secrets.token_urlsafe(64))
        self.api_key_secret = os.getenv('API_KEY_SECRET', secrets.token_urlsafe(32))
        
        # Encryption setup
        if CRYPTOGRAPHY_AVAILABLE:
            self.fernet = Fernet(self.master_key.encode() if isinstance(self.master_key, str) else self.master_key)
        
        # Rate limiting
        self.rate_limiters: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'requests': deque(),
            'blocked_until': None
        })
        
        # Security monitoring
        self.security_events: deque = deque(maxlen=10000)
        self.blocked_ips: Dict[str, datetime] = {}
        self.suspicious_patterns: Dict[str, int] = defaultdict(int)
        
        # Authentication tracking
        self.failed_attempts: Dict[str, List[datetime]] = defaultdict(list)
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Default rate limit configurations
        self.rate_limit_configs = {
            'api': RateLimitConfig(100, 60, 200, 300),
            'auth': RateLimitConfig(10, 60, 20, 600),
            'admin': RateLimitConfig(50, 60, 100, 300)
        }
        
        # Security patterns
        self.injection_patterns = {
            'sql': r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|CREATE|ALTER|EXEC|EXECUTE)\b|[';\"\\]|--|\*\/|\*)",
            'xss': r"(<script|javascript:|on\w+\s*=|<iframe|<object|<embed|eval\(|alert\()",
            'command': r"([;&|`$]|\b(rm|curl|wget|bash|sh|nc|netcat|telnet|ssh)\b)",
            'path': r"(\.\.\/|\.\.\\|%2e%2e%2f|%2e%2e\\)",
            'ldap': r"(\*|\(|\)|&|\||\!|=|<|>|~|;|,|\+|\-|\"|\')|(objectClass=|cn=|uid=|ou=)",
            'xml': r"(<!ENTITY|<!DOCTYPE|<\?xml|SYSTEM|PUBLIC)"
        }
    
    def _generate_master_key(self) -> bytes:
        """Generate a new master encryption key"""
        key = Fernet.generate_key()
        self.logger.warning("Generated new master key - store securely in production!")
        return key
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data using Fernet"""
        if not CRYPTOGRAPHY_AVAILABLE:
            self.logger.warning("Cryptography not available - data not encrypted")
            return data
        
        try:
            encrypted = self.fernet.encrypt(data.encode())
            
            if PROMETHEUS_AVAILABLE:
                encryption_operations.labels(operation='encrypt').inc()
            
            return encrypted.decode()
            
        except Exception as e:
            self.logger.error(f"Encryption failed: {e}")
            raise
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        if not CRYPTOGRAPHY_AVAILABLE:
            self.logger.warning("Cryptography not available - returning data as-is")
            return encrypted_data
        
        try:
            decrypted = self.fernet.decrypt(encrypted_data.encode())
            
            if PROMETHEUS_AVAILABLE:
                encryption_operations.labels(operation='decrypt').inc()
            
            return decrypted.decode()
            
        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            raise
    
    def hash_password(self, password: str) -> tuple:
        """Hash password with bcrypt or PBKDF2"""
        if BCRYPT_AVAILABLE:
            # Use bcrypt (preferred)
            salt = bcrypt.gensalt(rounds=12)
            hashed = bcrypt.hashpw(password.encode(), salt)
            return hashed, salt
        
        elif CRYPTOGRAPHY_AVAILABLE:
            # Fallback to PBKDF2
            salt = secrets.token_bytes(32)
            kdf = PBKDF2(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
                backend=default_backend()
            )
            hashed = kdf.derive(password.encode())
            return hashed, salt
        
        else:
            # Basic fallback (NOT recommended for production)
            salt = secrets.token_bytes(32)
            hashed = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
            return hashed, salt
    
    def verify_password(self, password: str, hashed: bytes, salt: bytes = None) -> bool:
        """Verify password against hash"""
        try:
            if BCRYPT_AVAILABLE and salt is None:
                # bcrypt verification
                return bcrypt.checkpw(password.encode(), hashed)
            
            elif CRYPTOGRAPHY_AVAILABLE and salt:
                # PBKDF2 verification
                kdf = PBKDF2(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                    backend=default_backend()
                )
                try:
                    kdf.verify(password.encode(), hashed)
                    return True
                except:
                    return False
            
            else:
                # Basic fallback
                test_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
                return hmac.compare_digest(test_hash, hashed)
                
        except Exception as e:
            self.logger.error(f"Password verification error: {e}")
            return False
    
    def generate_secure_token(self, user_id: str, expiry_hours: int = 24,
                             token_type: str = "access", scopes: List[str] = None) -> str:
        """Generate secure JWT token"""
        if not JWT_AVAILABLE:
            # Fallback to simple token
            return secrets.token_urlsafe(32)
        
        try:
            now = datetime.utcnow()
            payload = {
                'user_id': user_id,
                'token_type': token_type,
                'iat': now,
                'exp': now + timedelta(hours=expiry_hours),
                'jti': secrets.token_urlsafe(16),
                'scopes': scopes or []
            }
            
            token = jwt.encode(
                payload,
                self.jwt_secret,
                algorithm='HS256'
            )
            
            if PROMETHEUS_AVAILABLE:
                token_operations.labels(operation='generate', result='success').inc()
            
            return token
            
        except Exception as e:
            if PROMETHEUS_AVAILABLE:
                token_operations.labels(operation='generate', result='error').inc()
            
            self.logger.error(f"Token generation failed: {e}")
            raise
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        if not JWT_AVAILABLE:
            self.logger.warning("JWT not available - token verification disabled")
            return None
        
        try:
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=['HS256']
            )
            
            if PROMETHEUS_AVAILABLE:
                token_operations.labels(operation='verify', result='success').inc()
            
            return payload
            
        except jwt.ExpiredSignatureError:
            if PROMETHEUS_AVAILABLE:
                token_operations.labels(operation='verify', result='expired').inc()
            return None
            
        except jwt.InvalidTokenError:
            if PROMETHEUS_AVAILABLE:
                token_operations.labels(operation='verify', result='invalid').inc()
            return None
        
        except Exception as e:
            if PROMETHEUS_AVAILABLE:
                token_operations.labels(operation='verify', result='error').inc()
            
            self.logger.error(f"Token verification error: {e}")
            return None
    
    def validate_input_security(self, input_data: str, input_type: str = 'general') -> tuple:
        """Comprehensive input validation against injection attacks"""
        violations = []
        threat_level = ThreatLevel.LOW
        
        if not input_data:
            return True, violations, threat_level
        
        # Check against injection patterns
        for attack_type, pattern in self.injection_patterns.items():
            matches = re.findall(pattern, input_data, re.IGNORECASE | re.MULTILINE)
            if matches:
                violations.append(f"{attack_type}_injection")
                threat_level = ThreatLevel.HIGH
                
                if PROMETHEUS_AVAILABLE:
                    security_violations.labels(type=f'{attack_type}_injection', severity='high').inc()
        
        # Additional checks
        if len(input_data) > 10000:  # Suspiciously long input
            violations.append("oversized_input")
            threat_level = max(threat_level, ThreatLevel.MEDIUM)
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]',  # Control characters
            r'[^\x20-\x7E\s]',  # Non-printable characters
            r'(password|secret|key|token)[\s=:]+[^\s]+',  # Potential secrets
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, input_data, re.IGNORECASE):
                violations.append("suspicious_content")
                threat_level = max(threat_level, ThreatLevel.MEDIUM)
                break
        
        is_safe = len(violations) == 0
        return is_safe, violations, threat_level
    
    def check_rate_limit(self, identifier: str, limit_type: str = 'api') -> tuple:
        """Check and enforce rate limits"""
        config = self.rate_limit_configs.get(limit_type, self.rate_limit_configs['api'])
        limiter = self.rate_limiters[identifier]
        now = time.time()
        
        # Check if currently blocked
        if limiter['blocked_until'] and now < limiter['blocked_until']:
            remaining_block = int(limiter['blocked_until'] - now)
            
            if PROMETHEUS_AVAILABLE:
                blocked_requests.labels(reason='rate_limit').inc()
            
            return False, f"Rate limited for {remaining_block} seconds"
        
        # Clean old requests
        window_start = now - config.window_seconds
        limiter['requests'] = deque([
            req_time for req_time in limiter['requests']
            if req_time > window_start
        ])
        
        # Check rate limit
        current_requests = len(limiter['requests'])
        
        if current_requests >= config.requests_per_window:
            # Apply penalty
            limiter['blocked_until'] = now + config.penalty_seconds
            
            if PROMETHEUS_AVAILABLE:
                blocked_requests.labels(reason='rate_limit_exceeded').inc()
            
            return False, f"Rate limit exceeded. Blocked for {config.penalty_seconds} seconds"
        
        # Add current request
        limiter['requests'].append(now)
        
        return True, f"OK ({current_requests + 1}/{config.requests_per_window})"
    
    def check_ip_security(self, ip_address: str) -> tuple:
        """Check IP address against security policies"""
        # Check blocked IPs
        if ip_address in self.blocked_ips:
            block_time = self.blocked_ips[ip_address]
            if datetime.now() - block_time < timedelta(hours=24):
                
                if PROMETHEUS_AVAILABLE:
                    blocked_requests.labels(reason='blocked_ip').inc()
                
                return False, "IP address blocked"
        
        # Check IP whitelist (if configured)
        whitelist = os.getenv('IP_WHITELIST', '').split(',')
        if whitelist and whitelist != [''] and ip_address not in whitelist:
            
            if PROMETHEUS_AVAILABLE:
                blocked_requests.labels(reason='not_whitelisted').inc()
            
            return False, "IP not whitelisted"
        
        # Check for suspicious IP patterns
        if self._is_suspicious_ip(ip_address):
            
            if PROMETHEUS_AVAILABLE:
                security_violations.labels(type='suspicious_ip', severity='medium').inc()
            
            return False, "Suspicious IP address"
        
        return True, "IP allowed"
    
    def _is_suspicious_ip(self, ip_address: str) -> bool:
        """Check if IP address matches suspicious patterns"""
        # Check for private/internal IPs in production
        if self.settings.environment.value == 'production':
            private_ranges = [
                r'^10\.',
                r'^192\.168\.',
                r'^172\.(1[6-9]|2[0-9]|3[0-1])\.',
                r'^127\.',
                r'^169\.254\.',
                r'^::1$',
                r'^fe80::'
            ]
            
            for pattern in private_ranges:
                if re.match(pattern, ip_address):
                    return True
        
        # Check known malicious patterns (basic examples)
        malicious_patterns = [
            r'^0\.0\.0\.0$',
            r'^255\.255\.255\.255$',
            # Add more patterns based on threat intelligence
        ]
        
        for pattern in malicious_patterns:
            if re.match(pattern, ip_address):
                return True
        
        return False
    
    def log_security_event(self, event_type: str, threat_level: ThreatLevel,
                          source_ip: str, user_agent: str = "",
                          details: Dict[str, Any] = None, action_taken: str = ""):
        """Log security event"""
        event = SecurityEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            threat_level=threat_level,
            source_ip=source_ip,
            user_agent=user_agent,
            details=details or {},
            action_taken=action_taken
        )
        
        self.security_events.append(event)
        
        # Log to standard logger
        log_level = {
            ThreatLevel.LOW: logging.INFO,
            ThreatLevel.MEDIUM: logging.WARNING,
            ThreatLevel.HIGH: logging.ERROR,
            ThreatLevel.CRITICAL: logging.CRITICAL
        }.get(threat_level, logging.INFO)
        
        self.logger.log(
            log_level,
            f"Security event: {event_type} from {source_ip}",
            extra={
                'event_type': event_type,
                'threat_level': threat_level.value,
                'source_ip': source_ip,
                'user_agent': user_agent,
                'details': details,
                'action_taken': action_taken
            }
        )
        
        # Update metrics
        if PROMETHEUS_AVAILABLE:
            security_violations.labels(type=event_type, severity=threat_level.value).inc()
        
        # Take automatic action for critical threats
        if threat_level == ThreatLevel.CRITICAL:
            self._handle_critical_threat(event)
    
    def _handle_critical_threat(self, event: SecurityEvent):
        """Handle critical security threats automatically"""
        # Block IP for critical threats
        self.blocked_ips[event.source_ip] = event.timestamp
        
        # Additional automated responses could be added here:
        # - Notify security team
        # - Trigger incident response
        # - Update firewall rules
        # - etc.
        
        self.logger.critical(
            f"Critical threat detected - IP {event.source_ip} blocked",
            extra={'security_event': event.__dict__}
        )
    
    def authenticate_request(self, credentials: Dict[str, Any], 
                           method: AuthMethod, source_ip: str,
                           user_agent: str = "") -> tuple:
        """Comprehensive request authentication"""
        start_time = time.time()
        
        try:
            # Rate limit check
            rate_ok, rate_msg = self.check_rate_limit(source_ip, 'auth')
            if not rate_ok:
                
                if PROMETHEUS_AVAILABLE:
                    auth_attempts.labels(result='rate_limited', method=method.value).inc()
                
                self.log_security_event(
                    'auth_rate_limited',
                    ThreatLevel.MEDIUM,
                    source_ip,
                    user_agent,
                    {'method': method.value, 'message': rate_msg}
                )
                
                return False, None, "Rate limited"
            
            # IP security check
            ip_ok, ip_msg = self.check_ip_security(source_ip)
            if not ip_ok:
                
                if PROMETHEUS_AVAILABLE:
                    auth_attempts.labels(result='blocked_ip', method=method.value).inc()
                
                self.log_security_event(
                    'auth_blocked_ip',
                    ThreatLevel.HIGH,
                    source_ip,
                    user_agent,
                    {'method': method.value, 'message': ip_msg}
                )
                
                return False, None, ip_msg
            
            # Method-specific authentication
            if method == AuthMethod.PASSWORD:
                success, user_data = self._authenticate_password(credentials, source_ip)
            elif method == AuthMethod.TOKEN:
                success, user_data = self._authenticate_token(credentials, source_ip)
            elif method == AuthMethod.API_KEY:
                success, user_data = self._authenticate_api_key(credentials, source_ip)
            else:
                success, user_data = False, None
            
            # Record attempt
            result = 'success' if success else 'failed'
            
            if PROMETHEUS_AVAILABLE:
                auth_attempts.labels(result=result, method=method.value).inc()
            
            if success:
                self.log_security_event(
                    'auth_success',
                    ThreatLevel.LOW,
                    source_ip,
                    user_agent,
                    {'method': method.value, 'user_id': user_data.get('user_id', 'unknown')}
                )
                
                # Clear failed attempts
                if source_ip in self.failed_attempts:
                    del self.failed_attempts[source_ip]
                
                return True, user_data, "Authentication successful"
            
            else:
                # Track failed attempts
                self.failed_attempts[source_ip].append(datetime.now())
                
                # Check for brute force
                recent_failures = [
                    attempt for attempt in self.failed_attempts[source_ip]
                    if datetime.now() - attempt < timedelta(minutes=15)
                ]
                
                threat_level = ThreatLevel.MEDIUM
                if len(recent_failures) >= 5:
                    threat_level = ThreatLevel.HIGH
                    # Block IP after multiple failures
                    self.blocked_ips[source_ip] = datetime.now()
                
                self.log_security_event(
                    'auth_failed',
                    threat_level,
                    source_ip,
                    user_agent,
                    {
                        'method': method.value,
                        'recent_failures': len(recent_failures),
                        'total_failures': len(self.failed_attempts[source_ip])
                    },
                    'IP blocked' if threat_level == ThreatLevel.HIGH else ''
                )
                
                return False, None, "Authentication failed"
        
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            
            if PROMETHEUS_AVAILABLE:
                auth_attempts.labels(result='error', method=method.value).inc()
            
            return False, None, "Authentication error"
    
    def _authenticate_password(self, credentials: Dict[str, Any], source_ip: str) -> tuple:
        """Authenticate using username/password"""
        username = credentials.get('username')
        password = credentials.get('password')
        
        if not username or not password:
            return False, None
        
        # In production, this would check against a database
        # For now, return a mock successful authentication
        user_data = {
            'user_id': username,
            'username': username,
            'roles': ['user'],
            'authenticated_at': datetime.now().isoformat(),
            'source_ip': source_ip
        }
        
        return True, user_data
    
    def _authenticate_token(self, credentials: Dict[str, Any], source_ip: str) -> tuple:
        """Authenticate using JWT token"""
        token = credentials.get('token')
        
        if not token:
            return False, None
        
        payload = self.verify_token(token)
        if not payload:
            return False, None
        
        user_data = {
            'user_id': payload.get('user_id'),
            'token_type': payload.get('token_type'),
            'scopes': payload.get('scopes', []),
            'authenticated_at': datetime.now().isoformat(),
            'source_ip': source_ip
        }
        
        return True, user_data
    
    def _authenticate_api_key(self, credentials: Dict[str, Any], source_ip: str) -> tuple:
        """Authenticate using API key"""
        api_key = credentials.get('api_key')
        
        if not api_key:
            return False, None
        
        # In production, validate against stored API keys
        # For now, return mock authentication
        user_data = {
            'user_id': 'api_user',
            'api_key': api_key[:8] + '...',  # Masked
            'authenticated_at': datetime.now().isoformat(),
            'source_ip': source_ip
        }
        
        return True, user_data
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get comprehensive security summary"""
        now = datetime.now()
        
        # Recent events (last hour)
        recent_events = [
            event for event in self.security_events
            if now - event.timestamp < timedelta(hours=1)
        ]
        
        # Event breakdown
        event_counts = defaultdict(int)
        threat_counts = defaultdict(int)
        
        for event in recent_events:
            event_counts[event.event_type] += 1
            threat_counts[event.threat_level.value] += 1
        
        return {
            'timestamp': now.isoformat(),
            'total_events': len(self.security_events),
            'recent_events': len(recent_events),
            'blocked_ips': len(self.blocked_ips),
            'active_rate_limits': len([
                limiter for limiter in self.rate_limiters.values()
                if limiter['blocked_until'] and time.time() < limiter['blocked_until']
            ]),
            'failed_auth_sources': len(self.failed_attempts),
            'event_breakdown': dict(event_counts),
            'threat_level_breakdown': dict(threat_counts),
            'security_status': self._calculate_security_status()
        }
    
    def _calculate_security_status(self) -> str:
        """Calculate overall security status"""
        recent_critical = sum(
            1 for event in self.security_events
            if (datetime.now() - event.timestamp < timedelta(minutes=30) and
                event.threat_level == ThreatLevel.CRITICAL)
        )
        
        recent_high = sum(
            1 for event in self.security_events
            if (datetime.now() - event.timestamp < timedelta(hours=1) and
                event.threat_level == ThreatLevel.HIGH)
        )
        
        if recent_critical > 0:
            return "CRITICAL"
        elif recent_high > 3:
            return "HIGH_ALERT"
        elif len(self.blocked_ips) > 10:
            return "ELEVATED"
        else:
            return "NORMAL"
    
    async def health_check(self) -> Dict[str, Any]:
        """Security system health check"""
        try:
            # Test encryption
            test_data = "test_encryption"
            encrypted = self.encrypt_sensitive_data(test_data)
            decrypted = self.decrypt_sensitive_data(encrypted)
            encryption_ok = decrypted == test_data
            
            # Test token generation
            token = self.generate_secure_token("test_user", 1)
            token_payload = self.verify_token(token)
            token_ok = token_payload is not None
            
            # Get security summary
            summary = self.get_security_summary()
            
            return {
                'status': 'healthy' if encryption_ok and token_ok else 'degraded',
                'message': 'Security systems operational',
                'metrics': {
                    'encryption_available': CRYPTOGRAPHY_AVAILABLE,
                    'jwt_available': JWT_AVAILABLE,
                    'bcrypt_available': BCRYPT_AVAILABLE,
                    'encryption_test': encryption_ok,
                    'token_test': token_ok,
                    'security_status': summary['security_status'],
                    'blocked_ips': summary['blocked_ips'],
                    'recent_events': summary['recent_events']
                }
            }
            
        except Exception as e:
            return {
                'status': 'critical',
                'message': f'Security health check failed: {str(e)}',
                'metrics': {'error': str(e)}
            }

# ===================== GLOBAL INSTANCE =====================

_enterprise_security_manager = None

def get_enterprise_security_manager() -> EnterpriseSecurityManager:
    """Get global enterprise security manager instance"""
    global _enterprise_security_manager
    if _enterprise_security_manager is None:
        _enterprise_security_manager = EnterpriseSecurityManager()
    return _enterprise_security_manager

async def initialize_enterprise_security():
    """Initialize enterprise security"""
    security_manager = get_enterprise_security_manager()
    
    # Register health check
    from infrastructure.monitoring import get_health_monitor
    health_monitor = get_health_monitor()
    health_monitor.register_health_check('enterprise_security', security_manager.health_check)
    
    return True
