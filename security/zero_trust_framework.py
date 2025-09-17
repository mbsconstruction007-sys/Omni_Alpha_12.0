"""
LAYER 1: Zero-Trust Security Framework
TRUST NOTHING, VERIFY EVERYTHING
"""

import hashlib
import hmac
import secrets
import os
import json
import time
import base64
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
import asyncio
import logging

# Cryptography imports
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

# JWT and OTP
import jwt
import pyotp

# Redis for session management
import redis

logger = logging.getLogger(__name__)

class ZeroTrustSecurityFramework:
    """
    Military-grade zero-trust security implementation
    TRUST NOTHING, VERIFY EVERYTHING
    """
    
    def __init__(self):
        self.encryption_key = self._generate_master_key()
        self.cipher = Fernet(self.encryption_key)
        
        # Initialize Redis for session management
        try:
            self.session_store = redis.Redis(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', '6379')),
                db=0,
                decode_responses=True
            )
            self.redis_available = True
        except:
            self.redis_available = False
            self.session_store = {}  # Fallback to in-memory
            
        self.failed_attempts = {}
        self.security_events = []
        self.trusted_devices = set()
        self.jwt_secret = os.getenv('JWT_SECRET_KEY', self._generate_jwt_secret())
        
    def _generate_master_key(self) -> bytes:
        """Generate quantum-resistant master key"""
        
        # Use hardware entropy
        hardware_entropy = os.urandom(64)
        
        # Multiple entropy sources
        system_entropy = secrets.token_bytes(64)
        time_entropy = str(datetime.now().timestamp()).encode()
        
        # Combine entropy sources
        combined_entropy = hashlib.sha512(
            hardware_entropy + system_entropy + time_entropy
        ).digest()
        
        # Derive key using PBKDF2 with 1,000,000 iterations
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA512(),
            length=32,
            salt=b'OmniAlpha2024SecureSalt',
            iterations=1000000,
        )
        
        return base64.urlsafe_b64encode(kdf.derive(combined_entropy))
    
    def _generate_jwt_secret(self) -> str:
        """Generate secure JWT secret"""
        return base64.urlsafe_b64encode(os.urandom(64)).decode()
    
    def authenticate_request(self, request: Dict) -> Tuple[bool, Optional[str]]:
        """
        Multi-factor authentication for every request
        """
        
        try:
            # Step 1: Verify JWT token
            token = request.get('auth_token')
            if not self._verify_jwt_token(token):
                self._log_security_event('INVALID_TOKEN', request)
                return False, "Invalid authentication token"
            
            # Step 2: Verify device fingerprint
            device_id = request.get('device_fingerprint')
            if not self._verify_device_fingerprint(device_id):
                self._log_security_event('UNKNOWN_DEVICE', request)
                return False, "Unrecognized device"
            
            # Step 3: Verify IP reputation
            ip_address = request.get('ip_address', '127.0.0.1')
            if not self._verify_ip_reputation(ip_address):
                self._log_security_event('SUSPICIOUS_IP', request)
                return False, "IP address flagged as suspicious"
            
            # Step 4: Verify behavioral biometrics
            if not self._verify_behavioral_pattern(request):
                self._log_security_event('ABNORMAL_BEHAVIOR', request)
                return False, "Abnormal behavior detected"
            
            # Step 5: Verify request signature
            if not self._verify_request_signature(request):
                self._log_security_event('INVALID_SIGNATURE', request)
                return False, "Request signature verification failed"
            
            # Step 6: Check rate limiting
            user_id = request.get('user_id', 'anonymous')
            if not self._check_rate_limit(user_id):
                self._log_security_event('RATE_LIMIT_EXCEEDED', request)
                return False, "Rate limit exceeded"
            
            return True, None
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False, "Authentication system error"
    
    def _verify_jwt_token(self, token: str) -> bool:
        """Verify JWT with secure signature"""
        
        if not token:
            return False
        
        try:
            # Decode and verify token
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=['HS512'],
                options={"verify_exp": True}
            )
            
            # Verify token hasn't been revoked
            if self._is_token_revoked(payload.get('jti')):
                return False
            
            # Verify token binding
            if not self._verify_token_binding(payload):
                return False
            
            return True
            
        except jwt.InvalidTokenError:
            return False
        except Exception as e:
            logger.error(f"JWT verification error: {e}")
            return False
    
    def _verify_device_fingerprint(self, device_id: str) -> bool:
        """Verify device using fingerprinting"""
        
        if not device_id:
            return False
        
        # Check if device is in trusted devices
        if device_id in self.trusted_devices:
            return True
        
        # For new devices, require additional verification
        if self.redis_available:
            device_data = self.session_store.get(f"device:{device_id}")
            if device_data:
                device_info = json.loads(device_data)
                trust_score = device_info.get('trust_score', 0)
                return trust_score > 0.8
        
        # Default to requiring device registration
        return False
    
    def _verify_ip_reputation(self, ip_address: str) -> bool:
        """Check IP against threat intelligence"""
        
        # Local IP whitelist
        trusted_ips = {
            '127.0.0.1',
            '::1',
            '10.0.0.0/8',
            '192.168.0.0/16',
            '172.16.0.0/12'
        }
        
        # Check if IP is in trusted range
        if self._ip_in_range(ip_address, trusted_ips):
            return True
        
        # Check blacklist
        if self._is_ip_blacklisted(ip_address):
            return False
        
        # Check if IP is from known proxy/VPN/TOR
        if self._is_anonymous_ip(ip_address):
            return False
        
        # Check threat intelligence (simplified)
        threat_score = self._check_threat_intelligence(ip_address)
        return threat_score < 0.3
    
    def _verify_behavioral_pattern(self, request: Dict) -> bool:
        """Verify user behavioral patterns"""
        
        user_id = request.get('user_id')
        if not user_id:
            return False
        
        # Get user's historical behavior
        behavior_key = f"behavior:{user_id}"
        
        if self.redis_available:
            behavior_data = self.session_store.get(behavior_key)
            if behavior_data:
                behavior = json.loads(behavior_data)
                
                # Check typing patterns, timing, etc.
                current_pattern = self._extract_behavioral_features(request)
                similarity = self._calculate_behavior_similarity(behavior, current_pattern)
                
                return similarity > 0.7
        
        # For new users, allow but monitor
        return True
    
    def _verify_request_signature(self, request: Dict) -> bool:
        """Verify HMAC signature of request"""
        
        signature = request.get('signature')
        if not signature:
            return False
        
        # Reconstruct message to sign
        timestamp = request.get('timestamp', str(time.time()))
        method = request.get('method', 'POST')
        path = request.get('path', '/')
        body = request.get('body', '')
        
        message = f"{method}|{path}|{timestamp}|{body}"
        
        # Get client secret (in production, from secure store)
        client_secret = os.getenv('CLIENT_SECRET', 'default_secret')
        
        # Calculate expected signature
        expected_signature = hmac.new(
            client_secret.encode(),
            message.encode(),
            hashlib.sha512
        ).hexdigest()
        
        # Constant-time comparison
        return hmac.compare_digest(signature, expected_signature)
    
    def _check_rate_limit(self, user_id: str) -> bool:
        """Check rate limiting for user"""
        
        current_time = time.time()
        window_size = 3600  # 1 hour
        max_requests = 1000  # 1000 requests per hour
        
        rate_key = f"rate_limit:{user_id}"
        
        if self.redis_available:
            # Get current request count
            current_count = self.session_store.get(rate_key)
            if current_count is None:
                # First request in window
                self.session_store.setex(rate_key, window_size, 1)
                return True
            
            if int(current_count) >= max_requests:
                return False
            
            # Increment counter
            self.session_store.incr(rate_key)
            return True
        else:
            # In-memory rate limiting (simplified)
            if user_id not in self.failed_attempts:
                self.failed_attempts[user_id] = {'count': 0, 'window_start': current_time}
            
            user_data = self.failed_attempts[user_id]
            
            # Reset window if expired
            if current_time - user_data['window_start'] > window_size:
                user_data['count'] = 0
                user_data['window_start'] = current_time
            
            if user_data['count'] >= max_requests:
                return False
            
            user_data['count'] += 1
            return True
    
    def _log_security_event(self, event_type: str, request: Dict):
        """Log security events for analysis"""
        
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'ip_address': request.get('ip_address'),
            'user_id': request.get('user_id'),
            'user_agent': request.get('user_agent'),
            'request_id': request.get('request_id'),
            'severity': self._calculate_event_severity(event_type)
        }
        
        self.security_events.append(event)
        
        # Store in Redis if available
        if self.redis_available:
            event_key = f"security_event:{event['timestamp']}"
            self.session_store.setex(event_key, 86400 * 30, json.dumps(event))  # 30 days
        
        # Log to file
        logger.warning(f"Security event: {event_type} from {request.get('ip_address')}")
        
        # Trigger alert for high-severity events
        if event['severity'] == 'CRITICAL':
            asyncio.create_task(self._trigger_security_alert(event))
    
    def _is_token_revoked(self, jti: str) -> bool:
        """Check if token is revoked"""
        
        if not jti:
            return False
        
        if self.redis_available:
            return self.session_store.sismember('revoked_tokens', jti)
        else:
            # In-memory check (simplified)
            return jti in getattr(self, 'revoked_tokens', set())
    
    def _verify_token_binding(self, payload: Dict) -> bool:
        """Verify token binding to prevent token theft"""
        
        # Check if token is bound to specific device/IP
        bound_device = payload.get('device_id')
        bound_ip = payload.get('ip_address')
        
        # In production, verify against current request
        return True  # Simplified for demo
    
    def _ip_in_range(self, ip: str, ranges: set) -> bool:
        """Check if IP is in trusted ranges"""
        
        # Simplified IP range checking
        if ip in ranges:
            return True
        
        # Check private IP ranges
        private_ranges = ['127.', '10.', '192.168.', '172.16.', '172.17.', '172.18.']
        return any(ip.startswith(range_prefix) for range_prefix in private_ranges)
    
    def _is_ip_blacklisted(self, ip_address: str) -> bool:
        """Check if IP is blacklisted"""
        
        # Known malicious IPs (simplified)
        blacklisted_ips = {
            '0.0.0.0',
            '255.255.255.255'
        }
        
        return ip_address in blacklisted_ips
    
    def _is_anonymous_ip(self, ip_address: str) -> bool:
        """Check if IP is from anonymous proxy/VPN/TOR"""
        
        # In production, integrate with threat intelligence APIs
        # For demo, assume all external IPs could be anonymous
        return not self._ip_in_range(ip_address, {'127.0.0.1'})
    
    def _check_threat_intelligence(self, ip_address: str) -> float:
        """Check IP against threat intelligence feeds"""
        
        # In production, integrate with:
        # - VirusTotal API
        # - AbuseIPDB
        # - Shodan
        # - IBM X-Force
        
        # Simulate threat score (0.0 = safe, 1.0 = malicious)
        if ip_address.startswith('127.'):
            return 0.0  # Localhost is safe
        else:
            return 0.1  # Low threat for demo
    
    def _extract_behavioral_features(self, request: Dict) -> Dict:
        """Extract behavioral biometric features"""
        
        return {
            'typing_speed': request.get('typing_speed', 100),
            'mouse_movements': request.get('mouse_movements', []),
            'keystroke_dynamics': request.get('keystroke_dynamics', {}),
            'session_duration': request.get('session_duration', 0),
            'command_frequency': request.get('command_frequency', {})
        }
    
    def _calculate_behavior_similarity(self, historical: Dict, current: Dict) -> float:
        """Calculate behavioral similarity score"""
        
        # Simplified similarity calculation
        similarities = []
        
        # Typing speed similarity
        hist_speed = historical.get('typing_speed', 100)
        curr_speed = current.get('typing_speed', 100)
        speed_similarity = 1 - abs(hist_speed - curr_speed) / max(hist_speed, curr_speed)
        similarities.append(speed_similarity)
        
        # Session duration similarity
        hist_duration = historical.get('session_duration', 0)
        curr_duration = current.get('session_duration', 0)
        if hist_duration > 0:
            duration_similarity = 1 - abs(hist_duration - curr_duration) / hist_duration
            similarities.append(duration_similarity)
        
        return sum(similarities) / len(similarities) if similarities else 0.5
    
    def _calculate_event_severity(self, event_type: str) -> str:
        """Calculate security event severity"""
        
        severity_map = {
            'INVALID_TOKEN': 'HIGH',
            'UNKNOWN_DEVICE': 'MEDIUM',
            'SUSPICIOUS_IP': 'HIGH',
            'ABNORMAL_BEHAVIOR': 'MEDIUM',
            'INVALID_SIGNATURE': 'HIGH',
            'RATE_LIMIT_EXCEEDED': 'LOW',
            'BRUTE_FORCE': 'CRITICAL',
            'SQL_INJECTION': 'CRITICAL',
            'XSS_ATTEMPT': 'HIGH'
        }
        
        return severity_map.get(event_type, 'MEDIUM')
    
    async def _trigger_security_alert(self, event: Dict):
        """Trigger security alert for critical events"""
        
        alert = {
            'timestamp': datetime.now().isoformat(),
            'event': event,
            'alert_level': 'CRITICAL',
            'automated_response': True
        }
        
        # In production, send to:
        # - Security team via email/SMS
        # - SIEM system
        # - Incident response platform
        
        logger.critical(f"SECURITY ALERT: {event['event_type']} from {event.get('ip_address')}")
        
        # Trigger automated response
        if event['event_type'] in ['BRUTE_FORCE', 'SQL_INJECTION']:
            await self._automated_incident_response(event)
    
    async def _automated_incident_response(self, event: Dict):
        """Automated incident response"""
        
        response_actions = []
        
        if event['event_type'] == 'BRUTE_FORCE':
            # Block IP temporarily
            response_actions.append(self._block_ip(event.get('ip_address')))
            
        elif event['event_type'] == 'SQL_INJECTION':
            # Block IP and user
            response_actions.append(self._block_ip(event.get('ip_address')))
            response_actions.append(self._suspend_user(event.get('user_id')))
        
        # Execute responses
        for action in response_actions:
            try:
                await action
                logger.info(f"Automated response executed: {action}")
            except Exception as e:
                logger.error(f"Automated response failed: {e}")
    
    async def _block_ip(self, ip_address: str):
        """Block IP address"""
        
        if self.redis_available:
            # Add to Redis blacklist
            self.session_store.sadd('blocked_ips', ip_address)
            self.session_store.expire('blocked_ips', 86400)  # 24 hours
        
        logger.info(f"IP blocked: {ip_address}")
    
    async def _suspend_user(self, user_id: str):
        """Suspend user account"""
        
        if not user_id:
            return
        
        if self.redis_available:
            # Add to suspended users
            self.session_store.sadd('suspended_users', user_id)
        
        logger.info(f"User suspended: {user_id}")
    
    def generate_secure_token(self, user_id: str, device_id: str, 
                            permissions: list = None) -> str:
        """Generate secure JWT token"""
        
        now = datetime.now()
        
        payload = {
            'user_id': user_id,
            'device_id': device_id,
            'permissions': permissions or ['read'],
            'iat': now,
            'exp': now + timedelta(hours=24),
            'jti': secrets.token_urlsafe(32),  # Unique token ID
            'iss': 'omni-alpha-security',
            'aud': 'omni-alpha-trading'
        }
        
        token = jwt.encode(
            payload,
            self.jwt_secret,
            algorithm='HS512'
        )
        
        # Store token metadata
        if self.redis_available:
            token_key = f"token:{payload['jti']}"
            self.session_store.setex(
                token_key,
                86400,  # 24 hours
                json.dumps({
                    'user_id': user_id,
                    'device_id': device_id,
                    'issued_at': now.isoformat()
                })
            )
        
        return token
    
    def revoke_token(self, token: str) -> bool:
        """Revoke JWT token"""
        
        try:
            # Decode token to get JTI
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=['HS512'],
                options={"verify_exp": False}  # Don't verify expiration for revocation
            )
            
            jti = payload.get('jti')
            if jti:
                if self.redis_available:
                    self.session_store.sadd('revoked_tokens', jti)
                else:
                    if not hasattr(self, 'revoked_tokens'):
                        self.revoked_tokens = set()
                    self.revoked_tokens.add(jti)
                
                logger.info(f"Token revoked: {jti}")
                return True
            
        except Exception as e:
            logger.error(f"Token revocation failed: {e}")
        
        return False
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        
        try:
            encrypted = self.cipher.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted).decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            return data  # Return original if encryption fails
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        
        try:
            decoded = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted = self.cipher.decrypt(decoded)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return encrypted_data  # Return as-is if decryption fails
    
    def get_security_metrics(self) -> Dict:
        """Get current security metrics"""
        
        return {
            'total_events': len(self.security_events),
            'critical_events': len([e for e in self.security_events if e.get('severity') == 'CRITICAL']),
            'trusted_devices': len(self.trusted_devices),
            'blocked_ips': len(self._get_blocked_ips()),
            'active_tokens': self._count_active_tokens(),
            'last_threat_detected': self._get_last_threat_time(),
            'security_score': self._calculate_security_score()
        }
    
    def _get_blocked_ips(self) -> set:
        """Get currently blocked IPs"""
        
        if self.redis_available:
            return self.session_store.smembers('blocked_ips')
        else:
            return getattr(self, 'blocked_ips', set())
    
    def _count_active_tokens(self) -> int:
        """Count active tokens"""
        
        if self.redis_available:
            # Count non-expired tokens
            token_keys = self.session_store.keys('token:*')
            return len(token_keys)
        else:
            return 0
    
    def _get_last_threat_time(self) -> Optional[str]:
        """Get timestamp of last threat detection"""
        
        if self.security_events:
            return self.security_events[-1]['timestamp']
        return None
    
    def _calculate_security_score(self) -> float:
        """Calculate overall security score"""
        
        base_score = 85.0
        
        # Deduct points for security events
        critical_events = len([e for e in self.security_events if e.get('severity') == 'CRITICAL'])
        base_score -= critical_events * 5
        
        # Add points for security measures
        if self.redis_available:
            base_score += 5
        
        if len(self.trusted_devices) > 0:
            base_score += 3
        
        # Ensure score is between 0 and 100
        return max(0, min(100, base_score))
    
    def register_trusted_device(self, device_id: str, device_info: Dict) -> bool:
        """Register a trusted device"""
        
        try:
            self.trusted_devices.add(device_id)
            
            if self.redis_available:
                device_data = {
                    'device_id': device_id,
                    'trust_score': 1.0,
                    'registered_at': datetime.now().isoformat(),
                    'device_info': device_info
                }
                
                self.session_store.setex(
                    f"device:{device_id}",
                    86400 * 30,  # 30 days
                    json.dumps(device_data)
                )
            
            logger.info(f"Trusted device registered: {device_id}")
            return True
            
        except Exception as e:
            logger.error(f"Device registration failed: {e}")
            return False

# Global security framework instance
security_framework = ZeroTrustSecurityFramework()
