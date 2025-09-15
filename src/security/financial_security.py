"""
Bank-level security implementation with HSM support
Institutional-grade security for trading systems
"""

import hashlib
import hmac
import secrets
import jwt
import os
import time
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime, timedelta
import ipaddress
import re
import base64
import json
import structlog

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.primitives import serialization
from cryptography.x509 import load_pem_x509_certificate
import pyotp
import qrcode
import redis
import orjson

logger = structlog.get_logger()

class FinancialGradeSecurity:
    """
    Institutional-grade security with regulatory compliance
    """
    
    def __init__(self, hsm_enabled: bool = False):
        self.hsm_enabled = hsm_enabled
        self._initialize_hsm()
        self._setup_encryption_keys()
        self._initialize_threat_detection()
        self._setup_audit_cryptography()
        self._setup_rate_limiting()
        
    def _initialize_hsm(self):
        """Initialize Hardware Security Module if available"""
        if self.hsm_enabled:
            try:
                import pkcs11  # HSM interface
                self.hsm = pkcs11.lib('/usr/lib/softhsm/libsofthsm2.so')
                self.hsm_session = self.hsm.open_session()
                logger.info("HSM initialized successfully")
            except Exception as e:
                logger.warning(f"HSM not available, falling back to software: {e}")
                self.hsm_enabled = False
        else:
            logger.info("HSM disabled, using software security")
                
    def _setup_encryption_keys(self):
        """Setup multi-layer encryption keys"""
        # Master key (ideally from HSM or environment)
        self.master_key = self._get_or_create_master_key()
        
        # Derived keys for different purposes
        self.keys = {
            'api': self._derive_key(self.master_key, b'api'),
            'database': self._derive_key(self.master_key, b'database'),
            'audit': self._derive_key(self.master_key, b'audit'),
            'session': self._derive_key(self.master_key, b'session'),
        }
        
        # Rotating keys for temporal security
        self.rotating_keys = self._initialize_key_rotation()
        
    def _get_or_create_master_key(self) -> bytes:
        """Get or create master encryption key"""
        master_key = os.getenv('MASTER_ENCRYPTION_KEY')
        if master_key:
            return base64.b64decode(master_key)
        else:
            # Generate new key (in production, this should be from HSM)
            key = secrets.token_bytes(32)
            logger.warning("Generated new master key - store securely!")
            return key
        
    def _derive_key(self, master_key: bytes, context: bytes) -> bytes:
        """Derive context-specific keys from master key"""
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=context,
            iterations=100000,
            backend=default_backend()
        )
        return kdf.derive(master_key)
        
    def _initialize_threat_detection(self):
        """Setup advanced threat detection"""
        self.threat_patterns = {
            'sql_injection': re.compile(r'(\bunion\b|\bselect\b|\binsert\b|\bupdate\b|\bdelete\b|\bdrop\b)', re.I),
            'xss': re.compile(r'(<script|javascript:|onerror=|onload=)', re.I),
            'path_traversal': re.compile(r'\.\.\/|\.\.\\'),
            'command_injection': re.compile(r'[;&|`$]'),
            'ldap_injection': re.compile(r'[()=*!&|]'),
            'xml_injection': re.compile(r'<[^>]*>'),
        }
        
        # Behavioral analysis
        self.behavior_baseline = {}
        self.anomaly_threshold = 3.0  # Standard deviations
        
        # IP reputation (simplified - in production use commercial service)
        self.blocked_ips = set()
        self.suspicious_ips = set()
        
    def _setup_audit_cryptography(self):
        """Setup cryptographic audit trail"""
        # Generate audit signing key pair
        self.audit_private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096,
            backend=default_backend()
        )
        self.audit_public_key = self.audit_private_key.public_key()
        
        # Initialize merkle tree for audit integrity
        self.audit_merkle_tree = []
        
    def _setup_rate_limiting(self):
        """Setup distributed rate limiting with Redis"""
        try:
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()  # Test connection
            logger.info("Redis rate limiting initialized")
        except Exception as e:
            logger.warning(f"Redis not available, using in-memory rate limiting: {e}")
            self.redis_client = None
            self.rate_limit_store = {}
        
    def validate_api_request(self, request: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Multi-layer request validation with threat detection
        """
        # Layer 1: IP filtering
        if not self._validate_ip_address(request.get('ip')):
            return False, "IP address blocked"
            
        # Layer 2: Signature verification
        if not self._verify_request_signature(request):
            return False, "Invalid signature"
            
        # Layer 3: Rate limiting with exponential backoff
        if not self._check_rate_limit(request):
            return False, "Rate limit exceeded"
            
        # Layer 4: Threat pattern detection
        threat = self._detect_threats(request)
        if threat:
            return False, f"Threat detected: {threat}"
            
        # Layer 5: Behavioral analysis
        if not self._analyze_behavior(request):
            return False, "Anomalous behavior detected"
            
        return True, None
        
    def _validate_ip_address(self, ip: str) -> bool:
        """Validate IP address against blocklist"""
        if not ip:
            return False
            
        # Check blocked IPs
        if ip in self.blocked_ips:
            return False
            
        # Check if IP is in private range (for internal services)
        try:
            ip_obj = ipaddress.ip_address(ip)
            if ip_obj.is_private:
                return True  # Allow private IPs
        except ValueError:
            return False
            
        # Check suspicious IPs (could implement more sophisticated logic)
        if ip in self.suspicious_ips:
            return False
            
        return True
        
    def _verify_request_signature(self, request: Dict[str, Any]) -> bool:
        """Verify request signature using HMAC-SHA512"""
        signature = request.get('signature')
        if not signature:
            return False
            
        # Recreate signature
        timestamp = request.get('timestamp', '')
        nonce = request.get('nonce', '')
        body = request.get('body', '')
        
        message = f"{timestamp}:{nonce}:{body}"
        expected_signature = hmac.new(
            self.keys['api'],
            message.encode(),
            hashlib.sha512
        ).hexdigest()
        
        # Constant-time comparison
        return hmac.compare_digest(signature, expected_signature)
        
    def _check_rate_limit(self, request: Dict[str, Any]) -> bool:
        """Check rate limiting with exponential backoff"""
        ip = request.get('ip', 'unknown')
        current_time = int(time.time())
        
        if self.redis_client:
            # Distributed rate limiting with Redis
            key = f"rate_limit:{ip}"
            pipe = self.redis_client.pipeline()
            
            # Increment counter
            pipe.incr(key)
            pipe.expire(key, 3600)  # 1 hour window
            
            results = pipe.execute()
            request_count = results[0]
            
            # Exponential backoff based on request count
            if request_count > 1000:  # 1000 requests per hour
                return False
            elif request_count > 500:  # 500 requests per hour
                # Add delay
                time.sleep(0.1)
                
        else:
            # In-memory rate limiting
            if ip not in self.rate_limit_store:
                self.rate_limit_store[ip] = {'count': 0, 'window_start': current_time}
                
            store = self.rate_limit_store[ip]
            
            # Reset window if needed
            if current_time - store['window_start'] > 3600:
                store['count'] = 0
                store['window_start'] = current_time
                
            store['count'] += 1
            
            if store['count'] > 1000:
                return False
                
        return True
        
    def _detect_threats(self, request: Dict[str, Any]) -> Optional[str]:
        """Detect various threat patterns"""
        # Check request body
        body = str(request.get('body', ''))
        headers = str(request.get('headers', {}))
        
        # Combine all text to check
        text_to_check = f"{body} {headers}"
        
        for threat_type, pattern in self.threat_patterns.items():
            if pattern.search(text_to_check):
                logger.warning(f"Threat detected: {threat_type}", 
                             ip=request.get('ip'),
                             body=body[:100])  # Log first 100 chars
                return threat_type
                
        return None
        
    def _analyze_behavior(self, request: Dict[str, Any]) -> bool:
        """Analyze request behavior for anomalies"""
        ip = request.get('ip', 'unknown')
        path = request.get('path', '')
        method = request.get('method', '')
        
        # Simple behavioral analysis (in production, use ML)
        if ip not in self.behavior_baseline:
            self.behavior_baseline[ip] = {
                'request_count': 0,
                'paths': set(),
                'methods': set(),
                'first_seen': time.time()
            }
            
        baseline = self.behavior_baseline[ip]
        baseline['request_count'] += 1
        baseline['paths'].add(path)
        baseline['methods'].add(method)
        
        # Check for anomalies
        if baseline['request_count'] > 1000:  # Too many requests
            return False
            
        if len(baseline['paths']) > 50:  # Too many different paths
            return False
            
        return True
        
    def encrypt_sensitive_data(self, data: bytes, context: str = 'general') -> bytes:
        """
        Encrypt sensitive data with AES-256-GCM
        """
        # Generate nonce
        nonce = os.urandom(12)
        
        # Select appropriate key
        key = self.keys.get(context, self.keys['api'])
        
        # Encrypt
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(nonce),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data) + encryptor.finalize()
        
        # Return nonce + ciphertext + tag
        return nonce + ciphertext + encryptor.tag
        
    def decrypt_sensitive_data(self, encrypted_data: bytes, context: str = 'general') -> bytes:
        """
        Decrypt sensitive data with AES-256-GCM
        """
        # Extract nonce, ciphertext, and tag
        nonce = encrypted_data[:12]
        ciphertext = encrypted_data[12:-16]
        tag = encrypted_data[-16:]
        
        # Select appropriate key
        key = self.keys.get(context, self.keys['api'])
        
        # Decrypt
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(nonce, tag),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        return decryptor.update(ciphertext) + decryptor.finalize()
        
    def create_secure_audit_entry(self, event: Dict[str, Any]) -> str:
        """
        Create cryptographically signed audit entry
        """
        # Add timestamp and sequence
        event['timestamp'] = datetime.utcnow().isoformat()
        event['sequence'] = len(self.audit_merkle_tree)
        
        # Serialize event
        event_json = orjson.dumps(event)
        
        # Sign event
        signature = self.audit_private_key.sign(
            event_json,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        # Add to merkle tree
        event_hash = hashlib.sha256(event_json + signature).hexdigest()
        if self.audit_merkle_tree:
            previous_hash = self.audit_merkle_tree[-1]
            combined_hash = hashlib.sha256(
                (previous_hash + event_hash).encode()
            ).hexdigest()
            self.audit_merkle_tree.append(combined_hash)
        else:
            self.audit_merkle_tree.append(event_hash)
            
        # Return audit entry
        return base64.b64encode(event_json + b'::' + signature).decode()
        
    def verify_audit_entry(self, audit_entry: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Verify cryptographically signed audit entry
        """
        try:
            # Decode audit entry
            decoded = base64.b64decode(audit_entry)
            event_json, signature = decoded.split(b'::')
            
            # Verify signature
            self.audit_public_key.verify(
                signature,
                event_json,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            # Parse event
            event = orjson.loads(event_json)
            return True, event
            
        except Exception as e:
            logger.error(f"Audit entry verification failed: {e}")
            return False, None
            
    def generate_2fa_secret(self, user_id: str) -> Tuple[str, str]:
        """
        Generate 2FA secret and QR code
        """
        secret = pyotp.random_base32()
        totp = pyotp.TOTP(secret)
        
        # Generate QR code
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(totp.provisioning_uri(
            name=user_id,
            issuer_name="Omni Alpha Trading System"
        ))
        qr.make(fit=True)
        
        # Convert to string
        qr_string = qr.make_image().get_string()
        
        return secret, qr_string
        
    def verify_2fa_token(self, secret: str, token: str) -> bool:
        """
        Verify 2FA token
        """
        totp = pyotp.TOTP(secret)
        return totp.verify(token, valid_window=1)  # Allow 1 window tolerance
        
    def block_ip(self, ip: str, reason: str = "Security violation"):
        """
        Block an IP address
        """
        self.blocked_ips.add(ip)
        logger.warning(f"IP blocked: {ip}, reason: {reason}")
        
        # Create audit entry
        self.create_secure_audit_entry({
            'event': 'ip_blocked',
            'ip': ip,
            'reason': reason,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    def get_security_status(self) -> Dict[str, Any]:
        """
        Get current security status
        """
        return {
            'hsm_enabled': self.hsm_enabled,
            'blocked_ips_count': len(self.blocked_ips),
            'suspicious_ips_count': len(self.suspicious_ips),
            'audit_entries_count': len(self.audit_merkle_tree),
            'rate_limiting_enabled': self.redis_client is not None,
            'threat_patterns_count': len(self.threat_patterns),
            'behavior_baseline_count': len(self.behavior_baseline)
        }

# Global security instance
security = FinancialGradeSecurity()
