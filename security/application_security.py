"""
LAYER 4: Application Security Layer
OWASP Top 10 Protection & Input Validation
"""

import re
import html
import urllib.parse
import json
from typing import Any, Dict, List, Optional
import logging
import hashlib
import secrets
import time

# Bleach for HTML sanitization
import bleach

logger = logging.getLogger(__name__)

class ApplicationSecurityLayer:
    """
    Comprehensive application security implementation
    OWASP Top 10 Protection
    """
    
    def __init__(self):
        self.sql_injection_patterns = self._load_sql_patterns()
        self.xss_patterns = self._load_xss_patterns()
        self.command_injection_patterns = self._load_command_patterns()
        self.input_validators = self._setup_validators()
        self.csrf_tokens = {}
        self.session_security = {}
        
    def _load_sql_patterns(self) -> List[str]:
        """Load SQL injection detection patterns"""
        
        return [
            # Basic SQL injection patterns
            r"(\s|^)(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE|UNION|FROM|WHERE)(\s|$)",
            r"(--|\#|\/\*|\*\/)",  # SQL comments
            r"(;|\||&&)",  # Command chaining
            r"(\'|\"|`)",  # Quote characters
            r"(OR|AND)\s+\d+\s*=\s*\d+",  # Always true conditions
            r"(SLEEP|BENCHMARK|WAITFOR|DELAY)",  # Time-based attacks
            r"(0x[0-9a-fA-F]+)",  # Hex encoding
            r"(CHAR|CONCAT|CHR|ASCII|SUBSTRING)",  # String manipulation
            r"(INFORMATION_SCHEMA|SYSOBJECTS|SYSCOLUMNS)",  # Schema enumeration
            r"(LOAD_FILE|INTO\s+OUTFILE|INTO\s+DUMPFILE)",  # File operations
            r"(@@VERSION|@@SERVERNAME|USER\(\)|DATABASE\(\))",  # System functions
            r"(UNION\s+ALL\s+SELECT)",  # Union-based attacks
            r"(\bOR\b\s+\b1\b\s*=\s*\b1\b)",  # Classic OR 1=1
            r"(\bAND\b\s+\b1\b\s*=\s*\b2\b)",  # Classic AND 1=2
            r"(HAVING\s+\d+\s*=\s*\d+)",  # HAVING clause attacks
        ]
    
    def _load_xss_patterns(self) -> List[str]:
        """Load XSS detection patterns"""
        
        return [
            r"<script[^>]*>.*?</script>",  # Script tags
            r"javascript:",  # JavaScript protocol
            r"on\w+\s*=",  # Event handlers
            r"<iframe[^>]*>",  # Iframe injection
            r"<object[^>]*>",  # Object tags
            r"<embed[^>]*>",  # Embed tags
            r"<link[^>]*>",  # Link injection
            r"<meta[^>]*>",  # Meta refresh
            r"data:text/html",  # Data URLs
            r"vbscript:",  # VBScript protocol
            r"<svg[^>]*onload",  # SVG with onload
            r"<img[^>]*onerror",  # Image with onerror
            r"<body[^>]*onload",  # Body onload
            r"expression\s*\(",  # CSS expressions
            r"@import",  # CSS imports
        ]
    
    def _load_command_patterns(self) -> List[str]:
        """Load command injection patterns"""
        
        return [
            r"(;|\||&&|\|\|)",  # Command separators
            r"(`|$\()",  # Command substitution
            r"(wget|curl|nc|netcat)",  # Network tools
            r"(rm|del|format|fdisk)",  # Destructive commands
            r"(cat|type|more|less)",  # File reading
            r"(ps|tasklist|netstat)",  # System enumeration
            r"(chmod|attrib|icacls)",  # Permission changes
            r"(sudo|su|runas)",  # Privilege escalation
            r"(crontab|schtasks|at)",  # Task scheduling
            r"(python|perl|ruby|php|node)",  # Interpreters
        ]
    
    def _setup_validators(self) -> Dict:
        """Setup input validators for different data types"""
        
        return {
            'email': self._validate_email,
            'phone': self._validate_phone,
            'pan': self._validate_pan,
            'aadhar': self._validate_aadhar,
            'amount': self._validate_amount,
            'symbol': self._validate_symbol,
            'numeric': self._validate_numeric,
            'alphanumeric': self._validate_alphanumeric,
            'text': self._validate_text,
            'json': self._validate_json
        }
    
    def sanitize_input(self, user_input: str, input_type: str = 'general') -> str:
        """
        Multi-layer input sanitization
        """
        
        try:
            if not isinstance(user_input, str):
                user_input = str(user_input)
            
            # Step 1: Length check
            max_length = 10000
            if len(user_input) > max_length:
                raise ValueError(f"Input too long (max {max_length} characters)")
            
            # Step 2: Null byte injection prevention
            user_input = user_input.replace('\x00', '')
            
            # Step 3: SQL Injection prevention
            if self._detect_sql_injection(user_input):
                raise ValueError("SQL injection pattern detected")
            
            # Step 4: XSS prevention
            user_input = self._prevent_xss(user_input)
            
            # Step 5: Command injection prevention
            if self._detect_command_injection(user_input):
                raise ValueError("Command injection pattern detected")
            
            # Step 6: Path traversal prevention
            if self._detect_path_traversal(user_input):
                raise ValueError("Path traversal pattern detected")
            
            # Step 7: LDAP injection prevention
            user_input = self._prevent_ldap_injection(user_input)
            
            # Step 8: XML injection prevention
            user_input = self._prevent_xml_injection(user_input)
            
            # Step 9: NoSQL injection prevention
            if self._detect_nosql_injection(user_input):
                raise ValueError("NoSQL injection pattern detected")
            
            # Step 10: Type-specific validation
            validator = self.input_validators.get(input_type)
            if validator:
                if not validator(user_input):
                    raise ValueError(f"Invalid {input_type} format")
            
            return user_input
            
        except Exception as e:
            logger.warning(f"Input sanitization failed: {e}")
            raise
    
    def _detect_sql_injection(self, user_input: str) -> bool:
        """Advanced SQL injection detection"""
        
        # Check against all SQL patterns
        for pattern in self.sql_injection_patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                logger.warning(f"SQL injection pattern detected: {pattern}")
                return True
        
        # Check for encoded attacks
        try:
            decoded_input = urllib.parse.unquote(user_input)
            if decoded_input != user_input:
                return self._detect_sql_injection(decoded_input)
        except:
            pass
        
        # Check for base64 encoded attacks
        try:
            import base64
            decoded_b64 = base64.b64decode(user_input).decode('utf-8', errors='ignore')
            if len(decoded_b64) > 10:  # Meaningful decoded content
                return self._detect_sql_injection(decoded_b64)
        except:
            pass
        
        return False
    
    def _prevent_xss(self, user_input: str) -> str:
        """XSS prevention with multiple techniques"""
        
        # Use bleach for HTML sanitization
        allowed_tags = []  # No HTML tags allowed
        allowed_attributes = {}
        
        # Clean HTML
        cleaned = bleach.clean(
            user_input,
            tags=allowed_tags,
            attributes=allowed_attributes,
            strip=True
        )
        
        # Additional encoding
        cleaned = html.escape(cleaned)
        
        # Prevent JavaScript URI schemes
        cleaned = re.sub(r'javascript:', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'data:text/html', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'vbscript:', '', cleaned, flags=re.IGNORECASE)
        
        # Remove event handlers
        event_handlers = [
            'onerror', 'onload', 'onclick', 'onmouseover', 'onmouseout',
            'onfocus', 'onblur', 'onchange', 'onsubmit', 'onreset',
            'onselect', 'onunload', 'onbeforeunload', 'onresize'
        ]
        
        for handler in event_handlers:
            cleaned = re.sub(f'{handler}\\s*=', '', cleaned, flags=re.IGNORECASE)
        
        # Remove dangerous CSS
        cleaned = re.sub(r'expression\s*\(', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'@import', '', cleaned, flags=re.IGNORECASE)
        
        return cleaned
    
    def _detect_command_injection(self, user_input: str) -> bool:
        """Command injection detection"""
        
        for pattern in self.command_injection_patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                logger.warning(f"Command injection pattern detected: {pattern}")
                return True
        
        return False
    
    def _detect_path_traversal(self, user_input: str) -> bool:
        """Path traversal detection"""
        
        path_patterns = [
            r"\.\./",  # Directory traversal
            r"\.\.\\",  # Windows directory traversal
            r"%2e%2e%2f",  # URL encoded ../
            r"%2e%2e%5c",  # URL encoded ..\
            r"..%2f",  # Mixed encoding
            r"..%5c",  # Mixed encoding
            r"/etc/passwd",  # Linux sensitive files
            r"/etc/shadow",
            r"C:\\Windows\\System32",  # Windows sensitive paths
            r"C:/Windows/System32",
        ]
        
        for pattern in path_patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                logger.warning(f"Path traversal pattern detected: {pattern}")
                return True
        
        return False
    
    def _prevent_ldap_injection(self, user_input: str) -> str:
        """LDAP injection prevention"""
        
        # Escape LDAP special characters
        ldap_escape_chars = {
            '\\': '\\5c',
            '*': '\\2a',
            '(': '\\28',
            ')': '\\29',
            '\x00': '\\00'
        }
        
        for char, escaped in ldap_escape_chars.items():
            user_input = user_input.replace(char, escaped)
        
        return user_input
    
    def _prevent_xml_injection(self, user_input: str) -> str:
        """XML injection prevention"""
        
        # Escape XML special characters
        xml_escape_chars = {
            '<': '&lt;',
            '>': '&gt;',
            '&': '&amp;',
            '"': '&quot;',
            "'": '&#x27;'
        }
        
        for char, escaped in xml_escape_chars.items():
            user_input = user_input.replace(char, escaped)
        
        # Remove XML processing instructions
        user_input = re.sub(r'<\?.*?\?>', '', user_input)
        
        # Remove CDATA sections
        user_input = re.sub(r'<!\[CDATA\[.*?\]\]>', '', user_input)
        
        return user_input
    
    def _detect_nosql_injection(self, user_input: str) -> bool:
        """NoSQL injection detection"""
        
        nosql_patterns = [
            r'\$where',  # MongoDB where clause
            r'\$regex',  # MongoDB regex
            r'\$ne',     # MongoDB not equal
            r'\$gt',     # MongoDB greater than
            r'\$lt',     # MongoDB less than
            r'\$or',     # MongoDB OR
            r'\$and',    # MongoDB AND
            r'this\.',   # JavaScript this reference
            r'function\s*\(',  # JavaScript functions
        ]
        
        for pattern in nosql_patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                logger.warning(f"NoSQL injection pattern detected: {pattern}")
                return True
        
        return False
    
    def _validate_email(self, email: str) -> bool:
        """Validate email format"""
        
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(email_pattern, email))
    
    def _validate_phone(self, phone: str) -> bool:
        """Validate phone number (Indian format)"""
        
        # Remove common separators
        phone_clean = re.sub(r'[\s\-\(\)\+]', '', phone)
        
        # Indian phone number patterns
        patterns = [
            r'^91[6-9]\d{9}$',  # +91 with mobile prefix
            r'^[6-9]\d{9}$',    # Mobile without country code
            r'^0[1-9]\d{8,9}$'  # Landline with STD code
        ]
        
        return any(re.match(pattern, phone_clean) for pattern in patterns)
    
    def _validate_pan(self, pan: str) -> bool:
        """Validate PAN number format"""
        
        pan_pattern = r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$'
        return bool(re.match(pan_pattern, pan.upper()))
    
    def _validate_aadhar(self, aadhar: str) -> bool:
        """Validate Aadhar number format"""
        
        # Remove spaces and hyphens
        aadhar_clean = re.sub(r'[\s\-]', '', aadhar)
        
        # Check format (12 digits)
        if not re.match(r'^\d{12}$', aadhar_clean):
            return False
        
        # Verify checksum (Luhn algorithm)
        return self._verify_aadhar_checksum(aadhar_clean)
    
    def _verify_aadhar_checksum(self, aadhar: str) -> bool:
        """Verify Aadhar checksum using Verhoeff algorithm"""
        
        # Simplified checksum verification
        # In production, use proper Verhoeff algorithm
        
        # Check if all digits are same (invalid)
        if len(set(aadhar)) == 1:
            return False
        
        # Basic checksum
        checksum = sum(int(digit) * (i + 1) for i, digit in enumerate(aadhar[:-1])) % 10
        return checksum == int(aadhar[-1])
    
    def _validate_amount(self, amount: str) -> bool:
        """Validate monetary amount"""
        
        try:
            amount_float = float(amount)
            
            # Check reasonable range
            if amount_float < 0:
                return False
            
            if amount_float > 10000000000:  # 1000 crores max
                return False
            
            # Check decimal places (max 2)
            if '.' in amount:
                decimal_places = len(amount.split('.')[1])
                if decimal_places > 2:
                    return False
            
            return True
            
        except ValueError:
            return False
    
    def _validate_symbol(self, symbol: str) -> bool:
        """Validate trading symbol"""
        
        # Stock symbol pattern (Indian market)
        symbol_pattern = r'^[A-Z0-9]{1,20}$'
        
        if not re.match(symbol_pattern, symbol.upper()):
            return False
        
        # Check against known symbols (simplified)
        known_symbols = {
            'NIFTY', 'BANKNIFTY', 'FINNIFTY', 'AAPL', 'MSFT', 'GOOGL', 
            'TSLA', 'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK'
        }
        
        return symbol.upper() in known_symbols or len(symbol) <= 10
    
    def _validate_numeric(self, value: str) -> bool:
        """Validate numeric input"""
        
        try:
            float(value)
            return True
        except ValueError:
            return False
    
    def _validate_alphanumeric(self, value: str) -> bool:
        """Validate alphanumeric input"""
        
        return value.isalnum()
    
    def _validate_text(self, text: str) -> bool:
        """Validate general text input"""
        
        # Check for dangerous patterns
        dangerous_patterns = [
            r'<script',
            r'javascript:',
            r'data:text/html',
            r'on\w+\s*=',
            r'eval\s*\(',
            r'exec\s*\(',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False
        
        return True
    
    def _validate_json(self, json_string: str) -> bool:
        """Validate JSON input"""
        
        try:
            parsed = json.loads(json_string)
            
            # Check for dangerous keys
            dangerous_keys = ['__proto__', 'constructor', 'prototype']
            
            def check_dangerous_keys(obj):
                if isinstance(obj, dict):
                    for key in obj.keys():
                        if key in dangerous_keys:
                            return False
                        if not check_dangerous_keys(obj[key]):
                            return False
                elif isinstance(obj, list):
                    for item in obj:
                        if not check_dangerous_keys(item):
                            return False
                return True
            
            return check_dangerous_keys(parsed)
            
        except json.JSONDecodeError:
            return False
    
    def generate_csrf_token(self, session_id: str) -> str:
        """Generate CSRF token for session"""
        
        # Generate cryptographically secure token
        token = secrets.token_urlsafe(32)
        
        # Store token with expiration
        self.csrf_tokens[session_id] = {
            'token': token,
            'created_at': time.time(),
            'expires_at': time.time() + 3600  # 1 hour
        }
        
        return token
    
    def verify_csrf_token(self, session_id: str, provided_token: str) -> bool:
        """Verify CSRF token"""
        
        token_data = self.csrf_tokens.get(session_id)
        if not token_data:
            return False
        
        # Check expiration
        if time.time() > token_data['expires_at']:
            del self.csrf_tokens[session_id]
            return False
        
        # Constant-time comparison
        import hmac
        return hmac.compare_digest(token_data['token'], provided_token)
    
    def secure_session_management(self, session_id: str) -> Dict:
        """Secure session management"""
        
        session_data = {
            'session_id': session_id,
            'created_at': time.time(),
            'last_activity': time.time(),
            'ip_address': None,  # Set by caller
            'user_agent': None,  # Set by caller
            'csrf_token': self.generate_csrf_token(session_id),
            'security_flags': {
                'secure': True,
                'http_only': True,
                'same_site': 'Strict'
            }
        }
        
        self.session_security[session_id] = session_data
        
        return session_data
    
    def validate_session_security(self, session_id: str, request_data: Dict) -> bool:
        """Validate session security"""
        
        session = self.session_security.get(session_id)
        if not session:
            return False
        
        # Check session timeout (24 hours)
        if time.time() - session['last_activity'] > 86400:
            del self.session_security[session_id]
            return False
        
        # Check IP binding
        if session['ip_address'] and session['ip_address'] != request_data.get('ip_address'):
            logger.warning(f"Session IP mismatch for {session_id}")
            return False
        
        # Check User-Agent binding
        if session['user_agent'] and session['user_agent'] != request_data.get('user_agent'):
            logger.warning(f"Session User-Agent mismatch for {session_id}")
            return False
        
        # Update last activity
        session['last_activity'] = time.time()
        
        return True
    
    def secure_api_endpoint(self, endpoint_func):
        """
        Decorator for API endpoint security
        """
        
        def wrapper(*args, **kwargs):
            request = args[0] if args else kwargs.get('request')
            
            try:
                # Security checks
                self._perform_security_checks(request)
                
                # Execute endpoint
                result = endpoint_func(*args, **kwargs)
                
                # Secure output
                result = self._secure_output(result)
                
                return result
                
            except SecurityException as e:
                logger.warning(f"Security check failed: {e}")
                raise
            except Exception as e:
                logger.error(f"Endpoint error: {e}")
                raise
        
        return wrapper
    
    def _perform_security_checks(self, request: Dict):
        """Perform comprehensive security checks"""
        
        # Authentication check
        if not self._verify_authentication(request):
            raise AuthenticationError("Authentication required")
        
        # Authorization check
        if not self._verify_authorization(request):
            raise AuthorizationError("Insufficient permissions")
        
        # Rate limiting
        if not self._check_rate_limit(request):
            raise RateLimitError("Rate limit exceeded")
        
        # Input validation
        self._validate_all_inputs(request)
        
        # CSRF protection
        if request.get('method') in ['POST', 'PUT', 'DELETE']:
            if not self.verify_csrf_token(
                request.get('session_id'),
                request.get('csrf_token')
            ):
                raise CSRFError("CSRF token validation failed")
    
    def _validate_all_inputs(self, request: Dict):
        """Validate all request inputs"""
        
        for key, value in request.items():
            if isinstance(value, str):
                # Determine input type based on key name
                input_type = self._determine_input_type(key)
                request[key] = self.sanitize_input(value, input_type)
    
    def _determine_input_type(self, field_name: str) -> str:
        """Determine input type based on field name"""
        
        type_mapping = {
            'email': 'email',
            'phone': 'phone',
            'pan': 'pan',
            'aadhar': 'aadhar',
            'amount': 'amount',
            'symbol': 'symbol',
            'quantity': 'numeric',
            'price': 'amount'
        }
        
        for pattern, input_type in type_mapping.items():
            if pattern in field_name.lower():
                return input_type
        
        return 'text'
    
    def _secure_output(self, result: Any) -> Any:
        """Secure API output"""
        
        if isinstance(result, dict):
            # Remove sensitive fields
            sensitive_fields = ['password', 'secret', 'key', 'token']
            
            for field in sensitive_fields:
                if field in result:
                    result[field] = '[REDACTED]'
            
            # Encode output
            for key, value in result.items():
                if isinstance(value, str):
                    result[key] = html.escape(value)
        
        return result
    
    def get_security_headers(self) -> Dict[str, str]:
        """Get security headers for HTTP responses"""
        
        return {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'",
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Feature-Policy': "camera 'none'; microphone 'none'; geolocation 'none'",
            'X-Permitted-Cross-Domain-Policies': 'none',
            'Cache-Control': 'no-store, no-cache, must-revalidate, private',
            'Pragma': 'no-cache',
            'Expires': '0'
        }

# Custom security exceptions
class SecurityException(Exception):
    pass

class AuthenticationError(SecurityException):
    pass

class AuthorizationError(SecurityException):
    pass

class RateLimitError(SecurityException):
    pass

class CSRFError(SecurityException):
    pass

# Global application security instance
app_security = ApplicationSecurityLayer()
