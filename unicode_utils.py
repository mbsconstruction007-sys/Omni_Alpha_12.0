
# unicode_utils.py
import sys
import os
from typing import Dict

class UnicodeManager:
    '''Manage Unicode display across different platforms'''
    
    def __init__(self):
        self.emoji_map = {
            # Test and system emojis
            '🧪': '[TEST]',
            '🚀': '[LAUNCH]',
            '📊': '[METRICS]',
            '✅': '[PASS]',
            '❌': '[FAIL]',
            '⚠️': '[WARN]',
            '🎉': '[SUCCESS]',
            '🔧': '[FIX]',
            '📋': '[LIST]',
            '📈': '[CHART]',
            '📉': '[DOWN]',
            '💰': '[MONEY]',
            '🏛️': '[INSTITUTIONAL]',
            '⚡': '[FAST]',
            '🛡️': '[SECURITY]',
            '🌐': '[GLOBAL]',
            '🎯': '[TARGET]',
            '🔒': '[SECURE]',
            '🏆': '[TROPHY]',
            '💡': '[IDEA]',
            '🎊': '[CELEBRATION]',
            '📱': '[MOBILE]',
            '🖥️': '[COMPUTER]',
            '🔄': '[REFRESH]',
            '⏰': '[TIME]',
            '📝': '[NOTE]',
            '🎪': '[EVENT]',
            
            # Trading specific
            '📈': '[UP]',
            '📉': '[DOWN]',
            '💹': '[TRADING]',
            '💲': '[DOLLAR]',
            '🏦': '[BANK]',
            '💳': '[CARD]',
            '💸': '[MONEY_OUT]',
            '💰': '[MONEY_IN]',
            
            # Status indicators
            '🟢': '[GREEN]',
            '🔴': '[RED]',
            '🟡': '[YELLOW]',
            '⚫': '[BLACK]',
            '🔵': '[BLUE]',
        }
        
        self.setup_unicode_support()
    
    def setup_unicode_support(self):
        '''Setup Unicode support for the current platform'''
        
        if sys.platform == 'win32':
            # Windows-specific setup
            try:
                os.system('chcp 65001 > nul 2>&1')
                os.environ['PYTHONIOENCODING'] = 'utf-8'
            except:
                pass
    
    def safe_print(self, text: str, fallback: bool = True) -> str:
        '''Safely print text with Unicode fallback'''
        
        if not fallback:
            return text
        
        # Replace emojis with safe alternatives
        safe_text = text
        for emoji, replacement in self.emoji_map.items():
            safe_text = safe_text.replace(emoji, replacement)
        
        return safe_text
    
    def format_for_console(self, text: str) -> str:
        '''Format text for console output'''
        
        # Check if console supports Unicode
        if self.supports_unicode():
            return text
        else:
            return self.safe_print(text)
    
    def supports_unicode(self) -> bool:
        '''Check if current environment supports Unicode'''
        
        try:
            # Try to encode a simple emoji
            test_emoji = '✅'
            test_emoji.encode(sys.stdout.encoding or 'utf-8')
            return True
        except (UnicodeEncodeError, LookupError):
            return False
    
    def get_safe_status_indicator(self, status: str) -> str:
        '''Get safe status indicator'''
        
        indicators = {
            'pass': '✅' if self.supports_unicode() else '[PASS]',
            'fail': '❌' if self.supports_unicode() else '[FAIL]',
            'warn': '⚠️' if self.supports_unicode() else '[WARN]',
            'info': 'ℹ️' if self.supports_unicode() else '[INFO]',
            'success': '🎉' if self.supports_unicode() else '[SUCCESS]',
            'error': '💥' if self.supports_unicode() else '[ERROR]',
        }
        
        return indicators.get(status.lower(), f'[{status.upper()}]')

# Global Unicode manager
unicode_manager = UnicodeManager()

# Convenience functions
def safe_print(text: str, fallback: bool = True) -> str:
    '''Safe print with Unicode fallback'''
    return unicode_manager.safe_print(text, fallback)

def format_for_console(text: str) -> str:
    '''Format text for console'''
    return unicode_manager.format_for_console(text)

def status_indicator(status: str) -> str:
    '''Get status indicator'''
    return unicode_manager.get_safe_status_indicator(status)

def supports_unicode() -> bool:
    '''Check Unicode support'''
    return unicode_manager.supports_unicode()
