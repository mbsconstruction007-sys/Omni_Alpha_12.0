
# encoding_config.py
import sys
import locale
import os

def setup_encoding():
    '''Setup proper encoding for Windows compatibility'''
    
    # Set console encoding
    if sys.platform == 'win32':
        os.system('chcp 65001 > nul')
        
    # Set locale
    try:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except:
        try:
            locale.setlocale(locale.LC_ALL, 'C.UTF-8')
        except:
            pass
    
    # Set default encoding
    if hasattr(sys, 'set_int_max_str_digits'):
        sys.set_int_max_str_digits(0)

# Auto-setup when imported
setup_encoding()
