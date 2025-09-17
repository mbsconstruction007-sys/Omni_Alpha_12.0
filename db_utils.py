
# db_utils.py
import sqlite3
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class DatabaseManager:
    '''Database management with fallback to SQLite'''
    
    def __init__(self, db_path: str = 'omni_alpha.db'):
        self.db_path = Path(db_path)
        self.connection = None
        self.setup_database()
    
    def setup_database(self):
        '''Setup SQLite database with required tables'''
        
        try:
            self.connection = sqlite3.connect(str(self.db_path))
            self.connection.row_factory = sqlite3.Row  # Enable dict-like access
            
            # Create tables
            self.create_tables()
            logger.info(f"Database initialized: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            raise
    
    def create_tables(self):
        '''Create required database tables'''
        
        tables = {
            'clients': '''
                CREATE TABLE IF NOT EXISTS clients (
                    id TEXT PRIMARY KEY,
                    client_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    phone TEXT,
                    pan_number TEXT UNIQUE,
                    kyc_completed BOOLEAN DEFAULT FALSE,
                    net_worth REAL,
                    status TEXT DEFAULT 'ACTIVE',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''',
            'portfolios': '''
                CREATE TABLE IF NOT EXISTS portfolios (
                    id TEXT PRIMARY KEY,
                    client_id TEXT,
                    portfolio_name TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    initial_investment REAL NOT NULL,
                    current_value REAL,
                    total_return REAL DEFAULT 0,
                    status TEXT DEFAULT 'ACTIVE',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (client_id) REFERENCES clients (id)
                )
            ''',
            'positions': '''
                CREATE TABLE IF NOT EXISTS positions (
                    id TEXT PRIMARY KEY,
                    portfolio_id TEXT,
                    symbol TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    average_price REAL NOT NULL,
                    current_price REAL,
                    market_value REAL,
                    unrealized_pnl REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (portfolio_id) REFERENCES portfolios (id)
                )
            ''',
            'transactions': '''
                CREATE TABLE IF NOT EXISTS transactions (
                    id TEXT PRIMARY KEY,
                    client_id TEXT,
                    portfolio_id TEXT,
                    transaction_type TEXT,
                    symbol TEXT,
                    quantity REAL,
                    price REAL,
                    amount REAL,
                    status TEXT DEFAULT 'COMPLETED',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (client_id) REFERENCES clients (id),
                    FOREIGN KEY (portfolio_id) REFERENCES portfolios (id)
                )
            ''',
            'system_metrics': '''
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    metric_data TEXT,  -- JSON data
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            '''
        }
        
        cursor = self.connection.cursor()
        for table_name, table_sql in tables.items():
            cursor.execute(table_sql)
        
        self.connection.commit()
    
    def execute_query(self, query: str, params: tuple = None) -> List[sqlite3.Row]:
        '''Execute SELECT query and return results'''
        
        try:
            cursor = self.connection.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            return cursor.fetchall()
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return []
    
    def execute_update(self, query: str, params: tuple = None) -> bool:
        '''Execute INSERT/UPDATE/DELETE query'''
        
        try:
            cursor = self.connection.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            self.connection.commit()
            return True
            
        except Exception as e:
            logger.error(f"Update execution failed: {e}")
            return False
    
    def get_client_count(self) -> int:
        '''Get total number of clients'''
        
        result = self.execute_query("SELECT COUNT(*) as count FROM clients")
        return result[0]['count'] if result else 0
    
    def get_total_aum(self) -> float:
        '''Get total assets under management'''
        
        result = self.execute_query(
            "SELECT SUM(current_value) as total FROM portfolios WHERE status = 'ACTIVE'"
        )
        return result[0]['total'] if result and result[0]['total'] else 0.0
    
    def record_metric(self, metric_name: str, value: float, data: Dict = None):
        '''Record system metric'''
        
        data_json = json.dumps(data) if data else None
        
        self.execute_update(
            "INSERT INTO system_metrics (metric_name, metric_value, metric_data) VALUES (?, ?, ?)",
            (metric_name, value, data_json)
        )
    
    def get_recent_metrics(self, metric_name: str, limit: int = 100) -> List[Dict]:
        '''Get recent metrics'''
        
        results = self.execute_query(
            "SELECT * FROM system_metrics WHERE metric_name = ? ORDER BY recorded_at DESC LIMIT ?",
            (metric_name, limit)
        )
        
        metrics = []
        for row in results:
            metric = {
                'name': row['metric_name'],
                'value': row['metric_value'],
                'timestamp': row['recorded_at']
            }
            
            if row['metric_data']:
                try:
                    metric['data'] = json.loads(row['metric_data'])
                except:
                    metric['data'] = {}
            
            metrics.append(metric)
        
        return metrics
    
    def close(self):
        '''Close database connection'''
        
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")

# Global database manager
db_manager = DatabaseManager()

# Convenience functions
def get_db_connection():
    '''Get database connection'''
    return db_manager.connection

def execute_query(query: str, params: tuple = None):
    '''Execute query using global db manager'''
    return db_manager.execute_query(query, params)

def execute_update(query: str, params: tuple = None):
    '''Execute update using global db manager'''
    return db_manager.execute_update(query, params)

def record_metric(metric_name: str, value: float, data: Dict = None):
    '''Record metric using global db manager'''
    return db_manager.record_metric(metric_name, value, data)
