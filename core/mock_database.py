
"""
Mock database for testing when real database is not available
"""
import json
from datetime import datetime
from typing import Dict, List, Any

class MockDatabase:
    """In-memory database for testing"""
    
    def __init__(self):
        self.tables = {}
        self.connected = True
    
    def create_table(self, table_name: str, schema: Dict):
        """Create a mock table"""
        self.tables[table_name] = {
            'schema': schema,
            'data': []
        }
    
    def insert(self, table_name: str, data: Dict):
        """Insert data into mock table"""
        if table_name not in self.tables:
            self.tables[table_name] = {'data': []}
        
        data['id'] = len(self.tables[table_name]['data']) + 1
        data['created_at'] = datetime.now().isoformat()
        self.tables[table_name]['data'].append(data)
        
        return data['id']
    
    def select(self, table_name: str, conditions: Dict = None) -> List[Dict]:
        """Select data from mock table"""
        if table_name not in self.tables:
            return []
        
        data = self.tables[table_name]['data']
        
        if not conditions:
            return data
        
        # Simple filtering
        filtered_data = []
        for record in data:
            match = True
            for key, value in conditions.items():
                if record.get(key) != value:
                    match = False
                    break
            if match:
                filtered_data.append(record)
        
        return filtered_data
    
    def update(self, table_name: str, record_id: int, data: Dict):
        """Update record in mock table"""
        if table_name not in self.tables:
            return False
        
        for record in self.tables[table_name]['data']:
            if record.get('id') == record_id:
                record.update(data)
                record['updated_at'] = datetime.now().isoformat()
                return True
        
        return False
    
    def delete(self, table_name: str, record_id: int):
        """Delete record from mock table"""
        if table_name not in self.tables:
            return False
        
        original_length = len(self.tables[table_name]['data'])
        self.tables[table_name]['data'] = [
            record for record in self.tables[table_name]['data']
            if record.get('id') != record_id
        ]
        
        return len(self.tables[table_name]['data']) < original_length
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        stats = {
            'tables': len(self.tables),
            'total_records': sum(len(table['data']) for table in self.tables.values()),
            'connected': self.connected
        }
        
        for table_name, table in self.tables.items():
            stats[f'{table_name}_count'] = len(table['data'])
        
        return stats

# Global mock database instance
mock_db = MockDatabase()
