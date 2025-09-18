import pytest
import asyncio
from unittest.mock import Mock, patch
from orchestrator_fixed import FixedOrchestrator

class TestIntegration:
    """Integration tests for Steps 1 & 2"""
    
    @pytest.mark.asyncio
    async def test_full_system_initialization(self):
        """Test complete system startup"""
        with patch.dict('os.environ', {
            'ALPACA_API_KEY': 'test_key',
            'ALPACA_SECRET_KEY': 'test_secret',
            'MONITORING_ENABLED': 'true'
        }):
            orchestrator = FixedOrchestrator()
            
            # Mock external dependencies
            with patch('data_collection.fixed_alpaca_collector.TradingClient') as mock_client:
                mock_account = Mock()
                mock_account.cash = '100000'
                mock_client.return_value.get_account.return_value = mock_account
                
                # Initialize all components
                await orchestrator.initialize()
                
                # Verify all components are loaded
                assert 'database' in orchestrator.components
                assert 'health' in orchestrator.components
                
                # Verify system is running
                assert orchestrator.running == True
                
                # Shutdown
                await orchestrator.shutdown()
                assert orchestrator.running == False
    
    @pytest.mark.asyncio
    async def test_system_without_alpaca(self):
        """Test system initialization without Alpaca credentials"""
        with patch.dict('os.environ', {
            'ALPACA_API_KEY': '',
            'ALPACA_SECRET_KEY': '',
            'MONITORING_ENABLED': 'true'
        }):
            orchestrator = FixedOrchestrator()
            
            # Should still initialize successfully
            await orchestrator.initialize()
            
            # Core components should be present
            assert 'database' in orchestrator.components
            assert 'health' in orchestrator.components
            
            # Alpaca should not be present
            assert 'alpaca' not in orchestrator.components or orchestrator.components['alpaca'] is None
            
            # System should still be running
            assert orchestrator.running == True
            
            await orchestrator.shutdown()
    
    @pytest.mark.asyncio
    async def test_data_flow_integration(self):
        """Test data flow from collection to storage"""
        with patch.dict('os.environ', {
            'ALPACA_API_KEY': 'test_key',
            'ALPACA_SECRET_KEY': 'test_secret'
        }):
            with patch('data_collection.fixed_alpaca_collector.TradingClient') as mock_client:
                mock_account = Mock()
                mock_account.cash = '100000'
                mock_client.return_value.get_account.return_value = mock_account
                
                orchestrator = FixedOrchestrator()
                await orchestrator.initialize()
                
                if 'alpaca' in orchestrator.components and orchestrator.components['alpaca']:
                    # Mock data handler
                    received_data = []
                    
                    async def data_handler(data_type, data):
                        received_data.append((data_type, data))
                    
                    # Add handler to Alpaca collector
                    orchestrator.components['alpaca'].add_data_handler(data_handler)
                    
                    # Simulate data arrival
                    mock_bar = {'symbol': 'AAPL', 'close': 150.0, 'volume': 1000}
                    await orchestrator.components['alpaca']._handle_bar(mock_bar)
                    
                    # Verify data was received
                    assert len(received_data) > 0
                    assert received_data[0][0] == 'bar'
                    assert received_data[0][1]['symbol'] == 'AAPL'
                
                await orchestrator.shutdown()
    
    @pytest.mark.asyncio
    async def test_health_monitoring_integration(self):
        """Test health monitoring across components"""
        with patch.dict('os.environ', {
            'ALPACA_API_KEY': 'test_key',
            'ALPACA_SECRET_KEY': 'test_secret'
        }):
            with patch('data_collection.fixed_alpaca_collector.TradingClient') as mock_client:
                mock_account = Mock()
                mock_account.cash = '100000'
                mock_client.return_value.get_account.return_value = mock_account
                
                orchestrator = FixedOrchestrator()
                await orchestrator.initialize()
                
                health_check = orchestrator.components['health']
                
                # Get system health
                status = await health_check.check_all()
                
                assert 'overall_status' in status
                assert 'components' in status
                assert 'last_check' in status
                assert 'healthy_count' in status
                assert 'total_count' in status
                
                # Should have at least database and health components
                assert status['total_count'] >= 2
                
                await orchestrator.shutdown()
    
    @pytest.mark.asyncio
    async def test_monitoring_metrics_integration(self):
        """Test monitoring metrics collection"""
        with patch.dict('os.environ', {
            'MONITORING_ENABLED': 'true',
            'PROMETHEUS_PORT': '8001'
        }):
            orchestrator = FixedOrchestrator()
            await orchestrator.initialize()
            
            if 'monitoring' in orchestrator.components:
                monitor = orchestrator.components['monitoring']
                
                # Test metric recording
                monitor.record_trade()
                monitor.record_error('test_component')
                monitor.update_health(0.75)
                
                # Verify metrics are recorded
                assert monitor.trade_counter._value._value >= 1
                assert monitor.system_health._value._value == 0.75
            
            await orchestrator.shutdown()
    
    @pytest.mark.asyncio
    async def test_database_health_integration(self):
        """Test database health monitoring integration"""
        orchestrator = FixedOrchestrator()
        await orchestrator.initialize()
        
        # Test database health check
        db_health = await orchestrator._check_database_health()
        
        assert 'status' in db_health
        assert 'message' in db_health
        
        # Should be either healthy (PostgreSQL) or degraded (SQLite fallback)
        assert db_health['status'] in ['healthy', 'degraded']
        
        await orchestrator.shutdown()
    
    @pytest.mark.asyncio
    async def test_component_failure_handling(self):
        """Test system behavior when components fail"""
        with patch.dict('os.environ', {
            'ALPACA_API_KEY': 'test_key',
            'ALPACA_SECRET_KEY': 'test_secret'
        }):
            # Mock Alpaca to fail
            with patch('data_collection.fixed_alpaca_collector.TradingClient') as mock_client:
                mock_client.return_value.get_account.side_effect = Exception("API Error")
                
                orchestrator = FixedOrchestrator()
                await orchestrator.initialize()
                
                # System should still initialize (graceful degradation)
                assert orchestrator.running == True
                assert 'database' in orchestrator.components
                assert 'health' in orchestrator.components
                
                # Alpaca should either be missing or marked as failed
                if 'alpaca' in orchestrator.components:
                    alpaca_health = await orchestrator._check_alpaca_health()
                    assert alpaca_health['status'] in ['degraded', 'unhealthy']
                
                await orchestrator.shutdown()
    
    @pytest.mark.asyncio
    async def test_configuration_integration(self):
        """Test configuration loading across components"""
        test_config = {
            'DB_HOST': 'test_host',
            'DB_PORT': '5433',
            'REDIS_HOST': 'test_redis',
            'PROMETHEUS_PORT': '8002',
            'MONITORING_ENABLED': 'false'
        }
        
        with patch.dict('os.environ', test_config):
            orchestrator = FixedOrchestrator()
            
            # Check that configuration is loaded correctly
            config = orchestrator.config
            
            assert config['DB_HOST'] == 'test_host'
            assert config['DB_PORT'] == 5433
            assert config['REDIS_HOST'] == 'test_redis'
            assert config['PROMETHEUS_PORT'] == 8002
            assert config['MONITORING_ENABLED'] == False
    
    @pytest.mark.asyncio
    async def test_graceful_shutdown_integration(self):
        """Test graceful shutdown of all components"""
        orchestrator = FixedOrchestrator()
        await orchestrator.initialize()
        
        # Record initial component count
        initial_components = len(orchestrator.components)
        assert initial_components > 0
        
        # Test graceful shutdown
        await orchestrator.shutdown()
        
        # Verify system is stopped
        assert orchestrator.running == False
        
        # Components should still be tracked but closed
        assert len(orchestrator.components) == initial_components
    
    @pytest.mark.asyncio
    async def test_risk_engine_integration(self):
        """Test risk engine integration"""
        orchestrator = FixedOrchestrator()
        await orchestrator.initialize()
        
        # Check if risk engine is available
        if 'risk' in orchestrator.components:
            risk_health = await orchestrator._check_risk_health()
            assert 'status' in risk_health
            assert 'message' in risk_health
        
        await orchestrator.shutdown()
    
    @pytest.mark.asyncio
    async def test_concurrent_health_checks(self):
        """Test concurrent health check operations"""
        orchestrator = FixedOrchestrator()
        await orchestrator.initialize()
        
        health_check = orchestrator.components['health']
        
        # Run multiple concurrent health checks
        tasks = []
        for _ in range(10):
            tasks.append(health_check.check_all())
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should complete successfully
        assert len(results) == 10
        for result in results:
            assert not isinstance(result, Exception)
            assert 'overall_status' in result
        
        await orchestrator.shutdown()
    
    def test_enhanced_components_preserved(self):
        """Test that enhanced components are still available"""
        # Check that enhanced orchestrator exists
        try:
            from orchestrator_enhanced import EnhancedOrchestrator
            assert EnhancedOrchestrator is not None
        except ImportError:
            pytest.fail("Enhanced orchestrator missing")
        
        # Check that production orchestrator exists
        try:
            from orchestrator_production import ProductionOrchestrator
            assert ProductionOrchestrator is not None
        except ImportError:
            pytest.fail("Production orchestrator missing")
        
        # Check that enhanced infrastructure exists
        try:
            from infrastructure.monitoring import get_monitoring_manager
            assert get_monitoring_manager is not None
        except ImportError:
            pytest.fail("Enhanced monitoring missing")
    
    @pytest.mark.asyncio
    async def test_system_status_display(self):
        """Test system status display functionality"""
        orchestrator = FixedOrchestrator()
        await orchestrator.initialize()
        
        # Test that status display works (captures print output)
        import io
        import sys
        from unittest.mock import patch
        
        captured_output = io.StringIO()
        with patch('sys.stdout', captured_output):
            await orchestrator._print_status()
        
        output = captured_output.getvalue()
        
        # Verify key status information is displayed
        assert 'OMNI ALPHA 5.0' in output
        assert 'SYSTEM STATUS' in output
        assert 'Components:' in output
        
        await orchestrator.shutdown()
