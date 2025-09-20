#!/usr/bin/env python3
"""
OMNI ALPHA 5.0 - SPEED BENCHMARK & PERFORMANCE ANALYSIS
======================================================
Comprehensive speed testing and performance benchmarking
"""

import asyncio
import time
import statistics
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json
import sys
import os

# Import our perfect system
from step_1_2_final_perfect import PerfectSystemOrchestrator, PerfectCoreInfrastructure, PerfectDataCollection, config

class SpeedBenchmark:
    """Comprehensive speed and performance benchmark"""
    
    def __init__(self):
        self.results = {}
        self.system_metrics = {}
        
    async def run_comprehensive_benchmark(self):
        """Run comprehensive speed benchmark"""
        print("🏃 OMNI ALPHA 5.0 - COMPREHENSIVE SPEED BENCHMARK")
        print("=" * 70)
        print(f"Started: {datetime.now()}")
        print()
        
        # System initialization speed
        await self._benchmark_initialization_speed()
        
        # Database operation speed
        await self._benchmark_database_speed()
        
        # Data collection speed
        await self._benchmark_data_collection_speed()
        
        # API call speed
        await self._benchmark_api_speed()
        
        # Memory and CPU performance
        await self._benchmark_system_performance()
        
        # Concurrent operation speed
        await self._benchmark_concurrent_performance()
        
        # Generate comprehensive report
        await self._generate_speed_report()
        
        return self.results
    
    async def _benchmark_initialization_speed(self):
        """Benchmark system initialization speed"""
        print("🚀 BENCHMARK 1: SYSTEM INITIALIZATION SPEED")
        print("-" * 50)
        
        times = []
        
        for i in range(3):  # Test 3 times for accuracy
            start_time = time.perf_counter()
            
            orchestrator = PerfectSystemOrchestrator()
            await orchestrator.initialize_perfect_system()
            
            end_time = time.perf_counter()
            init_time = end_time - start_time
            times.append(init_time)
            
            print(f"   Run {i+1}: {init_time:.3f} seconds")
            
            # Cleanup
            await orchestrator._perfect_shutdown()
        
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        
        self.results['initialization'] = {
            'average_seconds': avg_time,
            'min_seconds': min_time,
            'max_seconds': max_time,
            'consistency': (max_time - min_time) / avg_time,
            'grade': 'EXCELLENT' if avg_time < 5 else 'GOOD' if avg_time < 10 else 'FAIR'
        }
        
        print(f"\n📊 INITIALIZATION SPEED RESULTS:")
        print(f"   Average: {avg_time:.3f} seconds")
        print(f"   Best: {min_time:.3f} seconds")
        print(f"   Worst: {max_time:.3f} seconds")
        print(f"   Consistency: {(1 - self.results['initialization']['consistency']):.1%}")
        print(f"   Grade: {self.results['initialization']['grade']}")
        print()
    
    async def _benchmark_database_speed(self):
        """Benchmark database operation speed"""
        print("🗄️ BENCHMARK 2: DATABASE OPERATION SPEED")
        print("-" * 50)
        
        # Initialize system
        infrastructure = PerfectCoreInfrastructure()
        await infrastructure.initialize()
        
        # Test database operations
        operation_times = {}
        
        # Test 1: Simple queries
        query_times = []
        for i in range(100):
            start_time = time.perf_counter()
            
            if infrastructure.db_connection and hasattr(infrastructure.db_connection, 'acquire'):
                # PostgreSQL
                async with infrastructure.db_connection.acquire() as conn:
                    await conn.fetchval('SELECT 1')
            else:
                # SQLite
                cursor = infrastructure.db_connection.execute('SELECT 1')
                cursor.fetchone()
            
            end_time = time.perf_counter()
            query_times.append((end_time - start_time) * 1000)  # Convert to milliseconds
        
        operation_times['simple_query'] = {
            'avg_ms': statistics.mean(query_times),
            'min_ms': min(query_times),
            'max_ms': max(query_times),
            'p95_ms': statistics.quantiles(query_times, n=20)[18],  # 95th percentile
            'p99_ms': statistics.quantiles(query_times, n=100)[98]  # 99th percentile
        }
        
        # Test 2: Insert operations
        insert_times = []
        for i in range(50):
            start_time = time.perf_counter()
            
            if hasattr(infrastructure.db_connection, 'acquire'):
                # PostgreSQL (if available)
                try:
                    async with infrastructure.db_connection.acquire() as conn:
                        await conn.execute(
                            'INSERT INTO trades (symbol, side, quantity, price) VALUES ($1, $2, $3, $4)',
                            f'TEST{i}', 'buy', 100, 150.0
                        )
                except:
                    # Fallback to SQLite
                    infrastructure.db_connection.execute(
                        'INSERT INTO trades (symbol, side, quantity, price) VALUES (?, ?, ?, ?)',
                        (f'TEST{i}', 'buy', 100, 150.0)
                    )
                    infrastructure.db_connection.commit()
            else:
                # SQLite
                infrastructure.db_connection.execute(
                    'INSERT INTO trades (symbol, side, quantity, price) VALUES (?, ?, ?, ?)',
                    (f'TEST{i}', 'buy', 100, 150.0)
                )
                infrastructure.db_connection.commit()
            
            end_time = time.perf_counter()
            insert_times.append((end_time - start_time) * 1000)
        
        operation_times['insert'] = {
            'avg_ms': statistics.mean(insert_times),
            'min_ms': min(insert_times),
            'max_ms': max(insert_times),
            'p95_ms': statistics.quantiles(insert_times, n=20)[18],
            'p99_ms': statistics.quantiles(insert_times, n=100)[98]
        }
        
        self.results['database'] = operation_times
        
        print(f"📊 DATABASE SPEED RESULTS:")
        print(f"   Simple Query Average: {operation_times['simple_query']['avg_ms']:.2f}ms")
        print(f"   Simple Query P95: {operation_times['simple_query']['p95_ms']:.2f}ms")
        print(f"   Insert Average: {operation_times['insert']['avg_ms']:.2f}ms")
        print(f"   Insert P95: {operation_times['insert']['p95_ms']:.2f}ms")
        
        # Grade database performance
        query_avg = operation_times['simple_query']['avg_ms']
        if query_avg < 1:
            db_grade = "EXCELLENT (Sub-millisecond)"
        elif query_avg < 5:
            db_grade = "GOOD (Sub-5ms)"
        elif query_avg < 10:
            db_grade = "FAIR (Sub-10ms)"
        else:
            db_grade = "NEEDS IMPROVEMENT"
        
        print(f"   Database Grade: {db_grade}")
        print()
    
    async def _benchmark_data_collection_speed(self):
        """Benchmark data collection speed"""
        print("📡 BENCHMARK 3: DATA COLLECTION SPEED")
        print("-" * 50)
        
        # Initialize system
        infrastructure = PerfectCoreInfrastructure()
        await infrastructure.initialize()
        
        data_collection = PerfectDataCollection(infrastructure)
        await data_collection.initialize()
        
        # Test data retrieval speed
        symbols = config.scan_symbols[:5]  # Test with 5 symbols
        retrieval_times = []
        
        for symbol in symbols:
            start_time = time.perf_counter()
            
            data = await data_collection.get_perfect_market_data(symbol, days=10)
            
            end_time = time.perf_counter()
            retrieval_time = (end_time - start_time) * 1000  # Convert to ms
            retrieval_times.append(retrieval_time)
            
            print(f"   {symbol}: {retrieval_time:.1f}ms ({len(data) if data is not None else 0} bars)")
        
        # Test quote retrieval speed
        quote_times = []
        for symbol in symbols:
            start_time = time.perf_counter()
            
            quote = data_collection.get_perfect_latest_quote(symbol)
            
            end_time = time.perf_counter()
            quote_time = (end_time - start_time) * 1000
            quote_times.append(quote_time)
        
        self.results['data_collection'] = {
            'historical_data': {
                'avg_ms': statistics.mean(retrieval_times),
                'min_ms': min(retrieval_times),
                'max_ms': max(retrieval_times)
            },
            'quotes': {
                'avg_ms': statistics.mean(quote_times),
                'min_ms': min(quote_times),
                'max_ms': max(quote_times)
            }
        }
        
        print(f"\n📊 DATA COLLECTION SPEED RESULTS:")
        print(f"   Historical Data Average: {statistics.mean(retrieval_times):.1f}ms")
        print(f"   Quote Retrieval Average: {statistics.mean(quote_times):.1f}ms")
        
        await data_collection.close()
        print()
    
    async def _benchmark_api_speed(self):
        """Benchmark API call speed"""
        print("🔗 BENCHMARK 4: API CALL SPEED")
        print("-" * 50)
        
        if not config.alpaca_secret:
            print("   Demo mode - API speed simulation")
            # Simulate API times
            self.results['api_speed'] = {
                'account_check_ms': 50.0,
                'market_data_ms': 100.0,
                'order_submission_ms': 25.0,
                'grade': 'SIMULATED'
            }
            print("   Account Check: 50.0ms (simulated)")
            print("   Market Data: 100.0ms (simulated)")
            print("   Order Submission: 25.0ms (simulated)")
            print()
            return
        
        try:
            import alpaca_trade_api as tradeapi
            api = tradeapi.REST(config.alpaca_key, config.alpaca_secret, 'https://paper-api.alpaca.markets')
            
            # Test account check speed
            account_times = []
            for i in range(10):
                start_time = time.perf_counter()
                account = api.get_account()
                end_time = time.perf_counter()
                account_times.append((end_time - start_time) * 1000)
            
            # Test market data speed
            data_times = []
            for i in range(5):
                start_time = time.perf_counter()
                bars = api.get_bars('AAPL', '1Day', start='2025-09-01', limit=10)
                end_time = time.perf_counter()
                data_times.append((end_time - start_time) * 1000)
            
            self.results['api_speed'] = {
                'account_check_ms': statistics.mean(account_times),
                'market_data_ms': statistics.mean(data_times),
                'account_check_p95': statistics.quantiles(account_times, n=20)[18],
                'market_data_p95': statistics.quantiles(data_times, n=5)[4]
            }
            
            print(f"   Account Check Average: {statistics.mean(account_times):.1f}ms")
            print(f"   Market Data Average: {statistics.mean(data_times):.1f}ms")
            print(f"   Account Check P95: {self.results['api_speed']['account_check_p95']:.1f}ms")
            print(f"   Market Data P95: {self.results['api_speed']['market_data_p95']:.1f}ms")
            
        except Exception as e:
            print(f"   API speed test error: {e}")
            self.results['api_speed'] = {'error': str(e)}
        
        print()
    
    async def _benchmark_system_performance(self):
        """Benchmark system-level performance"""
        print("💻 BENCHMARK 5: SYSTEM PERFORMANCE")
        print("-" * 50)
        
        # Get initial system state
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        initial_cpu = process.cpu_percent()
        
        # Initialize system and run for measurement
        orchestrator = PerfectSystemOrchestrator()
        await orchestrator.initialize_perfect_system()
        
        # Measure under load
        memory_samples = []
        cpu_samples = []
        
        for i in range(30):  # 30 seconds of monitoring
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent()
            
            memory_samples.append(memory_mb)
            cpu_samples.append(cpu_percent)
            
            await asyncio.sleep(1)
        
        # Calculate performance metrics
        self.results['system_performance'] = {
            'memory_usage': {
                'initial_mb': initial_memory,
                'avg_mb': statistics.mean(memory_samples),
                'max_mb': max(memory_samples),
                'growth_mb': max(memory_samples) - initial_memory
            },
            'cpu_usage': {
                'avg_percent': statistics.mean(cpu_samples),
                'max_percent': max(cpu_samples),
                'efficiency': 'HIGH' if statistics.mean(cpu_samples) < 20 else 'MEDIUM' if statistics.mean(cpu_samples) < 50 else 'LOW'
            }
        }
        
        print(f"📊 SYSTEM PERFORMANCE RESULTS:")
        print(f"   Memory Usage: {statistics.mean(memory_samples):.1f}MB average, {max(memory_samples):.1f}MB peak")
        print(f"   CPU Usage: {statistics.mean(cpu_samples):.1f}% average, {max(cpu_samples):.1f}% peak")
        print(f"   Memory Growth: {max(memory_samples) - initial_memory:.1f}MB")
        print(f"   Efficiency: {self.results['system_performance']['cpu_usage']['efficiency']}")
        
        await orchestrator._perfect_shutdown()
        print()
    
    async def _benchmark_concurrent_performance(self):
        """Benchmark concurrent operation performance"""
        print("🔄 BENCHMARK 6: CONCURRENT OPERATION SPEED")
        print("-" * 50)
        
        # Initialize system
        infrastructure = PerfectCoreInfrastructure()
        await infrastructure.initialize()
        
        data_collection = PerfectDataCollection(infrastructure)
        await data_collection.initialize()
        
        # Test concurrent data retrieval
        symbols = config.scan_symbols[:5]
        
        # Sequential timing
        start_time = time.perf_counter()
        for symbol in symbols:
            await data_collection.get_perfect_market_data(symbol, days=5)
        sequential_time = time.perf_counter() - start_time
        
        # Concurrent timing
        start_time = time.perf_counter()
        tasks = [data_collection.get_perfect_market_data(symbol, days=5) for symbol in symbols]
        await asyncio.gather(*tasks)
        concurrent_time = time.perf_counter() - start_time
        
        # Calculate speedup
        speedup = sequential_time / concurrent_time if concurrent_time > 0 else 1
        
        self.results['concurrent_performance'] = {
            'sequential_seconds': sequential_time,
            'concurrent_seconds': concurrent_time,
            'speedup_factor': speedup,
            'efficiency': speedup / len(symbols)
        }
        
        print(f"📊 CONCURRENT PERFORMANCE RESULTS:")
        print(f"   Sequential: {sequential_time:.3f} seconds")
        print(f"   Concurrent: {concurrent_time:.3f} seconds")
        print(f"   Speedup: {speedup:.1f}x faster")
        print(f"   Efficiency: {(speedup / len(symbols)):.1%}")
        
        await data_collection.close()
        print()
    
    async def _generate_speed_report(self):
        """Generate comprehensive speed report"""
        print("📊 COMPREHENSIVE SPEED ANALYSIS")
        print("=" * 70)
        
        # Calculate overall speed score
        scores = []
        
        # Initialization score
        init_time = self.results['initialization']['average_seconds']
        init_score = 10 if init_time < 3 else 9 if init_time < 5 else 8 if init_time < 10 else 6
        scores.append(init_score)
        
        # Database score
        if 'database' in self.results:
            db_time = self.results['database']['simple_query']['avg_ms']
            db_score = 10 if db_time < 1 else 9 if db_time < 5 else 8 if db_time < 10 else 6
            scores.append(db_score)
        
        # System performance score
        memory_mb = self.results['system_performance']['memory_usage']['avg_mb']
        cpu_pct = self.results['system_performance']['cpu_usage']['avg_percent']
        
        memory_score = 10 if memory_mb < 100 else 9 if memory_mb < 500 else 8 if memory_mb < 1000 else 6
        cpu_score = 10 if cpu_pct < 10 else 9 if cpu_pct < 20 else 8 if cpu_pct < 50 else 6
        
        scores.extend([memory_score, cpu_score])
        
        # Concurrent performance score
        speedup = self.results['concurrent_performance']['speedup_factor']
        concurrent_score = 10 if speedup > 4 else 9 if speedup > 3 else 8 if speedup > 2 else 6
        scores.append(concurrent_score)
        
        overall_speed_score = statistics.mean(scores)
        
        print(f"🏆 OVERALL SPEED SCORE: {overall_speed_score:.1f}/10")
        
        if overall_speed_score >= 9.5:
            speed_grade = "WORLD-CLASS (Tier 1)"
        elif overall_speed_score >= 9.0:
            speed_grade = "EXCELLENT (Tier 2)"
        elif overall_speed_score >= 8.0:
            speed_grade = "GOOD (Tier 3)"
        else:
            speed_grade = "NEEDS OPTIMIZATION"
        
        print(f"🎯 SPEED GRADE: {speed_grade}")
        
        # Detailed breakdown
        print(f"\n📋 DETAILED SPEED BREAKDOWN:")
        print(f"   Initialization: {init_score}/10 ({init_time:.3f}s)")
        if 'database' in self.results:
            print(f"   Database Queries: {db_score}/10 ({self.results['database']['simple_query']['avg_ms']:.2f}ms)")
        print(f"   Memory Efficiency: {memory_score}/10 ({memory_mb:.1f}MB)")
        print(f"   CPU Efficiency: {cpu_score}/10 ({cpu_pct:.1f}%)")
        print(f"   Concurrent Processing: {concurrent_score}/10 ({speedup:.1f}x speedup)")
        
        self.results['overall'] = {
            'speed_score': overall_speed_score,
            'speed_grade': speed_grade,
            'individual_scores': {
                'initialization': init_score,
                'database': db_score if 'database' in self.results else 0,
                'memory': memory_score,
                'cpu': cpu_score,
                'concurrent': concurrent_score
            }
        }
        
        print("=" * 70)

async def main():
    """Main benchmark execution"""
    benchmark = SpeedBenchmark()
    results = await benchmark.run_comprehensive_benchmark()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'speed_benchmark_report_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"📄 Detailed report saved: speed_benchmark_report_{timestamp}.json")
    
    return results['overall']['speed_score'] >= 8.0

if __name__ == "__main__":
    success = asyncio.run(main())
    print(f"\n🏆 SPEED BENCHMARK {'PASSED' if success else 'NEEDS IMPROVEMENT'}")
    sys.exit(0 if success else 1)

