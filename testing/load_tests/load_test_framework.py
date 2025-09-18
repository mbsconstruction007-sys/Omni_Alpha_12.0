"""
OMNI ALPHA 5.0 - LOAD TESTING FRAMEWORK
=======================================
Production-ready load testing with comprehensive metrics and reporting
"""

import asyncio
import aiohttp
import time
import statistics
import json
import csv
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import logging
import os

try:
    from locust import HttpUser, task, between, events
    from locust.env import Environment
    from locust.stats import stats_printer, stats_history
    from locust.log import setup_logging
    LOCUST_AVAILABLE = True
except ImportError:
    LOCUST_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from config.settings import get_settings
from config.logging_config import get_logger

logger = get_logger(__name__, 'load_testing')

@dataclass
class LoadTestConfig:
    """Load test configuration"""
    target_url: str
    num_users: int
    spawn_rate: float
    duration_seconds: int
    requests_per_user: Optional[int] = None
    request_timeout: float = 30.0
    think_time_min: float = 0.1
    think_time_max: float = 1.0
    ramp_up_duration: int = 60
    custom_headers: Dict[str, str] = field(default_factory=dict)
    test_scenarios: List[str] = field(default_factory=list)

@dataclass
class LoadTestResult:
    """Comprehensive load test results"""
    test_name: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    requests_per_second: float
    error_rate: float
    
    # Response time statistics
    avg_response_time: float
    median_response_time: float
    min_response_time: float
    max_response_time: float
    p90_response_time: float
    p95_response_time: float
    p99_response_time: float
    
    # Throughput statistics
    bytes_sent: int
    bytes_received: int
    throughput_mbps: float
    
    # Error breakdown
    errors_by_type: Dict[str, int] = field(default_factory=dict)
    
    # Performance over time
    response_times: List[float] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'test_name': self.test_name,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'duration_seconds': self.duration_seconds,
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'requests_per_second': self.requests_per_second,
            'error_rate': self.error_rate,
            'avg_response_time': self.avg_response_time,
            'median_response_time': self.median_response_time,
            'min_response_time': self.min_response_time,
            'max_response_time': self.max_response_time,
            'p90_response_time': self.p90_response_time,
            'p95_response_time': self.p95_response_time,
            'p99_response_time': self.p99_response_time,
            'bytes_sent': self.bytes_sent,
            'bytes_received': self.bytes_received,
            'throughput_mbps': self.throughput_mbps,
            'errors_by_type': self.errors_by_type
        }

class AsyncLoadTester:
    """Asynchronous load testing framework"""
    
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.logger = get_logger(__name__, 'async_load_tester')
        self.results = []
        self.errors = []
        self.start_time = None
        self.end_time = None
    
    async def run_load_test(self, test_name: str = "async_load_test") -> LoadTestResult:
        """Run asynchronous load test"""
        self.logger.info(f"Starting load test: {test_name}")
        self.logger.info(f"Target: {self.config.target_url}")
        self.logger.info(f"Users: {self.config.num_users}, Duration: {self.config.duration_seconds}s")
        
        self.start_time = datetime.now()
        
        # Create semaphore to limit concurrent connections
        semaphore = asyncio.Semaphore(self.config.num_users)
        
        # Create user sessions
        tasks = []
        for user_id in range(self.config.num_users):
            task = asyncio.create_task(
                self._user_session(user_id, semaphore)
            )
            tasks.append(task)
            
            # Stagger user startup
            if user_id > 0 and user_id % self.config.spawn_rate == 0:
                await asyncio.sleep(1)
        
        # Wait for all users to complete or timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.config.duration_seconds + 60
            )
        except asyncio.TimeoutError:
            self.logger.warning("Load test timed out, cancelling remaining tasks")
            for task in tasks:
                if not task.done():
                    task.cancel()
        
        self.end_time = datetime.now()
        
        # Calculate results
        result = self._calculate_results(test_name)
        
        self.logger.info(f"Load test completed: {result.total_requests} requests, "
                        f"{result.error_rate:.1%} error rate, "
                        f"{result.avg_response_time:.0f}ms avg response time")
        
        return result
    
    async def _user_session(self, user_id: int, semaphore: asyncio.Semaphore):
        """Simulate single user session"""
        session_results = []
        session_start = time.time()
        
        # Create HTTP session with custom configuration
        timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
        connector = aiohttp.TCPConnector(
            limit=10,
            limit_per_host=5,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        async with aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers=self.config.custom_headers
        ) as session:
            
            request_count = 0
            while (time.time() - session_start) < self.config.duration_seconds:
                async with semaphore:
                    try:
                        # Make request
                        start_time = time.time()
                        
                        async with session.get(self.config.target_url) as response:
                            content = await response.read()
                            
                        end_time = time.time()
                        response_time = end_time - start_time
                        
                        # Record result
                        result = {
                            'user_id': user_id,
                            'request_id': request_count,
                            'timestamp': datetime.fromtimestamp(start_time),
                            'response_time': response_time,
                            'status_code': response.status,
                            'success': response.status < 400,
                            'bytes_sent': len(str(self.config.custom_headers)),
                            'bytes_received': len(content),
                            'error': None
                        }
                        
                        session_results.append(result)
                        
                    except Exception as e:
                        # Record error
                        error_result = {
                            'user_id': user_id,
                            'request_id': request_count,
                            'timestamp': datetime.now(),
                            'response_time': self.config.request_timeout,
                            'status_code': 0,
                            'success': False,
                            'bytes_sent': 0,
                            'bytes_received': 0,
                            'error': str(e)
                        }
                        
                        session_results.append(error_result)
                        self.errors.append(error_result)
                
                request_count += 1
                
                # Check request limit
                if (self.config.requests_per_user and 
                    request_count >= self.config.requests_per_user):
                    break
                
                # Think time
                think_time = np.random.uniform(
                    self.config.think_time_min,
                    self.config.think_time_max
                ) if NUMPY_AVAILABLE else self.config.think_time_min
                
                await asyncio.sleep(think_time)
        
        # Add results to global list
        self.results.extend(session_results)
    
    def _calculate_results(self, test_name: str) -> LoadTestResult:
        """Calculate comprehensive test results"""
        if not self.results:
            raise ValueError("No test results available")
        
        # Basic metrics
        total_requests = len(self.results)
        successful_requests = sum(1 for r in self.results if r['success'])
        failed_requests = total_requests - successful_requests
        error_rate = failed_requests / total_requests if total_requests > 0 else 0
        
        # Duration
        duration = (self.end_time - self.start_time).total_seconds()
        requests_per_second = total_requests / duration if duration > 0 else 0
        
        # Response time statistics
        response_times = [r['response_time'] for r in self.results]
        successful_times = [r['response_time'] for r in self.results if r['success']]
        
        if successful_times:
            avg_response_time = statistics.mean(successful_times)
            median_response_time = statistics.median(successful_times)
            min_response_time = min(successful_times)
            max_response_time = max(successful_times)
            
            # Percentiles
            if NUMPY_AVAILABLE:
                p90_response_time = float(np.percentile(successful_times, 90))
                p95_response_time = float(np.percentile(successful_times, 95))
                p99_response_time = float(np.percentile(successful_times, 99))
            else:
                sorted_times = sorted(successful_times)
                p90_idx = int(len(sorted_times) * 0.9)
                p95_idx = int(len(sorted_times) * 0.95)
                p99_idx = int(len(sorted_times) * 0.99)
                
                p90_response_time = sorted_times[p90_idx] if p90_idx < len(sorted_times) else max_response_time
                p95_response_time = sorted_times[p95_idx] if p95_idx < len(sorted_times) else max_response_time
                p99_response_time = sorted_times[p99_idx] if p99_idx < len(sorted_times) else max_response_time
        else:
            avg_response_time = median_response_time = min_response_time = max_response_time = 0
            p90_response_time = p95_response_time = p99_response_time = 0
        
        # Throughput
        bytes_sent = sum(r['bytes_sent'] for r in self.results)
        bytes_received = sum(r['bytes_received'] for r in self.results)
        throughput_mbps = (bytes_received / (1024 * 1024)) / duration if duration > 0 else 0
        
        # Error breakdown
        errors_by_type = {}
        for result in self.results:
            if not result['success'] and result['error']:
                error_type = type(Exception(result['error'])).__name__
                errors_by_type[error_type] = errors_by_type.get(error_type, 0) + 1
        
        # Performance over time
        timestamps = [r['timestamp'] for r in self.results]
        
        return LoadTestResult(
            test_name=test_name,
            start_time=self.start_time,
            end_time=self.end_time,
            duration_seconds=duration,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            requests_per_second=requests_per_second,
            error_rate=error_rate,
            avg_response_time=avg_response_time * 1000,  # Convert to ms
            median_response_time=median_response_time * 1000,
            min_response_time=min_response_time * 1000,
            max_response_time=max_response_time * 1000,
            p90_response_time=p90_response_time * 1000,
            p95_response_time=p95_response_time * 1000,
            p99_response_time=p99_response_time * 1000,
            bytes_sent=bytes_sent,
            bytes_received=bytes_received,
            throughput_mbps=throughput_mbps,
            errors_by_type=errors_by_type,
            response_times=[rt * 1000 for rt in response_times],
            timestamps=timestamps
        )

class TradingSystemUser(HttpUser):
    """Locust user for trading system load testing"""
    
    wait_time = between(0.1, 2.0)
    
    def on_start(self):
        """Called when user starts"""
        self.client.headers.update({
            'User-Agent': 'OmniAlpha-LoadTest/5.0',
            'Accept': 'application/json'
        })
    
    @task(5)
    def get_market_data(self):
        """Test market data endpoint"""
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        symbol = self.environment.parsed_options.symbol if hasattr(self.environment, 'parsed_options') else 'AAPL'
        
        with self.client.get(
            f'/api/v1/market/data/{symbol}',
            catch_response=True,
            name="GET /market/data"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(3)
    def get_portfolio(self):
        """Test portfolio endpoint"""
        with self.client.get(
            '/api/v1/portfolio',
            catch_response=True,
            name="GET /portfolio"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(2)
    def place_order(self):
        """Test order placement"""
        order_data = {
            'symbol': 'AAPL',
            'quantity': 10,
            'side': 'buy',
            'type': 'limit',
            'limit_price': 150.00
        }
        
        with self.client.post(
            '/api/v1/orders',
            json=order_data,
            catch_response=True,
            name="POST /orders"
        ) as response:
            if response.status_code in [200, 201]:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    def get_risk_metrics(self):
        """Test risk metrics endpoint"""
        with self.client.get(
            '/api/v1/risk/metrics',
            catch_response=True,
            name="GET /risk/metrics"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")

class LoadTestRunner:
    """Comprehensive load test runner with reporting"""
    
    def __init__(self, settings=None):
        if settings is None:
            settings = get_settings()
        
        self.settings = settings
        self.logger = get_logger(__name__, 'load_test_runner')
        self.results_dir = os.getenv('LOAD_TEST_RESULTS_DIR', 'load_test_results')
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
    
    async def run_async_test(self, config: LoadTestConfig, test_name: str) -> LoadTestResult:
        """Run async load test"""
        tester = AsyncLoadTester(config)
        result = await tester.run_load_test(test_name)
        
        # Save results
        await self._save_results(result)
        
        return result
    
    def run_locust_test(self, config: LoadTestConfig, test_name: str) -> Optional[LoadTestResult]:
        """Run Locust-based load test"""
        if not LOCUST_AVAILABLE:
            self.logger.error("Locust not available for load testing")
            return None
        
        try:
            # Setup Locust environment
            env = Environment(user_classes=[TradingSystemUser])
            env.create_local_runner()
            
            # Configure parsed options
            env.parsed_options = type('Options', (), {
                'host': config.target_url,
                'num_users': config.num_users,
                'spawn_rate': config.spawn_rate,
                'run_time': f'{config.duration_seconds}s'
            })()
            
            # Start test
            env.runner.start(config.num_users, config.spawn_rate)
            
            # Wait for test completion
            import threading
            stop_event = threading.Event()
            
            def stop_test():
                time.sleep(config.duration_seconds)
                env.runner.stop()
                stop_event.set()
            
            stop_thread = threading.Thread(target=stop_test)
            stop_thread.start()
            
            # Wait for completion
            stop_event.wait()
            
            # Get results
            stats = env.runner.stats
            
            # Convert to our result format
            result = self._convert_locust_results(stats, test_name, config)
            
            # Save results
            asyncio.run(self._save_results(result))
            
            return result
            
        except Exception as e:
            self.logger.error(f"Locust test failed: {e}")
            return None
    
    def _convert_locust_results(self, stats, test_name: str, config: LoadTestConfig) -> LoadTestResult:
        """Convert Locust stats to LoadTestResult"""
        total_stats = stats.total
        
        return LoadTestResult(
            test_name=test_name,
            start_time=datetime.now() - timedelta(seconds=config.duration_seconds),
            end_time=datetime.now(),
            duration_seconds=config.duration_seconds,
            total_requests=total_stats.num_requests,
            successful_requests=total_stats.num_requests - total_stats.num_failures,
            failed_requests=total_stats.num_failures,
            requests_per_second=total_stats.current_rps,
            error_rate=total_stats.num_failures / total_stats.num_requests if total_stats.num_requests > 0 else 0,
            avg_response_time=total_stats.avg_response_time,
            median_response_time=total_stats.median_response_time,
            min_response_time=total_stats.min_response_time,
            max_response_time=total_stats.max_response_time,
            p90_response_time=total_stats.get_response_time_percentile(0.9),
            p95_response_time=total_stats.get_response_time_percentile(0.95),
            p99_response_time=total_stats.get_response_time_percentile(0.99),
            bytes_sent=0,  # Not available in Locust stats
            bytes_received=0,
            throughput_mbps=0,
            errors_by_type={},
            response_times=[],
            timestamps=[]
        )
    
    async def _save_results(self, result: LoadTestResult):
        """Save test results to files"""
        timestamp = result.start_time.strftime('%Y%m%d_%H%M%S')
        base_filename = f"{result.test_name}_{timestamp}"
        
        # Save JSON results
        json_file = os.path.join(self.results_dir, f"{base_filename}.json")
        with open(json_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        # Save CSV summary
        csv_file = os.path.join(self.results_dir, f"{base_filename}_summary.csv")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Test Name', result.test_name])
            writer.writerow(['Duration (s)', result.duration_seconds])
            writer.writerow(['Total Requests', result.total_requests])
            writer.writerow(['Successful Requests', result.successful_requests])
            writer.writerow(['Failed Requests', result.failed_requests])
            writer.writerow(['Requests/Second', f"{result.requests_per_second:.2f}"])
            writer.writerow(['Error Rate (%)', f"{result.error_rate * 100:.2f}"])
            writer.writerow(['Avg Response Time (ms)', f"{result.avg_response_time:.2f}"])
            writer.writerow(['P95 Response Time (ms)', f"{result.p95_response_time:.2f}"])
            writer.writerow(['P99 Response Time (ms)', f"{result.p99_response_time:.2f}"])
        
        self.logger.info(f"Load test results saved to {json_file} and {csv_file}")
    
    async def run_comprehensive_test_suite(self, base_url: str) -> List[LoadTestResult]:
        """Run comprehensive test suite with multiple scenarios"""
        results = []
        
        test_scenarios = [
            {
                'name': 'light_load',
                'users': 10,
                'duration': 60,
                'spawn_rate': 2
            },
            {
                'name': 'normal_load',
                'users': 50,
                'duration': 300,
                'spawn_rate': 5
            },
            {
                'name': 'peak_load',
                'users': 200,
                'duration': 600,
                'spawn_rate': 10
            },
            {
                'name': 'stress_test',
                'users': 500,
                'duration': 300,
                'spawn_rate': 20
            }
        ]
        
        for scenario in test_scenarios:
            self.logger.info(f"Running test scenario: {scenario['name']}")
            
            config = LoadTestConfig(
                target_url=base_url,
                num_users=scenario['users'],
                spawn_rate=scenario['spawn_rate'],
                duration_seconds=scenario['duration']
            )
            
            result = await self.run_async_test(config, scenario['name'])
            results.append(result)
            
            # Wait between tests
            await asyncio.sleep(30)
        
        # Generate comparison report
        await self._generate_comparison_report(results)
        
        return results
    
    async def _generate_comparison_report(self, results: List[LoadTestResult]):
        """Generate comparison report for multiple test results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = os.path.join(self.results_dir, f"comparison_report_{timestamp}.html")
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Load Test Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
                th {{ background-color: #f2f2f2; }}
                .test-name {{ text-align: left; }}
            </style>
        </head>
        <body>
            <h1>Load Test Comparison Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <table>
                <tr>
                    <th class="test-name">Test Name</th>
                    <th>Users</th>
                    <th>Duration (s)</th>
                    <th>Total Requests</th>
                    <th>RPS</th>
                    <th>Error Rate (%)</th>
                    <th>Avg Response (ms)</th>
                    <th>P95 Response (ms)</th>
                    <th>P99 Response (ms)</th>
                </tr>
        """
        
        for result in results:
            html_content += f"""
                <tr>
                    <td class="test-name">{result.test_name}</td>
                    <td>{result.total_requests // result.duration_seconds:.0f}</td>
                    <td>{result.duration_seconds:.0f}</td>
                    <td>{result.total_requests}</td>
                    <td>{result.requests_per_second:.1f}</td>
                    <td>{result.error_rate * 100:.2f}</td>
                    <td>{result.avg_response_time:.0f}</td>
                    <td>{result.p95_response_time:.0f}</td>
                    <td>{result.p99_response_time:.0f}</td>
                </tr>
            """
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"Comparison report generated: {report_file}")

# ===================== GLOBAL INSTANCE =====================

def get_load_test_runner() -> LoadTestRunner:
    """Get load test runner instance"""
    return LoadTestRunner()
