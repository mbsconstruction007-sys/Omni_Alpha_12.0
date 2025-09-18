"""
OMNI ALPHA 5.0 - MESSAGE QUEUE SYSTEM
=====================================
Production-ready message queue with Kafka for high-throughput, fault-tolerant messaging
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from collections import defaultdict

try:
    from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
    from aiokafka.errors import KafkaError, KafkaConnectionError, KafkaTimeoutError
    from kafka.errors import TopicAlreadyExistsError
    from kafka.admin import KafkaAdminClient, ConfigResource, ConfigResourceType
    from kafka import KafkaClient
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

try:
    from prometheus_client import Counter, Histogram, Gauge
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from config.settings import get_settings
from config.logging_config import get_logger

logger = get_logger(__name__, 'message_queue')

# Metrics (if available)
if PROMETHEUS_AVAILABLE:
    messages_published = Counter('messages_published_total', 'Total messages published', ['topic', 'status'])
    messages_consumed = Counter('messages_consumed_total', 'Total messages consumed', ['topic', 'status'])
    message_processing_time = Histogram('message_processing_duration_seconds', 'Message processing time', ['topic', 'handler'])
    message_errors = Counter('message_errors_total', 'Total message errors', ['topic', 'error_type'])
    queue_lag = Gauge('message_queue_lag', 'Consumer lag per topic partition', ['topic', 'partition'])
    active_consumers = Gauge('active_consumers_total', 'Number of active consumers', ['topic'])

class MessagePriority(Enum):
    """Message priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

class MessageType(Enum):
    """Standard message types"""
    TRADE_SIGNAL = "trade_signal"
    MARKET_DATA = "market_data"
    ORDER_UPDATE = "order_update"
    RISK_ALERT = "risk_alert"
    SYSTEM_EVENT = "system_event"
    HEALTH_CHECK = "health_check"

@dataclass
class Message:
    """Standard message format with full metadata"""
    id: str
    timestamp: str
    type: str
    source: str
    data: Dict[str, Any]
    priority: MessagePriority = MessagePriority.NORMAL
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    ttl_seconds: Optional[int] = None
    retry_count: int = 0
    max_retries: int = 3
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()
        if not self.correlation_id:
            self.correlation_id = str(uuid.uuid4())
    
    def to_json(self) -> str:
        """Serialize message to JSON"""
        data = asdict(self)
        data['priority'] = self.priority.value
        return json.dumps(data)
    
    @classmethod
    def from_json(cls, json_str: str):
        """Deserialize message from JSON"""
        data = json.loads(json_str)
        if 'priority' in data:
            data['priority'] = MessagePriority(data['priority'])
        return cls(**data)
    
    def is_expired(self) -> bool:
        """Check if message has expired"""
        if not self.ttl_seconds:
            return False
        
        created_time = datetime.fromisoformat(self.timestamp)
        return (datetime.utcnow() - created_time).total_seconds() > self.ttl_seconds
    
    def should_retry(self) -> bool:
        """Check if message should be retried"""
        return self.retry_count < self.max_retries

class MessageQueueManager:
    """Production message queue with Kafka backend"""
    
    def __init__(self, settings=None):
        if settings is None:
            settings = get_settings()
        
        self.settings = settings
        self.logger = get_logger(__name__, 'message_queue')
        
        # Kafka configuration
        self.kafka_brokers = os.getenv('KAFKA_BROKERS', 'localhost:9092').split(',')
        self.client_id = os.getenv('KAFKA_CLIENT_ID', f'omni-alpha-{os.getpid()}')
        
        # Producer configuration
        self.producer_config = {
            'bootstrap_servers': self.kafka_brokers,
            'client_id': f'{self.client_id}-producer',
            'value_serializer': lambda v: v.encode('utf-8') if isinstance(v, str) else json.dumps(v).encode('utf-8'),
            'key_serializer': lambda k: k.encode('utf-8') if k else None,
            'compression_type': os.getenv('KAFKA_COMPRESSION', 'gzip'),
            'max_batch_size': int(os.getenv('KAFKA_BATCH_SIZE', '16384')),
            'linger_ms': int(os.getenv('KAFKA_LINGER_MS', '10')),
            'acks': os.getenv('KAFKA_ACKS', 'all'),
            'enable_idempotence': True,
            'retries': int(os.getenv('KAFKA_RETRIES', '5')),
            'max_in_flight_requests_per_connection': 5,
            'request_timeout_ms': int(os.getenv('KAFKA_REQUEST_TIMEOUT', '30000')),
            'delivery_timeout_ms': int(os.getenv('KAFKA_DELIVERY_TIMEOUT', '120000'))
        }
        
        # Consumer configuration
        self.consumer_config = {
            'bootstrap_servers': self.kafka_brokers,
            'client_id': f'{self.client_id}-consumer',
            'value_deserializer': lambda m: m.decode('utf-8') if m else None,
            'key_deserializer': lambda k: k.decode('utf-8') if k else None,
            'auto_offset_reset': os.getenv('KAFKA_AUTO_OFFSET_RESET', 'earliest'),
            'enable_auto_commit': False,
            'max_poll_records': int(os.getenv('KAFKA_MAX_POLL_RECORDS', '100')),
            'session_timeout_ms': int(os.getenv('KAFKA_SESSION_TIMEOUT', '30000')),
            'heartbeat_interval_ms': int(os.getenv('KAFKA_HEARTBEAT_INTERVAL', '10000')),
            'fetch_min_bytes': int(os.getenv('KAFKA_FETCH_MIN_BYTES', '1')),
            'fetch_max_wait_ms': int(os.getenv('KAFKA_FETCH_MAX_WAIT', '500'))
        }
        
        # State management
        self.producer: Optional[Any] = None
        self.consumers: Dict[str, Any] = {}
        self.handlers: Dict[str, List[Callable]] = defaultdict(list)
        self.consumer_tasks: List[asyncio.Task] = []
        self.is_running = False
        
        # Dead letter queue
        self.dlq_suffix = os.getenv('KAFKA_DLQ_SUFFIX', '.dlq')
        
        # Monitoring
        self.message_stats = {
            'published': defaultdict(int),
            'consumed': defaultdict(int),
            'errors': defaultdict(int)
        }
    
    async def initialize(self) -> bool:
        """Initialize Kafka producer and admin client"""
        if not KAFKA_AVAILABLE:
            self.logger.error("Kafka client not available - message queue disabled")
            return False
        
        try:
            # Initialize producer
            self.producer = AIOKafkaProducer(**self.producer_config)
            await self.producer.start()
            
            # Create admin client for topic management
            self.admin_client = KafkaAdminClient(
                bootstrap_servers=self.kafka_brokers,
                client_id=f'{self.client_id}-admin'
            )
            
            self.is_running = True
            self.logger.info(f"Message queue initialized with brokers: {self.kafka_brokers}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize message queue: {e}")
            return False
    
    async def create_topic(self, topic: str, partitions: int = 3, replication_factor: int = 1, 
                          config: Optional[Dict[str, str]] = None):
        """Create Kafka topic with configuration"""
        try:
            from kafka.admin import NewTopic
            
            topic_config = config or {
                'retention.ms': str(7 * 24 * 60 * 60 * 1000),  # 7 days
                'cleanup.policy': 'delete',
                'compression.type': 'gzip'
            }
            
            new_topic = NewTopic(
                name=topic,
                num_partitions=partitions,
                replication_factor=replication_factor,
                topic_configs=topic_config
            )
            
            self.admin_client.create_topics([new_topic])
            self.logger.info(f"Created topic: {topic} with {partitions} partitions")
            
        except TopicAlreadyExistsError:
            self.logger.debug(f"Topic {topic} already exists")
        except Exception as e:
            self.logger.error(f"Failed to create topic {topic}: {e}")
            raise
    
    async def publish(self, topic: str, message: Message, partition: Optional[int] = None) -> bool:
        """Publish message to topic with retry and monitoring"""
        if not self.producer:
            raise RuntimeError("Message queue not initialized")
        
        # Check if message is expired
        if message.is_expired():
            self.logger.warning(f"Message {message.id} expired, not publishing")
            if PROMETHEUS_AVAILABLE:
                messages_published.labels(topic=topic, status='expired').inc()
            return False
        
        start_time = time.time()
        
        try:
            # Prepare message
            message_json = message.to_json()
            
            # Determine partition key for message ordering
            partition_key = None
            if message.correlation_id:
                partition_key = message.correlation_id
            elif message.data.get('symbol'):
                partition_key = message.data['symbol']
            
            # Create headers
            headers = [
                ('message-type', message.type.encode('utf-8')),
                ('source', message.source.encode('utf-8')),
                ('priority', message.priority.value.encode('utf-8')),
                ('correlation-id', (message.correlation_id or '').encode('utf-8')),
                ('timestamp', message.timestamp.encode('utf-8'))
            ]
            
            if message.reply_to:
                headers.append(('reply-to', message.reply_to.encode('utf-8')))
            
            # Send message
            await self.producer.send(
                topic,
                value=message_json,
                key=partition_key,
                partition=partition,
                headers=headers
            )
            
            # Update metrics
            processing_time = time.time() - start_time
            self.message_stats['published'][topic] += 1
            
            if PROMETHEUS_AVAILABLE:
                messages_published.labels(topic=topic, status='success').inc()
                message_processing_time.labels(topic=topic, handler='publisher').observe(processing_time)
            
            self.logger.debug(f"Published message {message.id} to {topic}")
            return True
            
        except Exception as e:
            self.message_stats['errors'][f'{topic}_publish'] += 1
            
            if PROMETHEUS_AVAILABLE:
                messages_published.labels(topic=topic, status='error').inc()
                message_errors.labels(topic=topic, error_type='publish').inc()
            
            self.logger.error(f"Failed to publish message {message.id} to {topic}: {e}")
            
            # Send to dead letter queue if max retries exceeded
            if not message.should_retry():
                await self._send_to_dlq(topic, message, str(e))
            
            raise
    
    async def subscribe(self, topic: str, handler: Callable, group_id: Optional[str] = None,
                       auto_create_topic: bool = True) -> bool:
        """Subscribe to topic with handler"""
        try:
            # Auto-create topic if requested
            if auto_create_topic:
                await self.create_topic(topic)
            
            consumer_group = group_id or f"{self.client_id}-{topic}"
            consumer_key = f"{topic}:{consumer_group}"
            
            # Create consumer if not exists
            if consumer_key not in self.consumers:
                consumer_config = self.consumer_config.copy()
                consumer_config['group_id'] = consumer_group
                consumer_config['client_id'] = f'{self.client_id}-consumer-{topic}'
                
                consumer = AIOKafkaConsumer(topic, **consumer_config)
                await consumer.start()
                
                self.consumers[consumer_key] = consumer
                
                # Start consumer task
                task = asyncio.create_task(self._consume_messages(topic, consumer, consumer_key))
                self.consumer_tasks.append(task)
                
                if PROMETHEUS_AVAILABLE:
                    active_consumers.labels(topic=topic).inc()
                
                self.logger.info(f"Created consumer for topic {topic} with group {consumer_group}")
            
            # Register handler
            self.handlers[consumer_key].append(handler)
            self.logger.info(f"Registered handler for topic {topic}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to subscribe to topic {topic}: {e}")
            return False
    
    async def _consume_messages(self, topic: str, consumer, consumer_key: str):
        """Consume messages from topic with error handling"""
        self.logger.info(f"Starting message consumption for {topic}")
        
        while self.is_running:
            try:
                # Get message batch
                msg_batch = await consumer.getmany(timeout_ms=1000, max_records=100)
                
                for topic_partition, messages in msg_batch.items():
                    for msg in messages:
                        start_time = time.time()
                        
                        try:
                            # Parse message
                            message = Message.from_json(msg.value)
                            
                            # Check if message is expired
                            if message.is_expired():
                                self.logger.warning(f"Received expired message {message.id}")
                                continue
                            
                            # Process with all handlers
                            for handler in self.handlers[consumer_key]:
                                try:
                                    handler_name = handler.__name__ if hasattr(handler, '__name__') else 'unknown'
                                    
                                    # Call handler
                                    if asyncio.iscoroutinefunction(handler):
                                        await handler(message)
                                    else:
                                        handler(message)
                                    
                                    # Update metrics
                                    processing_time = time.time() - start_time
                                    
                                    if PROMETHEUS_AVAILABLE:
                                        message_processing_time.labels(
                                            topic=topic, 
                                            handler=handler_name
                                        ).observe(processing_time)
                                    
                                except Exception as e:
                                    self.logger.error(f"Handler error for message {message.id}: {e}")
                                    if PROMETHEUS_AVAILABLE:
                                        message_errors.labels(topic=topic, error_type='handler').inc()
                                    
                                    # Send to DLQ if handler consistently fails
                                    message.retry_count += 1
                                    if not message.should_retry():
                                        await self._send_to_dlq(topic, message, str(e))
                            
                            # Update consumption metrics
                            self.message_stats['consumed'][topic] += 1
                            if PROMETHEUS_AVAILABLE:
                                messages_consumed.labels(topic=topic, status='success').inc()
                            
                        except Exception as e:
                            self.logger.error(f"Error processing message from {topic}: {e}")
                            self.message_stats['errors'][f'{topic}_consume'] += 1
                            
                            if PROMETHEUS_AVAILABLE:
                                messages_consumed.labels(topic=topic, status='error').inc()
                                message_errors.labels(topic=topic, error_type='processing').inc()
                    
                    # Commit offset after successful processing
                    try:
                        await consumer.commit()
                    except Exception as e:
                        self.logger.error(f"Failed to commit offset for {topic}: {e}")
                
            except asyncio.CancelledError:
                self.logger.info(f"Consumer task for {topic} cancelled")
                break
            except Exception as e:
                self.logger.error(f"Consumer error for {topic}: {e}")
                if PROMETHEUS_AVAILABLE:
                    message_errors.labels(topic=topic, error_type='consumer').inc()
                
                # Wait before retrying
                await asyncio.sleep(5)
    
    async def _send_to_dlq(self, original_topic: str, message: Message, error_message: str):
        """Send failed message to dead letter queue"""
        dlq_topic = f"{original_topic}{self.dlq_suffix}"
        
        try:
            # Add error information to metadata
            if not message.metadata:
                message.metadata = {}
            
            message.metadata.update({
                'original_topic': original_topic,
                'error_message': error_message,
                'failed_at': datetime.utcnow().isoformat(),
                'retry_count': message.retry_count
            })
            
            # Create DLQ topic if needed
            await self.create_topic(dlq_topic, partitions=1)
            
            # Send to DLQ
            dlq_message = Message(
                id=f"dlq_{message.id}",
                timestamp=datetime.utcnow().isoformat(),
                type=f"dlq_{message.type}",
                source=message.source,
                data=message.data,
                metadata=message.metadata
            )
            
            await self.publish(dlq_topic, dlq_message)
            self.logger.warning(f"Sent message {message.id} to DLQ: {dlq_topic}")
            
        except Exception as e:
            self.logger.error(f"Failed to send message {message.id} to DLQ: {e}")
    
    async def get_consumer_lag(self, topic: str, group_id: str) -> Dict[int, int]:
        """Get consumer lag for topic partitions"""
        try:
            # This would require additional Kafka admin operations
            # For now, return empty dict
            return {}
        except Exception as e:
            self.logger.error(f"Failed to get consumer lag for {topic}: {e}")
            return {}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get message queue statistics"""
        return {
            'is_running': self.is_running,
            'brokers': self.kafka_brokers,
            'active_consumers': len(self.consumers),
            'consumer_tasks': len(self.consumer_tasks),
            'message_stats': dict(self.message_stats),
            'topics': list(set(
                topic.split(':')[0] for topic in self.consumers.keys()
            ))
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        if not self.is_running or not self.producer:
            return {
                'status': 'critical',
                'message': 'Message queue not running',
                'metrics': {}
            }
        
        try:
            # Test producer connectivity
            metadata = await self.producer.client.fetch_metadata()
            
            healthy_brokers = len([
                broker for broker in metadata.brokers 
                if broker.nodeId >= 0
            ])
            
            return {
                'status': 'healthy' if healthy_brokers > 0 else 'critical',
                'message': f'{healthy_brokers} brokers available',
                'metrics': {
                    'healthy_brokers': healthy_brokers,
                    'total_brokers': len(metadata.brokers),
                    'active_consumers': len(self.consumers),
                    'messages_published': sum(self.message_stats['published'].values()),
                    'messages_consumed': sum(self.message_stats['consumed'].values()),
                    'total_errors': sum(self.message_stats['errors'].values())
                }
            }
            
        except Exception as e:
            return {
                'status': 'critical',
                'message': f'Health check failed: {str(e)}',
                'metrics': {'error': str(e)}
            }
    
    async def close(self):
        """Gracefully close all connections"""
        self.logger.info("Shutting down message queue...")
        self.is_running = False
        
        # Cancel consumer tasks
        for task in self.consumer_tasks:
            if not task.done():
                task.cancel()
        
        if self.consumer_tasks:
            await asyncio.gather(*self.consumer_tasks, return_exceptions=True)
        
        # Close consumers
        for consumer in self.consumers.values():
            try:
                await consumer.stop()
            except Exception as e:
                self.logger.error(f"Error closing consumer: {e}")
        
        # Close producer
        if self.producer:
            try:
                await self.producer.stop()
            except Exception as e:
                self.logger.error(f"Error closing producer: {e}")
        
        self.logger.info("Message queue shutdown complete")

# ===================== CONVENIENCE FUNCTIONS =====================

async def publish_trade_signal(symbol: str, signal_type: str, data: Dict[str, Any], 
                              priority: MessagePriority = MessagePriority.HIGH):
    """Convenience function to publish trade signals"""
    queue_manager = get_message_queue_manager()
    
    message = Message(
        id=str(uuid.uuid4()),
        timestamp=datetime.utcnow().isoformat(),
        type=MessageType.TRADE_SIGNAL.value,
        source='trading_engine',
        data={
            'symbol': symbol,
            'signal_type': signal_type,
            **data
        },
        priority=priority
    )
    
    return await queue_manager.publish('trade_signals', message)

async def publish_risk_alert(alert_type: str, message_data: Dict[str, Any]):
    """Convenience function to publish risk alerts"""
    queue_manager = get_message_queue_manager()
    
    message = Message(
        id=str(uuid.uuid4()),
        timestamp=datetime.utcnow().isoformat(),
        type=MessageType.RISK_ALERT.value,
        source='risk_engine',
        data={
            'alert_type': alert_type,
            **message_data
        },
        priority=MessagePriority.CRITICAL
    )
    
    return await queue_manager.publish('risk_alerts', message)

# ===================== GLOBAL INSTANCE =====================

_message_queue_manager = None

def get_message_queue_manager() -> MessageQueueManager:
    """Get global message queue manager instance"""
    global _message_queue_manager
    if _message_queue_manager is None:
        _message_queue_manager = MessageQueueManager()
    return _message_queue_manager

async def initialize_message_queue():
    """Initialize message queue"""
    queue_manager = get_message_queue_manager()
    success = await queue_manager.initialize()
    
    if success:
        # Register health check
        from infrastructure.monitoring import get_health_monitor
        health_monitor = get_health_monitor()
        health_monitor.register_health_check('message_queue', queue_manager.health_check)
    
    return success
