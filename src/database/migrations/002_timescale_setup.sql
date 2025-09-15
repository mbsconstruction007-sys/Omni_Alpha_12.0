-- TimescaleDB setup for market data
-- Create TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Switch to market_data schema
SET search_path TO market_data, public;

-- Market ticks table (hypertable)
CREATE TABLE IF NOT EXISTS market_ticks (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    bid DECIMAL(20, 8) NOT NULL,
    ask DECIMAL(20, 8) NOT NULL,
    last DECIMAL(20, 8) NOT NULL,
    mid DECIMAL(20, 8),
    bid_size INTEGER NOT NULL,
    ask_size INTEGER NOT NULL,
    last_size INTEGER,
    exchange VARCHAR(20),
    conditions TEXT[]
);

-- Convert to hypertable
SELECT create_hypertable('market_ticks', 'time', 
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

-- OHLCV data (hypertable)
CREATE TABLE IF NOT EXISTS ohlcv (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    open DECIMAL(20, 8) NOT NULL,
    high DECIMAL(20, 8) NOT NULL,
    low DECIMAL(20, 8) NOT NULL,
    close DECIMAL(20, 8) NOT NULL,
    volume BIGINT NOT NULL,
    vwap DECIMAL(20, 8),
    trade_count INTEGER,
    period_start TIMESTAMPTZ NOT NULL,
    period_end TIMESTAMPTZ NOT NULL
);

-- Convert to hypertable
SELECT create_hypertable('ohlcv', 'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE);

-- Create indexes
CREATE INDEX idx_ticks_symbol_time ON market_ticks(symbol, time DESC);
CREATE INDEX idx_ohlcv_symbol_timeframe_time ON ohlcv(symbol, timeframe, time DESC);

-- Create continuous aggregates for different timeframes
CREATE MATERIALIZED VIEW ohlcv_1min
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', time) AS bucket,
    symbol,
    FIRST(bid, time) AS open,
    MAX(bid) AS high,
    MIN(bid) AS low,
    LAST(bid, time) AS close,
    SUM(bid_size) AS volume
FROM market_ticks
GROUP BY bucket, symbol
WITH NO DATA;

-- Refresh policy for continuous aggregates
SELECT add_continuous_aggregate_policy('ohlcv_1min',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute',
    if_not_exists => TRUE);

-- Compression policy for old data
SELECT add_compression_policy('market_ticks', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_compression_policy('ohlcv', INTERVAL '30 days', if_not_exists => TRUE);

-- Retention policy (optional - keeps 1 year of tick data)
SELECT add_retention_policy('market_ticks', INTERVAL '1 year', if_not_exists => TRUE);

-- Grant permissions
GRANT ALL ON SCHEMA market_data TO omni;
GRANT ALL ON ALL TABLES IN SCHEMA market_data TO omni;
