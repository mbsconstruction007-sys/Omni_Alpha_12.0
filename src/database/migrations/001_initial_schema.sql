-- Initial database schema for Omni Alpha 5.0
-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text search
CREATE EXTENSION IF NOT EXISTS "btree_gin";  -- For compound indexes

-- Create schema
CREATE SCHEMA IF NOT EXISTS trading;
CREATE SCHEMA IF NOT EXISTS market_data;
CREATE SCHEMA IF NOT EXISTS analytics;

-- Set search path
SET search_path TO trading, public;

-- Accounts table
CREATE TABLE IF NOT EXISTS accounts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id VARCHAR(50) UNIQUE NOT NULL,
    account_type VARCHAR(20) NOT NULL,
    cash_balance DECIMAL(20, 4) NOT NULL DEFAULT 0,
    buying_power DECIMAL(20, 4) NOT NULL DEFAULT 0,
    portfolio_value DECIMAL(20, 4) NOT NULL DEFAULT 0,
    margin_used DECIMAL(20, 4) DEFAULT 0,
    margin_available DECIMAL(20, 4) DEFAULT 0,
    maintenance_margin DECIMAL(20, 4) DEFAULT 0,
    daily_pnl DECIMAL(20, 4) DEFAULT 0,
    total_pnl DECIMAL(20, 4) DEFAULT 0,
    risk_score FLOAT DEFAULT 0,
    day_trade_count INTEGER DEFAULT 0,
    pattern_day_trader BOOLEAN DEFAULT FALSE,
    active BOOLEAN DEFAULT TRUE,
    restricted BOOLEAN DEFAULT FALSE,
    restriction_reason TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Orders table (partitioned by date for performance)
CREATE TABLE IF NOT EXISTS orders (
    id UUID DEFAULT uuid_generate_v4(),
    order_id VARCHAR(100) UNIQUE NOT NULL,
    client_order_id VARCHAR(100),
    parent_order_id VARCHAR(100),
    account_id VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    asset_type VARCHAR(20) NOT NULL,
    exchange VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    order_type VARCHAR(20) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    filled_quantity DECIMAL(20, 8) DEFAULT 0,
    remaining_quantity DECIMAL(20, 8),
    limit_price DECIMAL(20, 8),
    stop_price DECIMAL(20, 8),
    average_fill_price DECIMAL(20, 8),
    time_in_force VARCHAR(10) DEFAULT 'day',
    expire_time TIMESTAMPTZ,
    submitted_at TIMESTAMPTZ,
    filled_at TIMESTAMPTZ,
    cancelled_at TIMESTAMPTZ,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    status_message TEXT,
    max_slippage DECIMAL(10, 4),
    position_size_pct DECIMAL(5, 2),
    stop_loss DECIMAL(20, 8),
    take_profit DECIMAL(20, 8),
    commission DECIMAL(20, 4) DEFAULT 0,
    fees DECIMAL(20, 4) DEFAULT 0,
    slippage DECIMAL(20, 4) DEFAULT 0,
    strategy_id VARCHAR(50),
    signal_id VARCHAR(50),
    tags TEXT[],
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

-- Create monthly partitions for orders
CREATE TABLE orders_2024_01 PARTITION OF orders FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
CREATE TABLE orders_2024_02 PARTITION OF orders FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');
CREATE TABLE orders_2024_03 PARTITION OF orders FOR VALUES FROM ('2024-03-01') TO ('2024-04-01');
CREATE TABLE orders_2024_04 PARTITION OF orders FOR VALUES FROM ('2024-04-01') TO ('2024-05-01');
CREATE TABLE orders_2024_05 PARTITION OF orders FOR VALUES FROM ('2024-05-01') TO ('2024-06-01');
CREATE TABLE orders_2024_06 PARTITION OF orders FOR VALUES FROM ('2024-06-01') TO ('2024-07-01');
CREATE TABLE orders_2024_07 PARTITION OF orders FOR VALUES FROM ('2024-07-01') TO ('2024-08-01');
CREATE TABLE orders_2024_08 PARTITION OF orders FOR VALUES FROM ('2024-08-01') TO ('2024-09-01');
CREATE TABLE orders_2024_09 PARTITION OF orders FOR VALUES FROM ('2024-09-01') TO ('2024-10-01');
CREATE TABLE orders_2024_10 PARTITION OF orders FOR VALUES FROM ('2024-10-01') TO ('2024-11-01');
CREATE TABLE orders_2024_11 PARTITION OF orders FOR VALUES FROM ('2024-11-01') TO ('2024-12-01');
CREATE TABLE orders_2024_12 PARTITION OF orders FOR VALUES FROM ('2024-12-01') TO ('2025-01-01');

-- Trades table (partitioned by date)
CREATE TABLE IF NOT EXISTS trades (
    id UUID DEFAULT uuid_generate_v4(),
    trade_id VARCHAR(100) UNIQUE NOT NULL,
    order_id VARCHAR(100) NOT NULL,
    account_id VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    price DECIMAL(20, 8) NOT NULL,
    commission DECIMAL(20, 4) DEFAULT 0,
    fees DECIMAL(20, 4) DEFAULT 0,
    executed_at TIMESTAMPTZ NOT NULL,
    settled_at TIMESTAMPTZ,
    exchange VARCHAR(20) NOT NULL,
    liquidity_indicator VARCHAR(10),
    execution_id VARCHAR(100),
    clearing_firm VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (id, executed_at)
) PARTITION BY RANGE (executed_at);

-- Create monthly partitions for trades
CREATE TABLE trades_2024_01 PARTITION OF trades FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
CREATE TABLE trades_2024_02 PARTITION OF trades FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');
CREATE TABLE trades_2024_03 PARTITION OF trades FOR VALUES FROM ('2024-03-01') TO ('2024-04-01');
CREATE TABLE trades_2024_04 PARTITION OF trades FOR VALUES FROM ('2024-04-01') TO ('2024-05-01');
CREATE TABLE trades_2024_05 PARTITION OF trades FOR VALUES FROM ('2024-05-01') TO ('2024-06-01');
CREATE TABLE trades_2024_06 PARTITION OF trades FOR VALUES FROM ('2024-06-01') TO ('2024-07-01');
CREATE TABLE trades_2024_07 PARTITION OF trades FOR VALUES FROM ('2024-07-01') TO ('2024-08-01');
CREATE TABLE trades_2024_08 PARTITION OF trades FOR VALUES FROM ('2024-08-01') TO ('2024-09-01');
CREATE TABLE trades_2024_09 PARTITION OF trades FOR VALUES FROM ('2024-09-01') TO ('2024-10-01');
CREATE TABLE trades_2024_10 PARTITION OF trades FOR VALUES FROM ('2024-10-01') TO ('2024-11-01');
CREATE TABLE trades_2024_11 PARTITION OF trades FOR VALUES FROM ('2024-11-01') TO ('2024-12-01');
CREATE TABLE trades_2024_12 PARTITION OF trades FOR VALUES FROM ('2024-12-01') TO ('2025-01-01');

-- Positions table
CREATE TABLE IF NOT EXISTS positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    position_id VARCHAR(50) UNIQUE NOT NULL,
    account_id VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    asset_type VARCHAR(20) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    available_quantity DECIMAL(20, 8) NOT NULL,
    locked_quantity DECIMAL(20, 8) DEFAULT 0,
    average_entry_price DECIMAL(20, 8) NOT NULL,
    current_price DECIMAL(20, 8),
    market_value DECIMAL(20, 4),
    unrealized_pnl DECIMAL(20, 4),
    realized_pnl DECIMAL(20, 4) DEFAULT 0,
    total_pnl DECIMAL(20, 4),
    pnl_percentage DECIMAL(10, 4),
    position_risk_score FLOAT,
    var_95 DECIMAL(20, 4),
    expected_shortfall DECIMAL(20, 4),
    strategy_id VARCHAR(50),
    opened_at TIMESTAMPTZ NOT NULL,
    last_modified TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Strategies table
CREATE TABLE IF NOT EXISTS strategies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    strategy_id VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    version VARCHAR(20) NOT NULL,
    enabled BOOLEAN DEFAULT TRUE,
    parameters JSONB NOT NULL,
    max_position_size DECIMAL(20, 4) NOT NULL,
    max_daily_loss DECIMAL(20, 4) NOT NULL,
    max_drawdown DECIMAL(20, 4) NOT NULL,
    position_limit INTEGER NOT NULL,
    total_pnl DECIMAL(20, 4) DEFAULT 0,
    win_rate FLOAT DEFAULT 0,
    sharpe_ratio FLOAT DEFAULT 0,
    max_drawdown_pct FLOAT DEFAULT 0,
    symbols TEXT[],
    asset_types TEXT[],
    status VARCHAR(20) DEFAULT 'active',
    last_signal_at TIMESTAMPTZ,
    last_trade_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Signals table
CREATE TABLE IF NOT EXISTS trading_signals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    signal_id VARCHAR(50) UNIQUE NOT NULL,
    strategy_id VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    action VARCHAR(10) NOT NULL,
    signal_strength FLOAT CHECK (signal_strength >= -1 AND signal_strength <= 1),
    confidence FLOAT CHECK (confidence >= 0 AND confidence <= 1),
    suggested_quantity DECIMAL(20, 8),
    suggested_price DECIMAL(20, 8),
    suggested_stop_loss DECIMAL(20, 8),
    suggested_take_profit DECIMAL(20, 8),
    valid_until TIMESTAMPTZ,
    indicators JSONB,
    reasoning TEXT,
    executed BOOLEAN DEFAULT FALSE,
    order_id VARCHAR(100),
    execution_time TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX idx_orders_account_id ON orders(account_id);
CREATE INDEX idx_orders_symbol ON orders(symbol);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_orders_created_at ON orders(created_at DESC);
CREATE INDEX idx_orders_strategy_id ON orders(strategy_id) WHERE strategy_id IS NOT NULL;

CREATE INDEX idx_trades_order_id ON trades(order_id);
CREATE INDEX idx_trades_symbol ON trades(symbol);
CREATE INDEX idx_trades_executed_at ON trades(executed_at DESC);

CREATE INDEX idx_positions_account_id ON positions(account_id);
CREATE INDEX idx_positions_symbol ON positions(symbol);

CREATE INDEX idx_signals_strategy_id ON trading_signals(strategy_id);
CREATE INDEX idx_signals_symbol ON trading_signals(symbol);
CREATE INDEX idx_signals_executed ON trading_signals(executed);

-- Create composite indexes for common queries
CREATE INDEX idx_orders_account_symbol ON orders(account_id, symbol);
CREATE INDEX idx_trades_account_symbol ON trades(account_id, symbol);
CREATE INDEX idx_positions_account_symbol ON positions(account_id, symbol);

-- Create triggers for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_accounts_updated_at BEFORE UPDATE ON accounts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    
CREATE TRIGGER update_orders_updated_at BEFORE UPDATE ON orders
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    
CREATE TRIGGER update_trades_updated_at BEFORE UPDATE ON trades
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    
CREATE TRIGGER update_positions_updated_at BEFORE UPDATE ON positions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    
CREATE TRIGGER update_strategies_updated_at BEFORE UPDATE ON strategies
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    
CREATE TRIGGER update_signals_updated_at BEFORE UPDATE ON trading_signals
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Performance views
CREATE OR REPLACE VIEW v_active_orders AS
SELECT * FROM orders
WHERE status IN ('pending', 'submitted', 'partially_filled')
ORDER BY created_at DESC;

CREATE OR REPLACE VIEW v_today_trades AS
SELECT * FROM trades
WHERE executed_at >= CURRENT_DATE
ORDER BY executed_at DESC;

CREATE OR REPLACE VIEW v_open_positions AS
SELECT p.*, 
       (p.current_price - p.average_entry_price) * p.quantity AS unrealized_pnl_calc
FROM positions p
WHERE p.quantity != 0;

-- Grant permissions
GRANT ALL ON SCHEMA trading TO omni;
GRANT ALL ON ALL TABLES IN SCHEMA trading TO omni;
GRANT ALL ON ALL SEQUENCES IN SCHEMA trading TO omni;
