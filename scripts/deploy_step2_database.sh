#!/bin/bash
# Deploy Step 2: World-Class Database Layer

set -e

echo "🚀 Deploying Step 2: World-Class Database Layer"

# Check prerequisites
command -v docker >/dev/null 2>&1 || { echo "Docker required but not installed. Aborting." >&2; exit 1; }
command -v psql >/dev/null 2>&1 || { echo "PostgreSQL client required but not installed. Aborting." >&2; exit 1; }

# Start database services
echo "📦 Starting database services..."
docker-compose -f deployment/docker-compose.db.yml up -d

# Wait for PostgreSQL to be ready
echo "⏳ Waiting for PostgreSQL..."
until docker exec omni-postgres pg_isready -U omni; do
    sleep 1
done

# Wait for TimescaleDB to be ready
echo "⏳ Waiting for TimescaleDB..."
until docker exec omni-timescale pg_isready -U omni; do
    sleep 1
done

# Run migrations
echo "🔄 Running database migrations..."
psql -h localhost -p 5432 -U omni -d omni_alpha -f src/database/migrations/001_initial_schema.sql
psql -h localhost -p 5433 -U omni -d market_data -f src/database/migrations/002_timescale_setup.sql

# Verify setup
echo "✅ Verifying database setup..."
python -c "
import asyncio
from src.database.connection import db_manager

async def verify():
    await db_manager.initialize()
    stats = await db_manager.get_pool_stats()
    print('Database connections verified:', stats)
    await db_manager.close()

asyncio.run(verify())
"

# Run tests
echo "🧪 Running database tests..."
pytest tests/test_step2_database.py -v

echo "✅ Step 2: Database Layer deployed successfully!"
echo "📊 Database endpoints:"
echo "   - PostgreSQL: localhost:5432"
echo "   - TimescaleDB: localhost:5433"
echo "   - Redis: localhost:6379"
echo "   - MongoDB: localhost:27017"
