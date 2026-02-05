#!/bin/bash
# Database setup script for ClaimGuard

set -e  # Exit on error

echo "🛡️  ClaimGuard Database Setup"
echo "================================"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop first."
    exit 1
fi

echo "✅ Docker is running"
echo ""

# Start Docker services
echo "🐳 Starting Docker services..."
docker-compose up -d postgres redis neo4j qdrant minio

# Wait for PostgreSQL to be ready
echo "⏳ Waiting for PostgreSQL to be ready..."
until docker-compose exec -T postgres pg_isready -U claimguard_user -d claimguard > /dev/null 2>&1; do
    sleep 1
done

echo "✅ PostgreSQL is ready"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  No .env file found. Creating from .env.example..."
    cp .env.example .env
    echo "📝 Please edit .env and add your OPENAI_API_KEY"
    echo ""
fi

# Create initial migration
echo "🔄 Creating initial database migration..."
alembic revision --autogenerate -m "Initial schema with all models"

echo ""
echo "🚀 Applying migration..."
alembic upgrade head

echo ""
echo "✅ Database setup complete!"
echo ""
echo "Services running:"
echo "  - PostgreSQL: localhost:5432"
echo "  - Redis: localhost:6379"
echo "  - Neo4j: localhost:7474 (Browser), localhost:7687 (Bolt)"
echo "  - Qdrant: localhost:6333"
echo "  - MinIO: localhost:9000 (API), localhost:9001 (Console)"
echo ""
echo "Next steps:"
echo "  1. Add your OPENAI_API_KEY to .env file"
echo "  2. Run: python scripts/ingest_kaggle_data.py"
echo "  3. Run: python scripts/ingest_vehide_data.py"
echo ""
