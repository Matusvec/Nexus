#!/bin/bash
set -e

# Run Alembic migrations before starting the application
echo "Running database migrations..."
alembic upgrade head

echo "Starting application..."
exec "$@"
