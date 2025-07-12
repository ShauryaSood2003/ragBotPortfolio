#!/bin/bash

echo "Starting RAG Bot..."

echo "Starting Milvus with Docker Compose..."
docker-compose up -d

echo "Waiting for Milvus to be ready..."
sleep 30

echo "Activating Python virtual environment..."
source venv/bin/activate

echo "Starting FastAPI server..."
python3 main.py