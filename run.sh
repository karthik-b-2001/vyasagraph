#!/bin/bash

# VyasaGraph service manager
# Usage:
#   ./run.sh start    Start all services
#   ./run.sh stop     Stop all services
#   ./run.sh status   Check what's running

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
FRONTEND_DIR="$PROJECT_DIR/frontend"
LOG_DIR="$PROJECT_DIR/.logs"

mkdir -p "$LOG_DIR"

start() {
    echo "Starting VyasaGraph..."
    echo ""

    INDEX_DIR="${INDEX_DIR:-$PROJECT_DIR/data/index}"
    if [ ! -f "$INDEX_DIR/chunks.jsonl" ]; then
        echo "  No vector index found. Build the data first:"
        echo "    docker compose up neo4j -d --wait && python -m src.ingest"
        exit 1
    fi

    # 1. Neo4j
    echo "  [1/4] Neo4j..."
    cd "$PROJECT_DIR"
    if ! docker compose up neo4j -d --wait; then
        echo "        Neo4j failed to start (is Docker running?)"
        exit 1
    fi
    echo "        http://localhost:7474"

    # 2. Ollama
    echo "  [2/4] Ollama..."
    if ! pgrep -x "ollama" > /dev/null 2>&1; then
        ollama serve > "$LOG_DIR/ollama.log" 2>&1 &
        echo $! > "$LOG_DIR/ollama.pid"
        sleep 2
    else
        echo "        already running"
    fi

    # 3. FastAPI backend
    echo "  [3/4] FastAPI backend..."
    cd "$PROJECT_DIR"
    if curl -s -o /dev/null http://localhost:8000; then
        echo "        port 8000 is already in use; run ./run.sh stop first"
        exit 1
    fi
    source venv/bin/activate 2>/dev/null || true
    python3 -m uvicorn src.api:app --port 8000 > "$LOG_DIR/backend.log" 2>&1 &
    echo $! > "$LOG_DIR/backend.pid"
    # Loading the embedding model takes a few seconds; wait until it answers.
    for _ in $(seq 1 60); do
        curl -sf http://localhost:8000/api/health > /dev/null && break
        sleep 1
    done
    if curl -sf http://localhost:8000/api/health > /dev/null; then
        echo "        http://localhost:8000"
    else
        echo "        failed to start, see $LOG_DIR/backend.log"
        exit 1
    fi

    # 4. Frontend
    echo "  [4/4] Frontend..."
    cd "$FRONTEND_DIR"
    npm run dev > "$LOG_DIR/frontend.log" 2>&1 &
    echo $! > "$LOG_DIR/frontend.pid"
    sleep 2
    echo "        http://localhost:5173"

    echo ""
    echo "  All services running."
    echo "  Open http://localhost:5173"
    echo ""
}

stop() {
    echo "Stopping VyasaGraph..."
    echo ""

    # Frontend
    if [ -f "$LOG_DIR/frontend.pid" ]; then
        kill $(cat "$LOG_DIR/frontend.pid") 2>/dev/null
        rm "$LOG_DIR/frontend.pid"
        echo "  Frontend stopped"
    fi

    # Backend
    if [ -f "$LOG_DIR/backend.pid" ]; then
        kill $(cat "$LOG_DIR/backend.pid") 2>/dev/null
        rm "$LOG_DIR/backend.pid"
        echo "  Backend stopped"
    fi

    # Also kill any uvicorn processes for this project
    pkill -f "uvicorn src.api:app" 2>/dev/null

    # Ollama (only if this script started it)
    if [ -f "$LOG_DIR/ollama.pid" ]; then
        kill $(cat "$LOG_DIR/ollama.pid") 2>/dev/null
        rm "$LOG_DIR/ollama.pid"
        echo "  Ollama stopped"
    fi

    # Neo4j
    cd "$PROJECT_DIR"
    docker compose down 2>/dev/null
    echo "  Neo4j stopped"

    echo ""
    echo "  All services stopped."
    echo ""
}

status() {
    echo "VyasaGraph status:"
    echo ""

    # Neo4j
    cd "$PROJECT_DIR"
    if docker compose ps --status running --services 2>/dev/null | grep -qx neo4j; then
        echo "  Neo4j:    running (http://localhost:7474)"
    else
        echo "  Neo4j:    stopped"
    fi

    # Ollama
    if pgrep -x "ollama" > /dev/null 2>&1; then
        echo "  Ollama:   running"
    else
        echo "  Ollama:   stopped"
    fi

    # Backend
    if pgrep -f "uvicorn src.api:app" > /dev/null 2>&1; then
        echo "  Backend:  running (http://localhost:8000)"
    else
        echo "  Backend:  stopped"
    fi

    # Frontend
    if [ -f "$LOG_DIR/frontend.pid" ] && kill -0 $(cat "$LOG_DIR/frontend.pid") 2>/dev/null; then
        echo "  Frontend: running (http://localhost:5173)"
    else
        echo "  Frontend: stopped"
    fi

    echo ""
}

case "$1" in
    start)  start ;;
    stop)   stop ;;
    status) status ;;
    *)
        echo "Usage: ./run.sh {start|stop|status}"
        exit 1
        ;;
esac