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

    # 1. Neo4j
    echo "  [1/4] Neo4j..."
    docker compose up neo4j -d 2>/dev/null
    sleep 3
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
    source venv/bin/activate 2>/dev/null || true
    python3 -m uvicorn src.api:app --port 8000 > "$LOG_DIR/backend.log" 2>&1 &
    echo $! > "$LOG_DIR/backend.pid"
    sleep 3
    echo "        http://localhost:8000"

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

    # Ollama
    if [ -f "$LOG_DIR/ollama.pid" ]; then
        kill $(cat "$LOG_DIR/ollama.pid") 2>/dev/null
        rm "$LOG_DIR/ollama.pid"
        echo "  Ollama stopped"
    fi
    pkill -x ollama 2>/dev/null

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
    if docker compose ps 2>/dev/null | grep -q "neo4j.*running"; then
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