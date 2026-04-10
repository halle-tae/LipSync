#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== LipSync App Startup Script ===${NC}\n"

# Function to kill process on a specific port
kill_port() {
    local port=$1
    echo -e "${YELLOW}Checking port $port...${NC}"

    # Find PIDs using the port
    local pids=$(netstat -ano | grep ":$port " | awk '{print $5}' | sort -u)

    if [ -z "$pids" ]; then
        echo -e "${GREEN}Port $port is free${NC}"
    else
        echo -e "${RED}Port $port is in use. Killing processes...${NC}"
        for pid in $pids; do
            if [ "$pid" != "0" ]; then
                echo "  Killing PID: $pid"
                taskkill //PID $pid //F 2>/dev/null
            fi
        done
        sleep 1
        echo -e "${GREEN}Port $port is now free${NC}"
    fi
}

# Kill processes on ports 3000 and 8000
kill_port 3000
kill_port 8000

echo -e "\n${GREEN}=== Starting Backend (Port 8000) ===${NC}"
cd backend
python api_server.py &
BACKEND_PID=$!
cd ..

# Wait a bit for backend to start
sleep 3

echo -e "\n${GREEN}=== Starting Frontend (Port 3000) ===${NC}"
cd web
npm run dev &
FRONTEND_PID=$!
cd ..

echo -e "\n${GREEN}=== Both services started! ===${NC}"
echo -e "Backend PID: ${BACKEND_PID}"
echo -e "Frontend PID: ${FRONTEND_PID}"
echo -e "\n${YELLOW}Frontend: http://localhost:3000${NC}"
echo -e "${YELLOW}Backend: http://localhost:8000${NC}"
echo -e "\n${RED}Press Ctrl+C to stop both services${NC}\n"

# Trap to kill both processes on exit
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM

# Wait for both processes
wait
