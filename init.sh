#!/bin/bash

# 1. Install System Dependencies (Added libopenmpi-dev)
if ! command -v cmake &> /dev/null || ! ldconfig -p | grep -q libmpi; then
    echo "Dependencies missing. Installing..."
    sudo apt-get update
    # cmake and build-essential for dlib
    # libopenmpi-dev for mpi4py
    sudo apt-get install -y cmake build-essential python3-dev libopenmpi-dev
else
    echo "System dependencies already present."
fi

# 2. Identify the repository
# Using -maxdepth 1 to ensure we only look at top-level folders in /work
MY_REPO=$(find /home/onyxia/work -maxdepth 1 -mindepth 1 -type d | head -n 1 | xargs basename)

# 3. Enter the repo and sync Python environment
cd "/home/onyxia/work/$MY_REPO"

# This will now succeed because cmake is present
uv sync

# 4. Set VSCode's default interpreter path
cd /home/onyxia/work
workspaceFolder=$(pwd)

# Recreate .vscode directory
if [ -d .vscode ]; then rm -rf .vscode; fi
mkdir -p .vscode

# Generate settings.json
cat > .vscode/settings.json << EOF
{
  "python.defaultInterpreterPath": "${workspaceFolder}/$MY_REPO/.venv/bin/python",
  "python.analysis.extraPaths": ["${workspaceFolder}/${MY_REPO}"],
  "jupyter.notebookFileRoot": "${workspaceFolder}/${MY_REPO}"
}
EOF