# Get the name of the repo export 
MY_REPO=$(ls -d "/home/onyxia/work"/*/ | head -n 1 | xargs basename) 

# Install the last version of GDAL
# curl -sL "https://url.geocarpentry.org/gdal-ubuntu" | bash

# Install dependencies
cd $MY_REPO
uv sync

# Set VSCode's default interpreter path
cd ..

if [ -d .vscode ]; then rm -rf .vscode; fi
mkdir -p .vscode

echo ${workspaceFolder}

# settings.json
cat > .vscode/settings.json << EOF
{
  "python.defaultInterpreterPath": "./$MY_REPO/.venv/bin/python",
  "python.analysis.extraPaths": ["\${workspaceFolder}"],
  "jupyter.notebookFileRoot": "\${workspaceFolder}/$MY_REPO"
}
EOF