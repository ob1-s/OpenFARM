#!/bin/bash

CUSTOM_DIR=".openfarm/skills"
# Add any agent folders you use here
AGENT_DIRS=(".agents/skills")

echo "Syncing custom skills..."

for agent_dir in "${AGENT_DIRS[@]}"; do
    # Ensure the target directory exists
    mkdir -p "$agent_dir"
    
    # Loop through every skill in your custom folder
    for skill_path in "$CUSTOM_DIR"/*; do
        if [ -d "$skill_path" ]; then
            skill_name=$(basename "$skill_path")
            
            # -s: symbolic, -f: force (overwrite if exists), -n: treat dest as normal file
            ln -sfn "../../$CUSTOM_DIR/$skill_name" "$agent_dir/$skill_name"
        fi
    done
done