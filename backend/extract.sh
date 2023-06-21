#!/bin/bash

# Get the current working directory.
repo_path=$(pwd)

# Append /backend to the path.
repo_path_backend="$repo_path/backend"
extraction_script="$repo_path_backend/iam/extraction.py"

# Check if the backend directory and extraction.py exist.
if [ -d "$repo_path_backend" ] && [ -f "$extraction_script" ]
then
    # The directory and extraction.py exist.
    # Append the repo path to the PYTHONPATH.
    export PYTHONPATH=$PYTHONPATH:$repo_path_backend

    echo "Path added to PYTHONPATH"

    # Run the extraction.py script.
    python $extraction_script
else
    # The directory or extraction.py doesn't exist.
    echo "Error: $repo_path_backend or $extraction_script does not exist."
    exit 1
fi
