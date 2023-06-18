#!/bin/bash

# Get the current working directory.
repo_path=$(pwd)

# Append /backend to the path.
repo_path_backend="$repo_path/backend"
app_path="$repo_path_backend/app.py"

# Check if the backend directory and app.py exist.
if [ -d "$repo_path_backend" ] && [ -f "$app_path" ]
then
    # The directory and app.py exist.
    # Append the repo path to the PYTHONPATH and set FLASK_APP.
    export PYTHONPATH=$PYTHONPATH:$repo_path_backend
    export FLASK_APP=$app_path

    echo "Path added to PYTHONPATH and FLASK_APP set"

    # Run the Flask development server.
    flask run
else
    # The directory or app.py doesn't exist.
    echo "Error: $repo_path_backend or $app_path does not exist."
    exit 1
fi
