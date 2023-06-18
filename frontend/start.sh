#!/bin/bash

# Check if the frontend directory exists.
if [ -d "./frontend" ]
then
    # The directory exists.
    cd frontend

    # Start the application.
    npm start
else
    # The directory doesn't exist.
    echo "Error: frontend directory does not exist in the current location."
    exit 1
fi
