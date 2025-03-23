#!/bin/bash

# Activate the virtual environment
source ../robot-env/bin/activate

# Run the face detection script
python face_detection_mac.py

# Deactivate the virtual environment when done
deactivate