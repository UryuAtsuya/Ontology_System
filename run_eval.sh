#!/bin/bash

# Use the python interpreter that has the dependencies installed
PYTHON_PATH="/usr/local/opt/python@3.11/bin/python3.11"

echo "Using Python: $PYTHON_PATH"
"$PYTHON_PATH" run_eval.py
