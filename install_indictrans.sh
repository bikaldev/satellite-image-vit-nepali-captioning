#!/bin/bash
# Install IndicTrans2 from GitHub repository

echo "Installing IndicTrans2 from AI4Bharat GitHub repository..."

# Activate virtual environment if it exists
if [ -d "env" ]; then
    source env/bin/activate
fi

# Install IndicTrans2 from GitHub
pip install git+https://github.com/AI4Bharat/IndicTrans2.git

echo "IndicTrans2 installation complete!"
