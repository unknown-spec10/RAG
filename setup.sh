#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Create required directories
mkdir -p data/pdfs data/chroma_db

# Copy secrets template
if [ ! -f ".streamlit/secrets.toml" ]; then
    cp .streamlit/secrets.toml.example .streamlit/secrets.toml
fi
