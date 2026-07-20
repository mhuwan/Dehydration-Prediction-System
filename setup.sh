#!/bin/bash
# สร้างไฟล์ config ของ Streamlit
mkdir -p .streamlit
echo "[server]" > .streamlit/config.toml
echo "headless = true" >> .streamlit/config.toml
echo "port = \$PORT" >> .streamlit/config.toml

# สร้างไฟล์ kaggle.json จาก Environment Variables (สำหรับ Deploy)
if [ -n "$KAGGLE_USERNAME" ] && [ -n "$KAGGLE_KEY" ]; then
    mkdir -p ~/.kaggle
    echo "{\"username\":\"$KAGGLE_USERNAME\",\"key\":\"$KAGGLE_KEY\"}" > ~/.kaggle/kaggle.json
    chmod 600 ~/.kaggle/kaggle.json
fi