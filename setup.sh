#!/bin/bash

# สร้างโฟลเดอร์ที่จำเป็น
mkdir -p .streamlit
mkdir -p ~/.kaggle
mkdir -p data

# ตั้งค่า Streamlit config
cat > .streamlit/config.toml <<EOF
[server]
headless = true
port = $PORT
enableCORS = false

[browser]
gatherUsageStats = false
EOF

# สร้างไฟล์ kaggle.json จาก Environment Variables
if [ -n "$KAGGLE_USERNAME" ] && [ -n "$KAGGLE_KEY" ]; then
    echo "{\"username\":\"$KAGGLE_USERNAME\",\"key\":\"$KAGGLE_KEY\"}" > ~/.kaggle/kaggle.json
    chmod 600 ~/.kaggle/kaggle.json
    echo "✅ Kaggle credentials configured"
else
    echo "⚠️ Kaggle credentials not found, will use synthetic data"
fi