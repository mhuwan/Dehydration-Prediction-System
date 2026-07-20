import json
import time
import requests
import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner

# --- Config ---
st.set_page_config(page_title="Dehydration Prediction", page_icon="💧", layout="wide")

# --- CSS Decoration ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%);
    }
    .main-header {
        text-align: center;
        padding: 2rem 0;
        color: #2c3e50;
    }
    .main-header h1 {
        font-size: 3.5rem;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .card {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 50px;
        padding: 0.6rem 2rem;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# --- Lottie Functions ---
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_url_hello = "https://lottie.host/3f647b41-61bf-4d39-93c3-0433420604cc/8NtmEbAWmO.json"
lottie_url_download = "https://lottie.host/290016a6-d650-4a32-85c7-4d77b0a892ca/4rNL8XZeZt.json"
lottie_hello = load_lottieurl(lottie_url_hello)
lottie_download = load_lottieurl(lottie_url_download)

# --- UI ---
st.markdown('<div class="main-header"><h1>💧 ระบบทำนายภาวะขาดน้ำ</h1><h3>Dehydration Prediction System using SVM</h3></div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("👋 ยินดีต้อนรับสู่ระบบวิเคราะห์ภาวะขาดน้ำ")
    st.write("""
    ระบบนี้ถูกพัฒนาขึ้นเพื่อช่วยคัดกรองและทำนายภาวะขาดน้ำ (Dehydration) 
    โดยใช้ Machine Learning อัลกอริทึม **Support Vector Machine (SVM)** 
    วิเคราะห์จากอาการทางคลินิกและสัญญาณชีพต่างๆ
    """)
    st.markdown("#### 🚀 เริ่มต้นใช้งาน")
    st.write("เลือกเมนูด้านซ้ายมือเพื่อเริ่มต้น:")
    st.write("- **📊 วิเคราะห์ข้อมูล**: สำรวจชุดข้อมูลและสถิติ")
    st.write("- **🔮 ทำนายภาวะขาดน้ำ**: กรอกอาการเพื่อทำนายผล")
    st.write("- **📈 ประสิทธิภาพโมเดล**: ดูค่าความแม่นยำของ SVM")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card" style="text-align:center;">', unsafe_allow_html=True)
    st_lottie(lottie_hello, key="hello", height=250)
    
    if st.button("🔄 โหลดข้อมูล Kaggle"):
        with st_lottie_spinner(lottie_download, key="download", height=100):
            from utils import load_kaggle_data
            df = load_kaggle_data()
            time.sleep(2)
        st.balloons()
        st.success(f"โหลดข้อมูลสำเร็จ! ({len(df)} รายการ)")
    st.markdown('</div>', unsafe_allow_html=True)