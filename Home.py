import time
import requests
import streamlit as st
from streamlit_lottie import st_lottie, st_lottie_spinner
from utils import load_kaggle_data

# ============ Config ============
st.set_page_config(
    page_title="Dehydration Prediction",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ CSS ============
st.markdown("""
<style>
/* ===== Background ===== */
.stApp {
    background: linear-gradient(
        135deg,
        #e8f5ff 0%,
        #d9f7e8 50%,
        #ffffff 100%
    );
}

/* ===== Header ===== */
.main-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem 0;
}

.main-header h1 {
    font-size: 3.2rem;
    font-weight: 900;
    color: #0b3d5c;
    letter-spacing: 1px;
    text-shadow: 2px 3px 8px rgba(0,0,0,0.12);
}

.main-header h3 {
    color: #287d8c;
    font-size: 1.3rem;
    font-weight: 500;
}

/* ===== Card ===== */
.card {
    background: rgba(255,255,255,0.85);
    backdrop-filter: blur(15px);
    padding: 2rem;
    border-radius: 25px;

    box-shadow:
        0 10px 30px rgba(0,0,0,0.08);

    margin-bottom: 1.5rem;

    border:
        1px solid rgba(255,255,255,0.6);

    transition: 0.3s ease;
}

.card:hover {
    transform: translateY(-5px);

    box-shadow:
        0 15px 35px rgba(0,0,0,0.12);
}


/* ===== Buttons ===== */
.stButton > button {

    background:
    linear-gradient(
        90deg,
        #00b4db 0%,
        #0083b0 100%
    );

    color:white;

    border-radius:50px;

    padding:
    0.7rem 2.5rem;

    font-size:1.1rem;

    font-weight:700;

    border:none;

    transition:0.3s ease;
}


.stButton > button:hover {

    transform:
    translateY(-3px);

    box-shadow:
    0 8px 20px rgba(0,180,219,0.4);

}


/* ===== Metrics ===== */

.metric-card {

    background:
    linear-gradient(
        135deg,
        #ffffff,
        #f0fbff
    );

    padding:1.5rem;

    border-radius:20px;

    text-align:center;

    box-shadow:
    0 6px 18px rgba(0,0,0,0.08);

    border-left:
    5px solid #00b4db;

}


/* ===== Sidebar ===== */

section[data-testid="stSidebar"] {

    background:
    linear-gradient(
        180deg,
        #0b3d5c,
        #145374
    );

}


section[data-testid="stSidebar"] * {

    color:white;

}


/* ===== Input Box ===== */

div[data-baseweb="input"] input {

    border-radius:12px;

}


/* ===== Success Alert ===== */

.stSuccess {

    border-radius:15px;

}


/* ===== Dataframe ===== */

.stDataFrame {

    border-radius:15px;

    overflow:hidden;

}

</style>
""", unsafe_allow_html=True)

# ============ Lottie ============
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None

lottie_hello = load_lottieurl("https://lottie.host/3f647b41-61bf-4d39-93c3-0433420604cc/8NtmEbAWmO.json")
lottie_download = load_lottieurl("https://lottie.host/290016a6-d650-4a32-85c7-4d77b0a892ca/4rNL8XZeZt.json")

# ============ UI ============
st.markdown('<div class="main-header"><h1>💧 ระบบทำนายภาวะขาดน้ำ</h1>'
            '<h3>Dehydration Prediction System using SVM</h3></div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("👋 ยินดีต้อนรับ")
    st.write("""
    ระบบนี้ใช้ Machine Learning อัลกอริทึม **Support Vector Machine (SVM)** 
    วิเคราะห์ข้อมูลจาก **Kaggle Dataset** เพื่อทำนายภาวะขาดน้ำจากปัจจัยต่างๆ เช่น
    อายุ, น้ำหนัก, ปริมาณน้ำที่ดื่ม, ระดับกิจกรรม, และสภาพอากาศ
    """)

    # แสดงสถิติข้อมูล
    df = load_kaggle_data()
    
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""<div class='metric-card'>
            <h4>📊 จำนวนข้อมูล</h4>
            <h2>{len(df):,}</h2>
            <p>รายการ</p></div>""", unsafe_allow_html=True)
    with m2:
        dehydrated = df['dehydration_status'].sum()
        st.markdown(f"""<div class='metric-card'>
            <h4>💧 ขาดน้ำ</h4>
            <h2>{dehydrated:,}</h2>
            <p>{dehydrated/len(df)*100:.1f}%</p></div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class='metric-card'>
            <h4>🎯 ตัวแปร</h4>
            <h2>{len(df.columns)-1}</h2>
            <p>features</p></div>""", unsafe_allow_html=True)

    st.markdown("#### 🚀 เริ่มต้นใช้งาน")
    st.info("""
    - **📊 วิเคราะห์ข้อมูล**: สำรวจ EDA และสถิติ\n
    - **🔮 ทำนายภาวะขาดน้ำ**: กรอกข้อมูลเพื่อทำนาย\n
    - **📈 ประสิทธิภาพโมเดล**: ดู Accuracy, Confusion Matrix, ROC
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card" style="text-align:center;">', unsafe_allow_html=True)
    if lottie_hello:
        st_lottie(lottie_hello, key="hello", height=280)
    
    if st.button("🔄 รีเฟรชข้อมูล"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    
    st.markdown("---")
    st.caption("📦 Dataset: Daily Water Intake & Hydration Patterns")
    st.caption("🔗 Source: Kaggle")
    st.markdown('</div>', unsafe_allow_html=True)