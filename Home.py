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

# ============ CSS (ปรับปรุงใหม่: แก้ปัญหาตัวหนังสือสีขาว + แต่งให้สวยพรีเมียม) ============
st.markdown("""
<style>
/* ===== 1. Global & Background ===== */
.stApp {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 50%, #ffffff 100%);
}

/* บังคับตัวหนังสือหลักให้เป็นสีเข้ม อ่านง่าย สบายตา */
.main .block-container, 
p, h1, h2, h3, h4, h5, h6, label, span, div {
    color: #1e293b !important; /* สีเทาเข้มเกือบดำ (Slate 800) */
}

/* ===== 2. Header ===== */
.main-header {
    text-align: center;
    padding: 2rem 0 1.5rem 0;
}

.main-header h1 {
    font-size: 3rem;
    font-weight: 900;
    color: #0369a1 !important; /* สีน้ำเงินเข้มสวยงาม */
    letter-spacing: 1px;
    text-shadow: 2px 3px 8px rgba(3, 105, 161, 0.15);
    margin-bottom: 0.5rem;
}

.main-header h3 {
    color: #0284c7 !important; /* สีฟ้าสดใส */
    font-size: 1.2rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}

/* ===== 3. Card (Glassmorphism) ===== */
.card {
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    padding: 2rem;
    border-radius: 24px;
    box-shadow: 0 8px 32px rgba(31, 38, 135, 0.07);
    border: 1px solid rgba(255, 255, 255, 0.8);
    transition: all 0.3s ease;
}

.card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(31, 38, 135, 0.12);
    border: 1px solid rgba(3, 105, 161, 0.2);
}

/* ตัวหนังสือใน Card ต้องเข้มชัดเจน */
.card h4, .card p, .card h2, .card div {
    color: #0f172a !important;
}

/* ===== 4. Metric Cards ===== */
.metric-card {
    background: linear-gradient(135deg, #ffffff, #f0f9ff);
    padding: 1.5rem;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    border-left: 5px solid #0ea5e9;
    transition: transform 0.2s ease;
}

.metric-card:hover {
    transform: scale(1.03);
}

.metric-card h4 {
    color: #64748b !important;
    font-size: 0.9rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.metric-card h2 {
    color: #0369a1 !important;
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0.2rem 0;
}

.metric-card p {
    color: #94a3b8 !important;
    font-size: 0.85rem;
    margin: 0;
}

/* ===== 5. Buttons ===== */
.stButton > button {
    background: linear-gradient(90deg, #0ea5e9 0%, #0284c7 100%);
    color: #ffffff !important;
    border-radius: 50px;
    padding: 0.7rem 2.5rem;
    font-size: 1.05rem;
    font-weight: 700;
    border: none;
    box-shadow: 0 4px 15px rgba(14, 165, 233, 0.3);
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(14, 165, 233, 0.4);
    background: linear-gradient(90deg, #0284c7 0%, #0369a1 100%);
}

/* ===== 6. Sidebar (แยกธีมชัดเจน: พื้นหลังเข้ม ตัวหนังสือขาว) ===== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e3a8a 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.1);
}

/* บังคับเฉพาะใน Sidebar ให้เป็นตัวหนังสือสีขาว */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div {
    color: #ffffff !important;
}

/* ปรับแต่ง Input ใน Sidebar ให้สวยขึ้น */
section[data-testid="stSidebar"] input, 
section[data-testid="stSidebar"] select {
    background-color: rgba(255, 255, 255, 0.1) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    color: #ffffff !important;
    border-radius: 8px !important;
}

/* ===== 7. Utilities ===== */
.stSuccess {
    background-color: #dcfce7 !important;
    color: #166534 !important;
    border-radius: 12px;
    border: 1px solid #86efac;
}

.stInfo {
    background-color: #e0f2fe !important;
    color: #075985 !important;
    border-radius: 12px;
    border: 1px solid #7dd3fc;
}

.stDataFrame {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

# ============ Lottie Animation ============
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

lottie_hello = load_lottieurl("https://lottie.host/3f647b41-61bf-4d39-93c3-0433420604cc/8NtmEbAWmO.json")
lottie_download = load_lottieurl("https://lottie.host/290016a6-d650-4a32-85c7-4d77b0a892ca/4rNL8XZeZt.json")

# ============ Main UI ============
st.markdown(
    '<div class="main-header">'
    '<h1>💧 ระบบทำนายภาวะขาดน้ำ</h1>'
    '<h3>Dehydration Prediction System using SVM</h3>'
    '</div>', 
    unsafe_allow_html=True
)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("👋 ยินดีต้อนรับสู่ระบบ")
    st.write("""
    ระบบนี้พัฒนาด้วย Machine Learning อัลกอริทึม **Support Vector Machine (SVM)** 
    เพื่อวิเคราะห์และทำนายภาวะขาดน้ำจากข้อมูลสุขภาพจริง (Kaggle Dataset) 
    โดยพิจารณาจากปัจจัยสำคัญ เช่น อายุ, น้ำหนัก, ปริมาณน้ำที่ดื่ม, ระดับกิจกรรม, และสภาพอากาศ
    """)

    # โหลดและแสดงสถิติข้อมูล
    with st.spinner("🔄 กำลังโหลดข้อมูล..."):
        df = load_kaggle_data()
    
    if df is not None and not df.empty:
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""<div class='metric-card'>
                <h4>📊 จำนวนข้อมูล</h4>
                <h2>{len(df):,}</h2>
                <p>รายการ</p></div>""", unsafe_allow_html=True)
        with m2:
            dehydrated = int(df['dehydration_status'].sum())
            st.markdown(f"""<div class='metric-card'>
                <h4>💧 เสี่ยงขาดน้ำ</h4>
                <h2>{dehydrated:,}</h2>
                <p>{(dehydrated/len(df)*100):.1f}% ของทั้งหมด</p></div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class='metric-card'>
                <h4>🎯 ตัวแปร (Features)</h4>
                <h2>{len(df.columns)-1}</h2>
                <p>ปัจจัยที่ใช้ทำนาย</p></div>""", unsafe_allow_html=True)

        st.markdown("#### 🚀 เริ่มต้นใช้งาน")
        st.info("""
        - **📊 วิเคราะห์ข้อมูล**: สำรวจกราฟ EDA และสถิติเชิงพรรณนา\n
        - **🔮 ทำนายภาวะขาดน้ำ**: กรอกข้อมูลสุขภาพเพื่อรับผลทำนายแบบ Real-time\n
        - **📈 ประสิทธิภาพโมเดล**: ตรวจสอบความแม่นยำ (Accuracy), Confusion Matrix และ ROC Curve
        """)
    else:
        st.error("❌ ไม่สามารถโหลดข้อมูลได้ กรุณาตรวจสอบการเชื่อมต่อหรือไฟล์ข้อมูล")

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card" style="text-align:center; display: flex; flex-direction: column; align-items: center; justify-content: center;">', unsafe_allow_html=True)
    
    if lottie_hello:
        st_lottie(lottie_hello, key="hello", height=250, speed=1)
    else:
        st.markdown("💧")
    
    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
    
    if st.button("🔄 รีเฟรชข้อมูลและโมเดล"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    
    st.markdown("---")
    st.caption("📦 Dataset: Daily Water Intake & Hydration Patterns")
    st.caption("🔗 Source: Kaggle")
    st.caption("⚙️ Model: Support Vector Machine (SVM)")
    st.markdown('</div>', unsafe_allow_html=True)