import time
import requests
import streamlit as st
from streamlit_lottie import st_lottie
from utils import load_kaggle_data

st.set_page_config(page_title="Dehydration Prediction", page_icon="💧", layout="wide", initial_sidebar_state="expanded")

# ============ CSS (แก้บั๊กตัวหนังสือสีขาว + ตกแต่งพรีเมียม) ============
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 50%, #ffffff 100%); }

/* บังคับตัวหนังสือหลักให้เป็นสีเข้ม อ่านง่าย 100% */
.main .block-container, p, h1, h2, h3, h4, h5, h6, label, span, div { color: #1e293b !important; }

.main-header { text-align: center; padding: 2rem 0 1.5rem 0; }
.main-header h1 { font-size: 3rem; font-weight: 900; color: #0369a1 !important; text-shadow: 2px 3px 8px rgba(3, 105, 161, 0.15); }
.main-header h3 { color: #0284c7 !important; font-size: 1.2rem; font-weight: 600; }

.card {
    background: rgba(255, 255, 255, 0.85); backdrop-filter: blur(12px);
    padding: 2rem; border-radius: 24px; box-shadow: 0 8px 32px rgba(31, 38, 135, 0.07);
    border: 1px solid rgba(255, 255, 255, 0.8); transition: all 0.3s ease;
    margin-bottom: 16px;
}
.card:hover { transform: translateY(-4px); box-shadow: 0 12px 40px rgba(31, 38, 135, 0.12); }

.metric-card {
    background: linear-gradient(135deg, #ffffff, #f0f9ff); padding: 1.5rem;
    border-radius: 20px; text-align: center; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    border-left: 5px solid #0ea5e9; transition: transform 0.2s ease;
}
.metric-card:hover { transform: scale(1.03); }
.metric-card h4 { color: #64748b !important; font-size: 0.9rem; font-weight: 600; text-transform: uppercase; }
.metric-card h2 { color: #0369a1 !important; font-size: 2.2rem; font-weight: 800; margin: 0.2rem 0; }
.metric-card p { color: #94a3b8 !important; font-size: 0.85rem; margin: 0; }

.stButton > button {
    background: linear-gradient(90deg, #0ea5e9 0%, #0284c7 100%); color: #ffffff !important;
    border-radius: 50px; padding: 0.7rem 2.5rem; font-weight: 700; border: none;
    box-shadow: 0 4px 15px rgba(14, 165, 233, 0.3); transition: all 0.3s ease;
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(14, 165, 233, 0.4); }

/* Sidebar แยกธีม: พื้นหลังเข้ม ตัวหนังสือขาว (ไม่กวนส่วนหลัก) */
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f172a 0%, #1e3a8a 100%) !important; }
section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, 
section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] span { color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except: return None

lottie_hello = load_lottieurl("https://lottie.host/3f647b41-61bf-4d39-93c3-0433420604cc/8NtmEbAWmO.json")

st.markdown('<div class="main-header"><h1>💧 ระบบทำนายภาวะขาดน้ำ</h1><h3>Dehydration Prediction System using SVM</h3></div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("👋 ยินดีต้อนรับสู่ระบบ")
    st.write("ระบบนี้พัฒนาด้วย Machine Learning อัลกอริทึม **Support Vector Machine (SVM)** เพื่อวิเคราะห์และทำนายภาวะขาดน้ำจากข้อมูลสุขภาพจริง โดยพิจารณาจากปัจจัยสำคัญ เช่น อายุ, น้ำหนัก, ปริมาณน้ำที่ดื่ม, ระดับกิจกรรม, และสภาพอากาศ")

    with st.spinner("🔄 กำลังโหลดข้อมูล..."):
        df = load_kaggle_data()
    
    if df is not None and not df.empty:
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"<div class='metric-card'><h4>📊 จำนวนข้อมูล</h4><h2>{len(df):,}</h2><p>รายการ</p></div>", unsafe_allow_html=True)
        with m2:
            dehydrated = int(df['dehydration_status'].sum())
            st.markdown(f"<div class='metric-card'><h4>💧 เสี่ยงขาดน้ำ</h4><h2>{dehydrated:,}</h2><p>{(dehydrated/len(df)*100):.1f}% ของทั้งหมด</p></div>", unsafe_allow_html=True)
        with m3:
            st.markdown(f"<div class='metric-card'><h4>🎯 ตัวแปร (Features)</h4><h2>{len(df.columns)-1}</h2><p>ปัจจัยที่ใช้ทำนาย</p></div>", unsafe_allow_html=True)

        st.markdown("#### 🚀 เริ่มต้นใช้งาน")
        st.info("- **📊 วิเคราะห์ข้อมูล**: สำรวจกราฟ EDA และสถิติ\n- **🔮 ทำนายภาวะขาดน้ำ**: กรอกข้อมูลสุขภาพเพื่อรับผลทำนาย\n- **📈 ประสิทธิภาพโมเดล**: ตรวจสอบความแม่นยำ (Accuracy) และ ROC Curve")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card" style="text-align:center; display: flex; flex-direction: column; align-items: center; justify-content: center;">', unsafe_allow_html=True)
    if lottie_hello: st_lottie(lottie_hello, key="hello", height=250)
    
    if st.button("🔄 รีเฟรชข้อมูลและโมเดล"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    
    st.markdown("---")
    st.caption("📦 Dataset: Daily Water Intake & Hydration Patterns")
    st.caption("🔗 Source: Kaggle | ⚙️ Model: SVM")
    st.markdown('</div>', unsafe_allow_html=True)