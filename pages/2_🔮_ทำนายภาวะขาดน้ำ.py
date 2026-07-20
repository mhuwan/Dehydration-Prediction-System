import streamlit as st
import pandas as pd
import numpy as np
from utils import load_kaggle_data, train_svm_model
from streamlit_lottie import st_lottie
import requests

st.set_page_config(page_title="ทำนายผล", page_icon="🔮", layout="wide")
st.markdown("# 🔮 ทำนายภาวะขาดน้ำ (SVM Prediction)")

# โหลด Lottie
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200: return None
    return r.json()

lottie_success = load_lottieurl("https://lottie.host/45b560d6-02c4-4801-80c8-328292991192/1TqR7yZJ7z.json") # Green check
lottie_danger = load_lottieurl("https://lottie.host/0b4b9a8e-5c8e-4b8c-9c8e-5c8e4b8c9c8e/1TqR7yZJ7z.json") # Red cross (ใช้ URL ตัวอย่าง)

df = load_kaggle_data()
model, scaler, _, _, _, feature_names = train_svm_model(df)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📝 กรอกข้อมูลอาการผู้ป่วย")

col1, col2, col3 = st.columns(3)
with col1:
    age = st.slider("อายุ (ปี)", 0, 100, 30)
    temperature = st.slider("อุณหภูมิร่างกาย (°C)", 35.0, 42.0, 37.0, 0.1)
    heart_rate = st.slider("อัตราการเต้นหัวใจ (bpm)", 40, 150, 80)
with col2:
    systolic_bp = st.slider("ความดัน Systolic (mmHg)", 70, 200, 120)
    urine_color = st.selectbox("สีปัสสาวะ (1=ใส, 5=เข้มมาก)", [1, 2, 3, 4, 5])
    skin_turgor = st.selectbox("ความยืดหยุ่นผิวหนัง (0=ปกติ, 2=ช้ามาก)", [0, 1, 2])
with col3:
    thirst_level = st.slider("ระดับความกระหาย (0-10)", 0, 10, 5)
    sunken_eyes = st.selectbox("ตาโหลลึก (0=ไม่, 1=ใช่)", [0, 1])

if st.button("🔮 ทำนายผลทันที", type="primary", use_container_width=True):
    input_data = pd.DataFrame({
        'age': [age], 'temperature': [temperature], 'heart_rate': [heart_rate],
        'systolic_bp': [systolic_bp], 'urine_color': [urine_color],
        'skin_turgor': [skin_turgor], 'thirst_level': [thirst_level], 'sunken_eyes': [sunken_eyes]
    })
    
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    st.markdown("---")
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        if prediction == 1:
            st.error("⚠️ ผลทำนาย: มีความเสี่ยงขาดน้ำ")
            st.write(f"ความมั่นใจ: **{probability[1]*100:.2f}%**")
        else:
            st.success("✅ ผลทำนาย: ร่างกายปกติ")
            st.write(f"ความมั่นใจ: **{probability[0]*100:.2f}%**")
            
    with res_col2:
        prob_df = pd.DataFrame({
            'สถานะ': ['ปกติ', 'ขาดน้ำ'],
            'ความน่าจะเป็น (%)': [probability[0]*100, probability[1]*100]
        })
        fig_bar = px.bar(prob_df, x='สถานะ', y='ความน่าจะเป็น (%)', color='สถานะ',
                         color_discrete_map={'ปกติ':'#8ec5fc', 'ขาดน้ำ':'#f093fb'})
        st.plotly_chart(fig_bar, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)