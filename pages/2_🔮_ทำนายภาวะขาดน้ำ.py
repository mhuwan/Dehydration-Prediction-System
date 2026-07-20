import streamlit as st
import pandas as pd
import plotly.express as px
from utils import load_kaggle_data, train_svm_model

st.set_page_config(page_title="ทำนายผล", page_icon="🔮", layout="wide")
st.markdown("# 🔮 ทำนายภาวะขาดน้ำ (SVM Prediction)")

df = load_kaggle_data()
model, scaler, _, _, _, feature_names = train_svm_model(df)

st.markdown("### 📝 กรอกข้อมูลสุขภาพ")
col1, col2, col3 = st.columns(3)
with col1:
    age = st.slider("👤 อายุ (ปี)", 5, 100, 30)
    weight = st.slider("⚖️ น้ำหนัก (kg)", 30, 150, 65)
with col2:
    water_intake = st.slider("💧 น้ำที่ดื่ม (ลิตร/วัน)", 0.0, 5.0, 2.0, 0.1)
    gender = st.selectbox("🚻 เพศ", ["หญิง", "ชาย"])
with col3:
    activity = st.selectbox("🏃 ระดับกิจกรรม", ["น้อย", "ปานกลาง", "มาก"])
    weather = st.selectbox("🌤️ สภาพอากาศ", ["ร้อน", "ปกติ", "เย็น"])

gender_map = {"หญิง": 0, "ชาย": 1}
activity_map = {"น้อย": 0, "ปานกลาง": 1, "มาก": 2}
weather_map = {"ร้อน": 0, "ปกติ": 1, "เย็น": 2}

if st.button("🔮 ทำนายผลทันที", type="primary", use_container_width=True):
    input_data = pd.DataFrame([{
        'age': age, 'weight': weight, 'water_intake': water_intake,
        'gender_encoded': gender_map[gender], 'activity_encoded': activity_map[activity], 'weather_encoded': weather_map[weather],
    }])
    input_data = input_data[[c for c in feature_names if c in input_data.columns]]
    
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    st.markdown("---")
    res1, res2 = st.columns([1, 2])
    with res1:
        if prediction == 1:
            st.error("⚠️ **ผลทำนาย: มีความเสี่ยงขาดน้ำ**")
            st.metric("ความมั่นใจ", f"{probability[1]*100:.2f}%")
            st.warning("**คำแนะนำ:** ควรดื่มน้ำมากขึ้นอย่างน้อย 2-3 ลิตร/วัน")
        else:
            st.success("✅ **ผลทำนาย: ร่างกายปกติ**")
            st.metric("ความมั่นใจ", f"{probability[0]*100:.2f}%")
            st.info("**คำแนะนำ:** รักษาระดับการดื่มน้ำไว้")

    with res2:
        prob_df = pd.DataFrame({'สถานะ': ['ปกติ', 'ขาดน้ำ'], 'ความน่าจะเป็น (%)': [probability[0]*100, probability[1]*100]})
        fig_bar = px.bar(prob_df, x='สถานะ', y='ความน่าจะเป็น (%)', color='สถานะ', color_discrete_map={'ปกติ':'#8ec5fc', 'ขาดน้ำ':'#f093fb'}, text_auto='.2f')
        st.plotly_chart(fig_bar, use_container_width=True)