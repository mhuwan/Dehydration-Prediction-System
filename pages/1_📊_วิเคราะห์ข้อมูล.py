import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utils import load_kaggle_data

st.set_page_config(page_title="วิเคราะห์ข้อมูล", page_icon="📊", layout="wide")
st.markdown("# 📊 วิเคราะห์และสำรวจข้อมูล (EDA)")

df = load_kaggle_data()

col1, col2 = st.columns(2)
with col1:
    st.markdown("### 🥧 สัดส่วนภาวะขาดน้ำ")
    fig_pie = px.pie(df, names='dehydration_status', title='สถานะภาวะขาดน้ำ',
                     color='dehydration_status', color_discrete_map={0:'#8ec5fc', 1:'#f093fb'},
                     hole=0.4)
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.markdown("### 🎂 การกระจายของอายุ")
    fig_hist = px.histogram(df, x='age', color='dehydration_status', nbins=30,
                            title='การกระจายของอายุแยกตามสถานะ',
                            color_discrete_map={0:'#8ec5fc', 1:'#f093fb'})
    st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("### 🌡️ ความสัมพันธ์ของตัวแปร (Correlation Heatmap)")
numeric_df = df.select_dtypes(include=['number'])
fig_corr = px.imshow(numeric_df.corr(), text_auto=".2f", color_continuous_scale='RdBu_r', aspect="auto")
st.plotly_chart(fig_corr, use_container_width=True)

st.markdown("### 📦 Boxplot: อัตราการเต้นของหัวใจ (Heart Rate)")
fig_box = px.box(df, x='dehydration_status', y='heart_rate', color='dehydration_status',
                 title='เปรียบเทียบ Heart Rate ระหว่างกลุ่มปกติและขาดน้ำ',
                 color_discrete_map={0:'#8ec5fc', 1:'#f093fb'})
st.plotly_chart(fig_box, use_container_width=True)

st.markdown("### 📋 ตัวอย่างข้อมูล (Data Preview)")
st.dataframe(df.head(10), use_container_width=True)