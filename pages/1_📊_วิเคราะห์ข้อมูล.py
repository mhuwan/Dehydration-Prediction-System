import streamlit as st
import plotly.express as px
from utils import load_kaggle_data

st.set_page_config(page_title="วิเคราะห์ข้อมูล", page_icon="📊", layout="wide")
st.markdown("# 📊 วิเคราะห์และสำรวจข้อมูล (EDA)")

df = load_kaggle_data()

with st.expander("🔍 ดูข้อมูลดิบ (Raw Data)"):
    st.dataframe(df.head(50), use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    fig_pie = px.pie(df, names='dehydration_status', title='สัดส่วนภาวะขาดน้ำ', color='dehydration_status', color_discrete_map={0:'#8ec5fc', 1:'#f093fb'}, hole=0.4)
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    fig_hist = px.histogram(df, x='age', color='dehydration_status', nbins=30, title='การกระจายของอายุ', color_discrete_map={0:'#8ec5fc', 1:'#f093fb'}, barmode='overlay')
    st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("### 🌡️ Correlation Heatmap")
numeric_df = df.select_dtypes(include=['number'])
fig_corr = px.imshow(numeric_df.corr(), text_auto=".2f", color_continuous_scale='RdBu_r', aspect="auto")
st.plotly_chart(fig_corr, use_container_width=True)

if 'water_intake' in df.columns:
    st.markdown("### 📦 Boxplot: ปริมาณน้ำที่ดื่ม vs ภาวะขาดน้ำ")
    fig_box = px.box(df, x='dehydration_status', y='water_intake', color='dehydration_status', title='เปรียบเทียบปริมาณน้ำที่ดื่ม', color_discrete_map={0:'#8ec5fc', 1:'#f093fb'})
    st.plotly_chart(fig_box, use_container_width=True)