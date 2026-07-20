import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.svm import LinearSVC
from utils import load_kaggle_data, train_svm_model

st.set_page_config(page_title="ประสิทธิภาพโมเดล", page_icon="📈", layout="wide")
st.markdown("# 📈 ประสิทธิภาพโมเดล SVM")

df = load_kaggle_data()
model, scaler, X_test, y_test, y_pred, feature_names = train_svm_model(df)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🎯 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['ปกติ', 'ขาดน้ำ'],
                yticklabels=['ปกติ', 'ขาดน้ำ'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    st.pyplot(fig)

with col2:
    st.markdown("### 📊 Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.background_gradient(cmap='Blues'),
                 use_container_width=True)

# ROC Curve
st.markdown("### 📈 ROC Curve")
y_prob = model.predict_proba(scaler.transform(X_test))[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

fig_roc = px.area(
    x=fpr, y=tpr,
    title=f'ROC Curve (AUC = {roc_auc:.4f})',
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
    width=800, height=500
)
fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
st.plotly_chart(fig_roc, use_container_width=True)

# Feature Importance
st.markdown("### 🧬 ความสำคัญของตัวแปร (Feature Importance)")
try:
    X_full = df.drop('dehydration_status', axis=1)
    y_full = df['dehydration_status']
    linear_model = LinearSVC(random_state=42, max_iter=10000)
    linear_model.fit(scaler.transform(X_full), y_full)

    importance = pd.DataFrame({
        'Feature': X_full.columns,
        'Importance': abs(linear_model.coef_[0])
    }).sort_values('Importance', ascending=False)

    fig_imp = px.bar(
        importance, x='Importance', y='Feature', orientation='h',
        color='Importance', color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig_imp, use_container_width=True)
except Exception as e:
    st.warning(f"ไม่สามารถคำนวณ Feature Importance ได้: {e}")

# Model Info
st.markdown("### ⚙️ ข้อมูลโมเดล")
st.json({
    'Algorithm': 'Support Vector Machine (SVM)',
    'Kernel': model.kernel,
    'C': model.C,
    'Gamma': str(model.gamma),
    'Support Vectors': model.n_support_.tolist(),
    'Training Samples': len(df) - len(X_test),
    'Testing Samples': len(X_test)
})