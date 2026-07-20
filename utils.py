import os
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==================== DATA LOADING ====================
@st.cache_data
def load_kaggle_data():
    """พยายามโหลดข้อมูลจาก Kaggle ถ้าไม่ได้จะใช้ข้อมูล Synthetic"""
    os.makedirs('data', exist_ok=True)
    file_path = 'data/dehydration_data.csv'
    
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    
    try:
        # ลองดึงจาก Kaggle (ต้องตั้งค่า kaggle.json ก่อน)
        import subprocess
        # ตัวอย่างคำสั่ง: เปลี่ยนชื่อ dataset ตามที่ต้องการ
        # subprocess.run(['kaggle', 'datasets', 'download', '-d', 'username/dataset-name', '-p', 'data', '--unzip'], check=True)
        # df = pd.read_csv(file_path) 
        # return df
        raise Exception("Not implemented")
    except Exception:
        # Fallback: สร้างข้อมูลตัวอย่างที่สมจริง (Simulate Kaggle Data)
        st.warning("⚠️ ไม่พบ Kaggle API หรือ Dataset กำลังใช้ข้อมูลจำลอง (Synthetic Data)")
        df = generate_synthetic_data()
        df.to_csv(file_path, index=False)
        return df

def generate_synthetic_data(n_samples=1500):
    """สร้างข้อมูลจำลองสำหรับฝึกโมเดล"""
    np.random.seed(42)
    data = {
        'age': np.random.randint(1, 85, n_samples),
        'temperature': np.random.uniform(36.0, 40.5, n_samples),
        'heart_rate': np.random.randint(55, 130, n_samples),
        'systolic_bp': np.random.randint(85, 170, n_samples),
        'urine_color': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.3, 0.3, 0.2, 0.1]), # 1=ใส, 5=เข้มมาก
        'skin_turgor': np.random.choice([0, 1, 2], n_samples), # 0=ปกติ, 2=ยืดตัวช้ามาก
        'thirst_level': np.random.randint(0, 11, n_samples),
        'sunken_eyes': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    }
    df = pd.DataFrame(data)
    
    # สร้าง Target (Dehydration) จากกฎทางการแพทย์
    score = (
        (df['temperature'] > 38.0) * 2 +
        (df['heart_rate'] > 100) * 2 +
        (df['systolic_bp'] < 100) * 2 +
        (df['urine_color'] >= 4) * 3 +
        (df['skin_turgor'] >= 1) * 2 +
        (df['thirst_level'] >= 7) * 2 +
        (df['sunken_eyes'] == 1) * 2
    )
    df['dehydration_status'] = (score >= 6).astype(int) # 0=ปกติ, 1=ขาดน้ำ
    return df

# ==================== MODEL TRAINING ====================
@st.cache_resource
def train_svm_model(df):
    """เทรนโมเดล SVM และคืนค่า Model, Scaler, และข้อมูล Test"""
    X = df.drop('dehydration_status', axis=1)
    y = df['dehydration_status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # SVM Model
    model = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    
    return model, scaler, X_test, y_test, y_pred, X.columns