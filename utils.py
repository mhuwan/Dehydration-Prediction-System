import os
import subprocess
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC

# ============================================================
# 🔹 ชื่อ Dataset จาก Kaggle
# ============================================================
KAGGLE_DATASET = "sonalshinde123/daily-water-intake-and-hydration-patterns-dataset"
LOCAL_CACHE_FILE = "data/hydration_data.csv"


# ============================================================
# 🔹 โหลดข้อมูลจาก Kaggle
# ============================================================
@st.cache_data(show_spinner="📥 กำลังโหลดข้อมูลจาก Kaggle...")
def load_kaggle_data():
    """โหลดข้อมูลจริงจาก Kaggle พร้อมระบบ Fallback"""
    os.makedirs('data', exist_ok=True)

    # 1. ถ้ามีไฟล์ cache แล้ว ให้ใช้เลย
    if os.path.exists(LOCAL_CACHE_FILE):
        df = pd.read_csv(LOCAL_CACHE_FILE)
        return df

    # 2. ลองดาวน์โหลดจาก Kaggle
    try:
        subprocess.run(
            ['kaggle', 'datasets', 'download', '-d', KAGGLE_DATASET,
             '-p', 'data', '--unzip', '--force'],
            check=True, capture_output=True
        )

        # ค้นหาไฟล์ CSV ในโฟลเดอร์ data
        csv_files = [f for f in os.listdir('data') if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError("ไม่พบไฟล์ CSV หลังดาวน์โหลด")

        df = pd.read_csv(f"data/{csv_files[0]}")
        df = preprocess_kaggle_data(df)
        df.to_csv(LOCAL_CACHE_FILE, index=False)
        return df

    except Exception as e:
        st.warning(f"⚠️ ไม่สามารถโหลดจาก Kaggle ได้: {e}")
        st.info("🔄 ใช้ข้อมูลจำลอง (Synthetic Data) แทน")
        df = generate_synthetic_data()
        df.to_csv(LOCAL_CACHE_FILE, index=False)
        return df


# ============================================================
# 🔹 ปรับแต่งข้อมูลจาก Kaggle
# ============================================================
def preprocess_kaggle_data(df):
    """แปลงข้อมูลดิบจาก Kaggle ให้พร้อมใช้"""
    # แปลง column names ให้เป็นตัวพิมพ์เล็กและไม่มีช่องว่าง
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

    # หา column ที่ต้องการ
    col_mapping = {}
    for col in df.columns:
        if 'age' in col: col_mapping['age'] = col
        elif 'gender' in col or 'sex' in col: col_mapping['gender'] = col
        elif 'weight' in col: col_mapping['weight'] = col
        elif 'water' in col or 'intake' in col: col_mapping['water_intake'] = col
        elif 'activity' in col: col_mapping['activity'] = col
        elif 'weather' in col or 'temperature' in col or 'climate' in col: col_mapping['weather'] = col
        elif 'hydration' in col or 'dehydrat' in col: col_mapping['hydration'] = col

    # ถ้าไม่เจอ column hydration ให้สร้างจากกฎ
    if 'hydration' not in col_mapping:
        df['dehydration_status'] = create_hydration_target(df)
    else:
        # แปลง hydration level เป็น binary (0=ปกติ, 1=ขาดน้ำ)
        hydration_col = col_mapping['hydration']
        df['dehydration_status'] = df[hydration_col].apply(
            lambda x: 1 if str(x).lower() in ['poor', 'dehydrated', 'low', '1'] else 0
        )

    # Encode categorical columns
    for key in ['gender', 'activity', 'weather']:
        if key in col_mapping:
            col = col_mapping[key]
            le = LabelEncoder()
            df[f'{key}_encoded'] = le.fit_transform(df[col].astype(str))
            df = df.drop(columns=[col])

    # เลือกเฉพาะ columns ที่ต้องการ
    final_cols = ['age', 'weight', 'water_intake',
                  'gender_encoded', 'activity_encoded', 'weather_encoded',
                  'dehydration_status']
    
    available_cols = [c for c in final_cols if c in df.columns]
    df = df[available_cols].copy()

    # เติมค่า NaN
    df = df.fillna(df.median(numeric_only=True))

    return df


def create_hydration_target(df):
    """สร้าง target variable จากกฎทางการแพทย์"""
    score = pd.Series(0, index=df.index)
    
    water_col = None
    for col in df.columns:
        if 'water' in col.lower() or 'intake' in col.lower():
            water_col = col
            break
    
    if water_col:
        score += (df[water_col] < 2.0).astype(int) * 3
        score += (df[water_col] < 1.5).astype(int) * 2
    
    return (score >= 4).astype(int)


# ============================================================
# 🔹 ข้อมูลจำลอง (Fallback)
# ============================================================
def generate_synthetic_data(n_samples=3000):
    """สร้างข้อมูลจำลอง"""
    np.random.seed(42)
    df = pd.DataFrame({
        'age': np.random.randint(10, 80, n_samples),
        'weight': np.random.uniform(40, 100, n_samples),
        'water_intake': np.random.uniform(0.5, 4.5, n_samples),
        'gender_encoded': np.random.choice([0, 1], n_samples),
        'activity_encoded': np.random.choice([0, 1, 2], n_samples),
        'weather_encoded': np.random.choice([0, 1, 2], n_samples),
    })

    score = (
        (df['water_intake'] < 2.0) * 3 +
        (df['activity_encoded'] == 2) * 2 +
        (df['weather_encoded'] == 0) * 2 +
        (df['age'] > 60) * 1 +
        (df['weight'] > 80) * 1
    )
    df['dehydration_status'] = (score >= 5).astype(int)
    return df


# ============================================================
# 🔹 เทรนโมเดล SVM
# ============================================================
@st.cache_resource
def train_svm_model(df):
    """เทรน SVM"""
    X = df.drop('dehydration_status', axis=1)
    y = df['dehydration_status']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    return model, scaler, X_test, y_test, y_pred, X.columns