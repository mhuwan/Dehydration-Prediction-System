import os
import subprocess
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC

KAGGLE_DATASET = "sonalshinde123/daily-water-intake-and-hydration-patterns-dataset"
LOCAL_CACHE_FILE = "data/hydration_data.csv"

@st.cache_data(show_spinner="📥 กำลังโหลดข้อมูลจาก Kaggle...")
def load_kaggle_data():
    os.makedirs('data', exist_ok=True)
    
    # 1. ใช้ไฟล์ Cache ถ้ามี
    if os.path.exists(LOCAL_CACHE_FILE):
        return pd.read_csv(LOCAL_CACHE_FILE)

    # 2. ลองดาวน์โหลดจาก Kaggle
    try:
        subprocess.run(
            ['kaggle', 'datasets', 'download', '-d', KAGGLE_DATASET, '-p', 'data', '--unzip', '--force'],
            check=True, capture_output=True
        )
        csv_files = [f for f in os.listdir('data') if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError("ไม่พบไฟล์ CSV")
        
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

def preprocess_kaggle_data(df):
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    
    col_mapping = {}
    for col in df.columns:
        if 'age' in col: col_mapping['age'] = col
        elif 'gender' in col or 'sex' in col: col_mapping['gender'] = col
        elif 'weight' in col: col_mapping['weight'] = col
        elif 'water' in col or 'intake' in col: col_mapping['water_intake'] = col
        elif 'activity' in col: col_mapping['activity'] = col
        elif 'weather' in col or 'temp' in col: col_mapping['weather'] = col
        elif 'hydration' in col or 'dehydrat' in col: col_mapping['hydration'] = col

    if 'hydration' not in col_mapping:
        df['dehydration_status'] = create_hydration_target(df)
    else:
        hydration_col = col_mapping['hydration']
        df['dehydration_status'] = df[hydration_col].apply(
            lambda x: 1 if str(x).lower() in ['poor', 'dehydrated', 'low', '1'] else 0
        )

    for key in ['gender', 'activity', 'weather']:
        if key in col_mapping:
            le = LabelEncoder()
            df[f'{key}_encoded'] = le.fit_transform(df[col_mapping[key]].astype(str))
            df = df.drop(columns=[col_mapping[key]])

    final_cols = ['age', 'weight', 'water_intake', 'gender_encoded', 'activity_encoded', 'weather_encoded', 'dehydration_status']
    df = df[[c for c in final_cols if c in df.columns]].copy()
    df = df.fillna(df.median(numeric_only=True))
    return df

def create_hydration_target(df):
    score = pd.Series(0, index=df.index)
    water_col = next((c for c in df.columns if 'water' in c.lower() or 'intake' in c.lower()), None)
    if water_col:
        score += (df[water_col] < 2.0).astype(int) * 3
        score += (df[water_col] < 1.5).astype(int) * 2
    return (score >= 4).astype(int)

def generate_synthetic_data(n_samples=3000):
    np.random.seed(42)
    df = pd.DataFrame({
        'age': np.random.randint(10, 80, n_samples),
        'weight': np.random.uniform(40, 100, n_samples),
        'water_intake': np.random.uniform(0.5, 4.5, n_samples),
        'gender_encoded': np.random.choice([0, 1], n_samples),
        'activity_encoded': np.random.choice([0, 1, 2], n_samples),
        'weather_encoded': np.random.choice([0, 1, 2], n_samples),
    })
    score = (df['water_intake'] < 2.0) * 3 + (df['activity_encoded'] == 2) * 2 + (df['weather_encoded'] == 0) * 2
    df['dehydration_status'] = (score >= 4).astype(int)
    return df

@st.cache_resource
def train_svm_model(df):
    X = df.drop('dehydration_status', axis=1)
    y = df['dehydration_status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, X_test, y_test, model.predict(X_test_scaled), X.columns