import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="FR-STMS: Smart Traffic Management",
    page_icon="🚦",
    layout="wide"
)

# Title
st.title("🚦 Festival & Rain Smart Traffic Management System (FR-STMS)")
st.markdown("_Real-time congestion prediction and adaptive signal control for Ludhiana_")

# Load and cache data
@st.cache_data
def load_data():
    df = pd.read_csv('My Dataset.csv', encoding='latin1')
    # Clean whitespace
    df['event_type'] = df['event_type'].astype(str).str.strip()
    df['season'] = df['season'].astype(str).str.strip()
    return df

@st.cache_resource
def train_model(df):
    # Encode congestion
    le = LabelEncoder()
    df['Congestion_Level'] = le.fit_transform(df['Traffic Congestion Level'])
    
    # Features
    feature_cols = ['Vehicle Count', 'Avg Speed (km/h)', 'Vehicle Density (%)',
                    'is_festival', 'is_rainy', 'season']
    X = df[feature_cols]
    y = df['Congestion_Level']
    
    # One-hot encode season
    X = pd.get_dummies(X, columns=['season'], drop_first=True)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, X.columns, accuracy, classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High'])

# Load data
df = load_data()

# Sidebar
st.sidebar.header("📊 Dashboard Controls")
sample_size = st.sidebar.slider("Sample Size for Prediction", 3, 10, 5)

# Train model
model, feature_names, accuracy, class_report = train_model(df)

# Main Metrics
col1, col2, col3 = st.columns(3)
col1.metric("✅ Model Accuracy", f"{accuracy * 100:.1f}%")
col2.metric("📈 Total Records", f"{len(df):,}")
col3.metric("🌧️ Rainy Samples", f"{df['is_rainy'].sum():,}")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🧠 Prediction", "📊 Insights", "📋 Report"])

# Tab 1: Overview
with tab1:
    st.subheader("Dataset Overview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Congestion Distribution
    fig1, ax1 = plt.subplots(figsize=(6, 3))
    df['Traffic Congestion Level'].value_counts().plot(kind='bar', ax=ax1, color=['#2ecc71', '#f39c12', '#e74c3c'])
    ax1.set_title("Traffic Congestion Distribution")
    ax1.set_ylabel("Count")
    st.pyplot(fig1)

# Tab 2: Real-time Prediction
with tab2:
    st.subheader("🚦 Real-Time Traffic Decision Dashboard")
    
    # Get random samples
    samples = df.sample(sample_size, random_state=42).reset_index(drop=True)
    
    for i, row in samples.iterrows():
        with st.container():
            st.markdown(f"### 🚗 Sample {i+1}: `{row['Timestamp']}`")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Vehicles", row['Vehicle Count'])
            col_b.metric("Avg Speed", f"{row['Avg Speed (km/h)']} km/h")
            col_c.metric("Density", f"{row['Vehicle Density (%)']}%")
            
            col_d, col_e, col_f = st.columns(3)
            col_d.write(f"**Festival**: {'Yes' if row['is_festival'] else 'No'}")
            col_e.write(f"**Rain**: {'Yes' if row['is_rainy'] else 'No'}")
            col_f.write(f"**Event**: {row['event_type']}")
            
            # Prepare input
            input_data = pd.DataFrame([{
                'Vehicle Count': row['Vehicle Count'],
                'Avg Speed (km/h)': row['Avg Speed (km/h)'],
                'Vehicle Density (%)': row['Vehicle Density (%)'],
                'is_festival': row['is_festival'],
                'is_rainy': row['is_rainy'],
                'season': row['season']
            }])
            input_data = pd.get_dummies(input_data, columns=['season'], drop_first=True)
            for col in feature_names:
                if col not in input_data.columns:
                    input_data[col] = 0
            input_data = input_data[feature_names]
            
            # Predict
            pred = model.predict(input_data)[0]
            prob = model.predict_proba(input_data)[0].max()
            congestion_map = {0: 'Low', 1: 'Medium', 2: 'High'}
            congestion = congestion_map[pred]
            
            # Recommendation
            base_green = 30
            if congestion == 'High':
                if row['is_festival'] and row['is_rainy']:
                    green_time = int(base_green * 1.8)
                    reason = "🚨 Festival + Rain: Extreme Congestion"
                elif row['is_festival']:
                    green_time = int(base_green * 1.5)
                    reason = "🎉 Festival Day"
                elif row['is_rainy']:
                    green_time = int(base_green * 1.4)
                    reason = "🌧️ Rainy Conditions"
                else:
                    green_time = int(base_green * 1.3)
                    reason = "⚠️ General High Congestion"
            elif congestion == 'Medium':
                green_time = int(base_green * 1.1)
                reason = "🟡 Moderate Traffic"
            else:
                green_time = base_green
                reason = "🟢 Smooth Flow"
            
            st.success(f"**Prediction**: {congestion} (Confidence: {prob*100:.1f}%)")
            st.info(f"**Recommendation**: {reason} → **Green Light: {green_time} seconds**")
            st.divider()

# Tab 3: Insights
with tab3:
    st.subheader("🔍 Key Insights")
    
    # Festival vs Non-Festival
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    festival_congestion = df.groupby('is_festival')['Traffic Congestion Level'].value_counts(normalize=True).unstack().fillna(0)
    festival_congestion.plot(kind='bar', ax=ax2, color=['#2ecc71', '#f39c12', '#e74c3c'])
    ax2.set_title("Congestion: Festival vs Non-Festival Days")
    ax2.set_xticklabels(['Non-Festival', 'Festival'], rotation=0)
    ax2.set_ylabel("Proportion")
    st.pyplot(fig2)
    
    # Rain Impact
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    rain_congestion = df.groupby('is_rainy')['Traffic Congestion Level'].value_counts(normalize=True).unstack().fillna(0)
    rain_congestion.plot(kind='bar', ax=ax3, color=['#2ecc71', '#f39c12', '#e74c3c'])
    ax3.set_title("Congestion: Rainy vs Dry Days")
    ax3.set_xticklabels(['Dry', 'Rainy'], rotation=0)
    ax3.set_ylabel("Proportion")
    st.pyplot(fig3)

# Tab 4: Model Report
with tab4:
    st.subheader("📋 Model Performance Report")
    st.code(f"Accuracy: {accuracy * 100:.2f}%", language="text")
    st.text("Classification Report:")
    st.text(class_report)

# Footer
st.markdown("---")
st.caption("Developed by Riya Sharma | M.Sc. Data Science, Chandigarh University")