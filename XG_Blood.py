import streamlit as st
st.set_page_config(page_title="Smart Blood Test Interpreter", page_icon="🩺", layout="wide")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, hamming_loss, classification_report, f1_score
)
import plotly.express as px
import time

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Final_Dataset.csv")
    return df

df = load_data()

# App Header
st.title("🩺 Smart Blood Test Interpreter")
st.write("This app predicts multiple medical conditions based on patient blood test data using machine learning.")

# Feature/Target Split
X = df.iloc[:, :22]
y = df.iloc[:, 22:]

# Encode Categorical Features
X = pd.get_dummies(X, drop_first=True)

# Feature Selection
model_temp = XGBClassifier(n_estimators=50, use_label_encoder=False, eval_metric='logloss', random_state=42)
model_temp.fit(X, y)
importances = model_temp.feature_importances_
feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
top_features = feat_df.sort_values(by='Importance', ascending=False)['Feature'].head(20).tolist()
X = X[top_features]

# Add Gaussian Noise
for col in X.select_dtypes(include=[np.number]).columns:
    X[col] += np.random.normal(0, 0.1, X.shape[0])

# Train-Test Split + Scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Multi-label Model
base_model = XGBClassifier(
    n_estimators=30, max_depth=5, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    use_label_encoder=False, eval_metric='logloss', random_state=42
)
model = MultiOutputClassifier(base_model)
model.fit(X_train, y_train)

# Fixed Evaluation Metrics
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred) * 100
test_accuracy = accuracy_score(y_test, y_test_pred) * 100
hamming = hamming_loss(y_test, y_test_pred)
f1_micro = f1_score(y_test, y_test_pred, average='micro')
f1_macro = f1_score(y_test, y_test_pred, average='macro')

# Display Model Metrics
st.subheader("📊 XGBoost Model Performance")
with st.spinner('Evaluating model...'):
    time.sleep(1)

st.success(f"✅ Training Accuracy: {train_accuracy:.2f}%")
st.success(f"✅ Testing Accuracy: {test_accuracy:.2f}%")
st.info(f"🧹 Hamming Loss: {hamming:.4f}")
st.info(f"🔁 F1 Score (Micro): {f1_micro:.4f}")
st.info(f"🔁 F1 Score (Macro): {f1_macro:.4f}")

report = classification_report(y_test, y_test_pred, target_names=y.columns, zero_division=0)
st.code(report)

# Sidebar Inputs
st.sidebar.header("🔍 Enter Patient Data")
input_data = {}
for col in X.columns:
    input_data[col] = st.sidebar.number_input(col, value=float(df[col].mean()))

if st.sidebar.button("🧪 Predict"):
    input_df = pd.DataFrame([input_data])
    input_df = scaler.transform(input_df)

    with st.spinner('Analyzing patient data...'):
        y_probs = model.predict_proba(input_df)
        prob_dict = {}
        for i, col in enumerate(y.columns):
            prob_dict[col] = y_probs[i][0][1] if len(y_probs[i][0]) > 1 else y_probs[i][0][0]

        prediction_df = pd.DataFrame([prob_dict])
        time.sleep(1)

    st.write("### 🏥 Prediction Results")
    st.write("#### Probability of each condition:")
    fig = px.bar(prediction_df.T, x=prediction_df.columns, y=0,
                 labels={'x': 'Condition', 'y': 'Probability'},
                 title="Predicted Condition Probabilities")
    st.plotly_chart(fig)

    threshold = 0.5
    likely_conditions = prediction_df.T[prediction_df.T[0] >= threshold].index.tolist()
    if likely_conditions:
        st.success(f"🧬 Likely Condition(s) Detected: {', '.join(likely_conditions)}")
    else:
        st.warning("⚠ No condition detected with high confidence.")

    st.balloons()