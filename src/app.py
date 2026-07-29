
import streamlit as st
import pandas as pd
import joblib
import yaml

st.set_page_config(page_title="Watch | Classification", layout="wide")
st.title("📊 Watch: Classification App")

try:
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    model = joblib.load(config['data']['model_path'])
except:
    st.error("Please run `python src/train.py` first.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    st.header("Inputs")
    f1 = st.slider("Feature 1", 0, 100, 50)
    f2 = st.slider("Feature 2", 0, 100, 50)
    f3 = st.slider("Feature 3", 0, 10, 3)
    if st.button("Predict"):
        input_df = pd.DataFrame([{'feature1': f1, 'feature2': f2, 'feature3': f3}])
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]
        st.session_state['pred'] = pred
        st.session_state['prob'] = prob

with col2:
    if 'pred' in st.session_state:
        st.header("Results")
        if st.session_state['pred'] == 1:
            st.error(f"Prediction: POSITIVE (Prob: {st.session_state['prob']:.2f})")
        else:
            st.success(f"Prediction: NEGATIVE (Prob: {st.session_state['prob']:.2f})")
