
import os
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "models/model.joblib"

st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="🛠️"
)

st.title("🛠️ Predictive Maintenance System")
st.write("Predict potential machine failure from sensor readings.")

if not os.path.exists(MODEL_PATH):
    st.error("Trained model not found. Upload models/model.joblib.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

machine_type = st.selectbox("Machine Type", ["L", "M", "H"])
air = st.number_input("Air Temperature (K)", value=298.5)
process = st.number_input("Process Temperature (K)", value=308.5)
speed = st.number_input("Rotational Speed (rpm)", value=1500)
torque = st.number_input("Torque (Nm)", value=40.0)
wear = st.number_input("Tool Wear (min)", value=100)

if st.button("Predict Machine Failure"):
    data = pd.DataFrame([{
        "Type": machine_type,
        "Air temperature [K]": air,
        "Process temperature [K]": process,
        "Rotational speed [rpm]": speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": wear
    }])

    prediction = int(model.predict(data)[0])
    probability = float(model.predict_proba(data)[0][1])

    if prediction == 1:
        st.error("⚠️ Potential Machine Failure Detected")
    else:
        st.success("✅ No Machine Failure Predicted")

    st.metric("Estimated Failure Probability", f"{probability:.2%}")
    st.caption(
        "This is a model estimate, not a calibrated real-world risk."
    )
