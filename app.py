import streamlit as st
import pandas as pd
import joblib
import json

st.set_page_config(page_title="Campus Placement Predictor", page_icon="🎓", layout="centered")

@st.cache_resource
def load_model():
    try:
        model = joblib.load('placement_model.pkl')
        with open('model_columns.json', 'r') as f:
            model_columns = json.load(f)
        return model, model_columns
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()

model, model_columns = load_model()

st.title("🎓 Campus Placement Predictor")
st.caption("Predict a student's placement chances based on academic and profile details.")

with st.form("placement_form"):
    st.header("📝 Enter Student Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        ssc_p = st.slider("SSC Percentage (10th Grade)", 0, 100, 70)
        hsc_p = st.slider("HSC Percentage (12th Grade)", 0, 100, 70)
        
    with col2:
        workex = st.selectbox("Work Experience", ["No", "Yes"])
        degree_p = st.slider("Degree Percentage (Bachelor's)", 0, 100, 70)
    
    hsc_s = st.selectbox("HSC Stream", ["Science", "Commerce", "Arts"])
    
    submit_button = st.form_submit_button("🔍 Predict Placement")

if submit_button:
    try:
        input_data = {
            'gender': 1 if gender == "Male" else 0,
            'ssc_p': ssc_p,
            'hsc_p': hsc_p,
            'degree_p': degree_p,
            'workex': 1 if workex == "Yes" else 0,
            'hsc_s_Science': 1 if hsc_s == "Science" else 0,
            'hsc_s_Commerce': 1 if hsc_s == "Commerce" else 0
        }
        
        input_df = pd.DataFrame(columns=model_columns)
        input_df.loc[0] = 0  
        
        for col, value in input_data.items():
            if col in input_df.columns:
                input_df.at[0, col] = value
        
        prediction = model.predict(input_df)[0]  
        
        st.subheader("🎯 Prediction Result")
        
        if prediction == 1:
            st.success("🎉 Congratulations! The student is likely to be *PLACED*.")
        else:
            st.error("😕 Unfortunately, the student is likely to *NOT get placed*.")
        
        if prediction == 0:
            st.info("🔔 Tip: Improving skills, certifications, or internships can boost placement chances!")

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.write("Debug info - Input DataFrame:", input_df)
        st.write("Model expects these columns:", model_columns)