import streamlit as st
import pandas as pd
import joblib
import json

# Load the model and column names
try:
    model = joblib.load('placement_model.pkl')
    with open('model_columns.json', 'r') as f:
        model_columns = json.load(f)
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()

# Create the Streamlit app
st.title("🎓 Campus Placement Predictor")

# Create the input form
with st.form("placement_form"):
    st.header("Student Details")
    
   
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        ssc_p = st.slider("SSC Percentage", 0, 100, 70)
        hsc_p = st.slider("HSC Percentage", 0, 100, 70)
        
    with col2:
        workex = st.selectbox("Work Experience", ["No", "Yes"])
        degree_p = st.slider("Degree Percentage", 0, 100, 70)
    
    
    hsc_s = st.selectbox("HSC Stream", ["Science", "Commerce", "Arts"])
    
   
    submit_button = st.form_submit_button("Predict Placement")

# form submission 
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
        
        # Create a DataFrame with all expected columns
        input_df = pd.DataFrame(columns=model_columns)
        input_df.loc[0] = 0 
        
        # Fill in the provided values
        for col, value in input_data.items():
            if col in input_df.columns:
                input_df[col] = value
        
        
        proba = model.predict_proba(input_df)[0][1]  # Probability of "Placed"
        
        
        st.subheader("Prediction Result")
        
        
        display_prob = max(0.05, min(0.95, proba)) 
        
        st.metric("Placement Probability", f"{display_prob:.1%}")
        st.progress(int(display_prob * 100))
        
        
        if display_prob > 0.7:
            st.success("High chance of placement")
        elif display_prob > 0.4:
            st.warning("Moderate chance of placement")
        else:
            st.error("Low chance of placement")
