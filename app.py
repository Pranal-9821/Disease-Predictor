import streamlit as st
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator

# --- 1. Page Configuration ---
st.set_page_config(page_title="AI Symptom Checker", layout="centered")

st.title("🩺 Medical Diagnosis Assistant")
st.write("Select your symptoms below to get the top 3 most likely diagnoses based on our Bayesian Network.")

# --- 2. Load Data and Train Model (Cached) ---
# The @st.cache_resource decorator ensures this heavy training 
# only happens once when the app starts, not on every click.
@st.cache_resource
def load_and_train_model():
    df = pd.read_csv('dataset/Training.csv')
    symptoms = [col for col in df.columns if col != 'prognosis']
    edges = [('prognosis', symptom) for symptom in symptoms]
    
    model = DiscreteBayesianNetwork(edges)
    model.fit(df, estimator=BayesianEstimator, prior_type='BDeu', equivalent_sample_size=10)
    
    return model, symptoms

# Show a loading spinner while the model trains the first time
with st.spinner("Booting up diagnostic engine..."):
    model, all_symptoms = load_and_train_model()

# --- 3. Clean UI Formatting ---
# Create a dictionary to map clean, readable symptom names back to the exact CSV columns
# Example: "skin_rash" -> "Skin Rash"
symptom_mapping = {s.replace('_', ' ').title(): s for s in all_symptoms}
clean_symptom_names = list(symptom_mapping.keys())

# --- 4. User Input (Multiselect) ---
st.subheader("Patient Symptoms")
selected_clean_symptoms = st.multiselect(
    "What symptoms are you experiencing? (Search or select multiple)", 
    options=clean_symptom_names
)

# --- 5. Prediction Logic ---
if st.button("Generate Diagnosis", type="primary"):
    if not selected_clean_symptoms:
        st.warning("Please select at least one symptom from the list.")
    else:
        # Convert the user's clean UI selections back to the CSV column formats
        selected_internal_symptoms = [symptom_mapping[s] for s in selected_clean_symptoms]

        # Create a fresh patient profile with all 0s
        test_patient = {symptom: 0 for symptom in all_symptoms}
        
        # Turn on the symptoms the user selected (set to 1)
        for s in selected_internal_symptoms:
            test_patient[s] = 1

        patient_df = pd.DataFrame([test_patient])

        # Run the model
        probabilities = model.predict_probability(patient_df)
        
        # Normalize the probabilities
        raw_probs = probabilities.iloc[0]
        normalized_probs = raw_probs / raw_probs.sum()
        top_3 = normalized_probs.sort_values(ascending=False).head(3)

        # --- 6. Display Results ---
        st.divider()
        st.subheader("Top 3 Possible Diagnoses:")
        
        for disease, prob in top_3.items():
            # Clean up the output names (e.g., 'prognosis_Fungal infection' -> 'Fungal Infection')
            clean_disease_name = disease.replace('prognosis_', '').title()
            percentage = prob * 100
            
            # Display using Streamlit columns for a nice layout
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{clean_disease_name}**")
            with col2:
                st.write(f"{percentage:.2f}%")
            
            # Add a visual progress bar
            st.progress(float(prob))