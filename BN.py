import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator

# 1. Load the dataset
df = pd.read_csv('dataset/Training.csv')

# 2. Define the structure
symptoms = [col for col in df.columns if col != 'prognosis']

# Create the Naive Bayes structure manually (Disease -> Symptom)
edges = [('prognosis', symptom) for symptom in symptoms]

# Initialize the discrete network with our custom edges
model = DiscreteBayesianNetwork(edges)

# 3. Train the network (Learn the Conditional Probability Tables)
print("Training the model with Laplace Smoothing...")
model.fit(df, 
          estimator=BayesianEstimator, 
          prior_type='BDeu', 
          equivalent_sample_size=10)
print("Training complete!")

# 4. Make a Prediction!
# Let's create a hypothetical patient who only has 'itching' and 'skin_rash'
test_patient = {symptom: 0 for symptom in symptoms} # Set all symptoms to 0
test_patient['itching'] = 1                         # Turn on itching
test_patient['skin_rash'] = 1                       # Turn on skin rash

# Convert to DataFrame as expected by pgmpy predict
patient_df = pd.DataFrame([test_patient])

# Predict the most likely disease
print("\nPredicting...")
prediction = model.predict(patient_df)
print(f"\nTop Predicted Disease: {prediction['prognosis'].values[0]}")

# 5. Get the exact probability percentages
probabilities = model.predict_probability(patient_df)

raw_probs = probabilities.iloc[0]


normalized_probs = raw_probs / raw_probs.sum()

# Sort them from highest to lowest and grab the top 3
top_3 = normalized_probs.sort_values(ascending=False).head(3)

print("\n--- TOP 3 DIAGNOSES ---")
for disease, prob in top_3.items():
    clean_name = disease.replace('prognosis_', '')
    print(f"{clean_name}: {prob * 100:.2f}%")