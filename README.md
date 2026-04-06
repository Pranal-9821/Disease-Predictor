# 🩺 AI Disease Predictor (Bayesian Diagnostic Engine)

An interactive, machine-learning-powered medical diagnostic tool built with Python and Streamlit. This application uses a **Discrete Bayesian Network** to analyze a patient's symptoms and calculate the mathematical probability of 41 different diseases.

---

## 🎯 The Core Problem & Solution
Standard probabilistic models often struggle with medical data due to the **Zero Frequency Problem** (where a single missing symptom in historical data collapses the entire probability calculation to exactly zero). 

**The Solution:** This engine utilizes the `pgmpy` library to construct a Bayesian Network and applies **Laplace Smoothing** via a Bayesian Estimator (BDeu Prior). This mathematically guarantees that rare or unusual symptom combinations do not break the model, allowing for robust, real-world probabilistic triage.

---

## ✨ Key Features
* **Probabilistic Triage:** Doesn't just guess a single disease—returns the top 3 most likely conditions with normalized percentage confidence scores.
* **Interactive Frontend:** Built with Streamlit for a clean, responsive, and user-friendly medical dashboard.
* **Smart Caching:** Model training is cached securely in memory (`@st.cache_resource`), resulting in instantaneous predictions after the initial boot-up.
* **Robust Math Engine:** Handles 132 distinct symptom variables in milliseconds.

---

## 🛠️ Tech Stack
* **Frontend UI:** Streamlit
* **Machine Learning:** `pgmpy` (Probabilistic Graphical Models in Python)
* **Data Processing:** Pandas
* **Language:** Python 3.8+

---

## ⚙️ Project Structure
```bash
Disease-Predictor/
├── .streamlit/
│   └── config.toml
├── dataset/
│   └── Training.csv
├── app.py
└── requirements.txt
```
## 🚀 Installation

### 1. Clone Repository
```bash
git clone https://github.com/Pranal-9821/Disease-Predictor.git
cd Disease-Predictor
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run App
```bash
streamlit run app.py
```