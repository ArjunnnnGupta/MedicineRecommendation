Medicine Recommendation System

A Flask-based web application that predicts diseases based on user-input symptoms and recommends appropriate medications, precautions, diet plans, and disease descriptions using a trained Machine Learning model.

---

## 🚀 Features

- Predicts the most likely disease based on up to 4 symptoms
- Recommends:
  - ✅ Medications
  - 🍽️ Dietary guidelines
  - ⚠️ Precautions
  - 📄 Disease descriptions
- Real-time response via Flask backend and Jinja2 templating
- Uses Support Vector Machine (SVM) for accurate classification

---

## 🧠 ML Model

- **Algorithm**: Support Vector Classifier (SVC)
- **Dataset**: Symptom-disease mapping dataset (`Training.csv`)
- **Input Format**: Binary vector (1 for selected symptom, 0 otherwise)
- **Output**: Predicted disease label, mapped to domain data


##🗂️ Project Structure
MedicineRecommendation/
│
├── main.py # Flask backend and route handling
├── Training.csv # Symptom-disease training dataset
├── precautions_df.csv # Precautions for each disease
├── diets.csv # Diet recommendations
├── description.csv # Disease descriptions
├── medications.csv # Suggested medications
└── templates/
└── result.html # UI for symptom selection and results
demo-<img width="1693" height="724" alt="image" src="https://github.com/user-attachments/assets/a8a89881-33e4-4f63-92d6-22d819a5fb1b" />


