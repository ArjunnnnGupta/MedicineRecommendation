from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load data
data = pd.read_csv("Training.csv")
symptom_list = list(data.columns[:-1])

# Load mappings
precautions = pd.read_csv("precautions_df.csv")
precautions = precautions.drop(columns=[precautions.columns[0]])
diet = pd.read_csv("diets.csv")
desc = pd.read_csv("description.csv")
medications = pd.read_csv("medications.csv")

medications_map = dict(zip(medications['Disease'], medications['Medication']))
precautions_dict = dict(zip(precautions["Disease"], precautions.iloc[:, 1:].values.tolist()))
desc_map = dict(zip(desc['Disease'], desc['Description']))
diet_map = dict(zip(diet['Disease'], diet['Diet']))

# Encode labels
le = LabelEncoder()
data["prognosis"] = le.fit_transform(data["prognosis"])
X = data[symptom_list]
y = data["prognosis"]

# Train model
model = SVC(probability=True)
model.fit(X, y)

@app.route("/", methods=["GET", "POST"])
def index():
    disease = description = None
    medication = precautions_out = diet_out = []
    symptoms = []

    if request.method == "POST":
        for i in range(1, 5):
            symptom = request.form.get(f"symptom{i}")
            if symptom:
                symptoms.append(symptom)

        input_vector = [1 if s in symptoms else 0 for s in symptom_list]
        prediction = model.predict([input_vector])[0]
        disease = le.inverse_transform([prediction])[0]

        description = desc_map.get(disease, "No description available.")
        medication = medications_map.get(disease, "").split(', ')
        precautions_out = precautions_dict.get(disease, [])
        diet_raw = diet_map.get(disease, "")
        diet_out = [d.strip() for d in diet_raw.split(",")] if diet_raw else []

    return render_template("result.html",
                           symptom_options=symptom_list,
                           disease=disease,
                           description=description,
                           medication=medication,
                           precautions=precautions_out,
                           diet=diet_out,
                           request=request)

if __name__ == "__main__":
    app.run(debug=True)
