from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load your data
symptom_data = pd.read_csv('output.csv')
medicine_data = pd.read_csv('medicine.csv')

# Function to match symptoms and predict the most probable disease
def match_symptoms(input_symptoms):
    vectorizer = CountVectorizer().fit(symptom_data.columns[1:])
    symptom_vector = vectorizer.transform([input_symptoms])
    max_similarity = 0  
    probable_disease = None

    for i, row in symptom_data.iterrows():
        disease_symptoms = " ".join([symptom for symptom in symptom_data.columns[1:] if row[symptom] == 'yes'])
        disease_vector = vectorizer.transform([disease_symptoms])
        similarity = cosine_similarity(symptom_vector, disease_vector)
        if similarity > max_similarity:
            max_similarity = similarity
            probable_disease = row['Disease']

    return probable_disease

# Function to get remedies based on the disease
def get_medicine(disease):
    if disease in medicine_data['conditions'].values:
        remedies = medicine_data[medicine_data['conditions'] == disease]['remedies'].values[0]
        return remedies
    else:
        return "No remedies found for the disease in the database."

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_symptoms = request.form.get('symptoms')
        
        if input_symptoms:
            probable_disease = match_symptoms(input_symptoms)
            if probable_disease:
                remedy = get_medicine(probable_disease)
                return render_template('main.html', disease=probable_disease, remedy=remedy)
            else:
                return render_template('main.html', error="No matching disease found.")
        else:
            return render_template('main.html', error="Please enter symptoms.")

    return render_template('main.html')

if __name__ == "__main__":
    app.run(debug=False)
