from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load your model
model = joblib.load("model_health_insurance.joblib")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = float(request.form['age'])
    bmi = float(request.form['bmi'])
    children = int(request.form['children'])
    smoker = request.form['smoker']
    sex = request.form['sex']
    region = request.form['region']

    input_data = pd.DataFrame([[age, bmi, children, smoker, sex, region]],
                              columns=['age', 'bmi', 'children', 'smoker', 'sex', 'region'])

    prediction = model.predict(input_data)[0]

    return render_template('index.html', prediction_text=f'Predicted Insurance Cost: ${prediction:.2f}')

if __name__ == "__main__":
    app.run(debug=True)
