from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
import joblib as p
import pandas as pd
import numpy as np
app = Flask(__name__)
cors = CORS(app)
try:
    with open('LinearRegressionModel.pkl', 'rb') as file:
        model = p.load(file)
except Exception as e:
    print("Error loading the pickle file:", e)
try:
    car = pd.read_csv('advertising.csv')
except Exception as e:
    print("Error loading the CSV file:", e)
@app.route('/', methods=['GET', 'POST'])
def index():
    # Get unique values for dropdowns
    DailyTimeSpentonSite = sorted(car['Daily Time Spent on Site'].unique())
    AreaIncome = sorted(car['Area Income'].unique())
    age = sorted(car['Age'].unique(), reverse=True)
    DailyInternetUsage = car['Daily Internet Usage'].unique()
    Male = car['Male'].unique()

    return render_template('index1.html', DailyTimeSpentonSite=DailyTimeSpentonSite, AreaIncomes=AreaIncome, age=age, DailyInternetUsage=DailyInternetUsage, male=Male)
@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    # Get form data from the request
    DailyTimeSpentonSite = float(request.form.get('DailyTimeSpentOnSite'))
    AreaIncome = float(request.form.get('AreaIncome'))
    age = int(request.form.get('age'))
    DailyInternetUsage = float(request.form.get('DailyInternetUsage'))
    Male = int(request.form.get('Male'))

# Prepare input data and make predictions
    input_data = np.array([DailyTimeSpentonSite, AreaIncome, age, DailyInternetUsage, Male]).reshape(1, -1)
    prediction = model.predict(pd.DataFrame(columns=['Daily Time Spent on Site', 'Area Income', 'Age', 'Daily Internet Usage', 'Male'], data=input_data))

# Return the prediction as a response
    return str(np.round(prediction[0], 2))

if __name__ == '__main__':
    app.run()
