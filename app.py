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
    DailyTimeSpentonSite = sorted(car[' Daily Time Spent on Site'].unique())
    AreaIncome = sorted(car['Area Income'].unique())
    age = sorted(car['Age'].unique(), reverse=True)
    DailyInternetUsage = car['Daily Internet Usage'].unique()
    Male = car['Male'].unique()

    return render_template('index.html', DailyTimeSpentonSite=DailyTimeSpentonSite, AreaIncomes=AreaIncome, age=age, DailyInternetUsage=DailyInternetUsage, male=Male)


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    DailyTimeSpentonSite = request.form.get('DailyTimeSpentOnSite')
    AreaIncome = request.form.get('AreaIncome')
    age = request.form.get('age')
    DailyInternetUsage = request.form.get('DailyInternetUsage')
    Male = request.form.get('Male')

    prediction = model.predict(pd.DataFrame(columns=['Daily Time Spent on Site', 'Area Income', 'Age', 'Daily Internet Usage', 'Male'], data=np.array([DailyTimeSpentonSite, AreaIncome, age, DailyInternetUsage, Male]).reshape(1, 5)))
    print(prediction)
    return str(np.round(prediction[0], 2))

if __name__ == '_main_':
    app.run(debug=True)