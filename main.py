# This is a sample Python script.
import pickle

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
import sklearn

price_predictor_app = Flask(__name__)
dataset = pd.read_csv('Islamabad_Car_Prices_Dataset.csv')
model = pickle.load(open("LinearRegressionModel.pkl", "rb"))


@price_predictor_app.route('/')
def index():
    companies = sorted(dataset['Make'].astype(str).unique())
    models = sorted(dataset['Model'].astype(str).unique())
    seating = sorted(dataset['Seating'].astype(int).unique())
    engine = sorted(dataset['Engine'].astype(int).unique())

    companies.insert(0, "Select Company")

    return render_template('index.html', companies=companies, models=models, seating=seating, engine_type=engine)


@price_predictor_app.route('/predict', methods=['POST'])
def predict():
    # extracting received data
    company = request.form.get('company')
    car_model = request.form.get('model')
    engine = int(request.form.get('engine'))
    seating = int(request.form.get('seating'))
    year = int(request.form.get('year'))

    print(company, car_model, engine, seating, year)
    prediction = model.predict(pd.DataFrame([[company, car_model, year, engine, seating]],
                                            columns=['Make', 'Model', 'Make_Year', 'Engine', 'Seating']))
    print(prediction)
    return str(np.round(prediction[0], 2))


if __name__ == '__main__':
    price_predictor_app.run(debug=True)
