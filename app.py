from flask import Flask, request, render_template
import pandas as pd
import pickle
import numpy as np

model = pickle.load(open('house.pkl', 'rb'))


app = Flask(__name__, template_folder='templates', static_url_path='')


@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'GET':
        return (render_template('main.html'))


@app.route('/add')
def add():
    return render_template('add.html')

@app.route('/estimate', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    features = [np.array(int_features)]
    prediction = model.predict(features)

    return render_template('main.html', prediction_text='Your estimated annual rent is ₦{}'.format(prediction))

if __name__ == '__main__':
    app.run(debug=True)