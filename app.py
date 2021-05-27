import numpy as np
from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

model = pickle.load(open('./salary_model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request.
    if request.method == "POST":
        #data = request.get_json(force=True)
        data = []
        output = {}
        for field in ['age', 'ctzn', 'gender', 'assoc', 'clic', 'hdgr', 'occfield', 'race']:
            value = request.form[field]
            print(value)
            data.append(int(value))
            output[field] = value

        # Make prediction  using model loaded from disk as per the data.
        prediction = model.predict([data])

        print("Data", prediction)

        # Take the first value of prediction
        output["output"] = round(prediction[0], 2)

        return render_template("results.html", **output)


if __name__ == '__main__':
    app.run(debug=True)
