from sys import stderr

from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

Random_model = pickle.load(open('Random_model.pkl', 'rb'))
col=['lotsize','bedrooms','bathrooms','garpl','heating','Pool','Basement','Stories','airco','pavement','prefarea']

@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final=[np.array(int_features, dtype=float)]
    prediction=Random_model.predict(final)
    output=round(prediction[0],2)

    return render_template('output.html', pred=' {} $.'.format(output))

if __name__ == '__main__':
    app.run(debug=True)
